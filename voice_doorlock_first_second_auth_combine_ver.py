# 1차, 2차 인증과정 통합

import sys
import os
import random
import torch
import torchaudio
import librosa
import sounddevice as sd
import soundfile as sf
import numpy as np

#PyQt5 라이브러리
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QMessageBox, QInputDialog, QDialog, QHBoxLayout, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl, QSize
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QMovie
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from speechbrain.pretrained import SpeakerRecognition
import whisper
from sentence_transformers import SentenceTransformer

#nodeMCU WiFi 통신 라이브러리
import socket, time, logging

# ========== 설정 ==========
SAMPLE_RATE = 16000
RECORD_DURATION = 10
RECORD_SECONDS_LOGIN = 10
PROFILES_DIR = "profiles"
SIMILARITY_THRESHOLD = 0.6
ALPHA = 0.5
os.makedirs(PROFILES_DIR, exist_ok=True)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1234")

# ===== 로깅 설정 =====
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def _excepthook(exc_type, exc, tb):
    logging.error("Uncaught exception", exc_info=(exc_type, exc, tb))

import sys as _sys
_sys.excepthook = _excepthook

# AP 모드라면 기본 IP, STA 모드라면 nodeMCU 시리얼 모니터에 뜬 IP로 바꾸세요.
NODEMCU_HOST = "192.168.4.1"   # 또는 예:  "192.168.4.1"  "192.168.123.103"
NODEMCU_PORT = 7777

# ---- PIR 자동 트리거(UDP 수신) 설정 ----
UDP_LISTEN_PORT = 7788        # ESP 스케치의 PC_UDP_PORT와 동일해야 함
AUTO_TRIGGER = True           # True면 PIR(1) 수신 시 자동으로 인증 시작
AUTO_COOLDOWN_MS = 8000       # 자동 트리거 연속 호출 간 최소 간격(ms)

# ========== 모델 불러오기 ==========
# 화자 식별 ecapa_model 로딩
ecapa_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="ecapa_model"
)
# 음성 Pitch 모델 로딩
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.eval()

# 텍스트 - 텍스트 의미 유사도 비교 모델 로딩
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Whisper는 MPS 말고 CPU/CUDA만 사용 (MPS에서 sparse 에러 방지)
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()  # CUDA일 때만 fp16

whisper_model = whisper.load_model("medium", device=WHISPER_DEVICE)

# Silero VAD 모델 로딩
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
(get_speech_timestamps, _, _, _, collect_chunks) = utils  # save_audio/read_audio/VADIterator 미사용

# ========== 임베딩/비교 함수 ==========
def extract_pitch(y, sr, target_len=512):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    vals = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1])]
    vals = np.nan_to_num(vals)
    vals = librosa.util.fix_length(vals, size=target_len)
    return torch.tensor(vals).float().unsqueeze(0)

def get_ecapa_embedding(path):
    wav, sr = torchaudio.load(path)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    emb = ecapa_model.encode_batch(wav).squeeze(0)
    return emb.unsqueeze(0)

def get_wav2vec_pitch_embedding(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    inputs = wav2vec_processor(y, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        out = wav2vec_model(inputs)
    wav_emb = torch.mean(out.last_hidden_state, dim=1)
    pitch_emb = torch.mean(extract_pitch(y, sr), dim=1, keepdim=True) / 300.0
    return torch.cat((wav_emb, pitch_emb), dim=1)

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b).mean().item()

# ecapa 유사도 점수 + wav2vec_pitch 유사도 점수 (alpha값으로 가중치 조절)
def compare_with_ensemble(emb1_dir, test_audio, alpha=ALPHA):
    try:
        emb1_ecapa = torch.load(os.path.join(emb1_dir, "ecapa.pt")).unsqueeze(0)
        emb1_wav = torch.load(os.path.join(emb1_dir, "wav2vec.pt")).unsqueeze(0)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        logging.warning("Embedding load failed for %s: %s", emb1_dir, e)
        return -1e9  # 유효하지 않은 프로필로 간주

    emb2_ecapa = get_ecapa_embedding(test_audio)
    emb2_wav = get_wav2vec_pitch_embedding(test_audio)

    sim_ecapa = cosine_similarity(emb1_ecapa, emb2_ecapa)
    sim_wav = cosine_similarity(emb1_wav, emb2_wav)
    return alpha * sim_wav + (1 - alpha) * sim_ecapa

def update_profile_embeddings(profile_dir, new_audio_path):
    """인증 성공 시 프로필 임베딩 업데이트 (추가 → 평균 저장)"""
    try:
        # 기존 임베딩 불러오기
        ecapa_stack = torch.load(os.path.join(profile_dir, "ecapa.pt"))
        wav_stack   = torch.load(os.path.join(profile_dir, "wav2vec.pt"))

        # 새 오디오에서 임베딩 추출
        new_ecapa = get_ecapa_embedding(new_audio_path).squeeze(0)
        new_wav   = get_wav2vec_pitch_embedding(new_audio_path).squeeze(0)

        # (N, D) 형태로 누적→ 평균
        ecapa_new = torch.cat([ecapa_stack, new_ecapa.unsqueeze(0)], dim=0)
        wav_new   = torch.cat([wav_stack, new_wav.unsqueeze(0)], dim=0)

        # 평균 적용 후 저장 (학습 효과 안정화)
        torch.save(ecapa_new, os.path.join(profile_dir, "ecapa.pt"))
        torch.save(wav_new,   os.path.join(profile_dir, "wav2vec.pt"))

        print(f"[✅ Update] 프로필 데이터 갱신 완료 → {profile_dir}")

    except Exception as e:
        print(f"[⚠️ Update 실패] {e}")

# 의미 유사도를 계산하여 true, false 반환 (0.8 이상 = true)
def semantic_similarity(a: str, b: str, threshold: float = 0.8) -> bool:
    # 안전 처리
    a = a or ""
    b = b or ""

    # STT로 얻은 로그인 텍스트를 바로 출력
    print("------------------------------------------------------------------------------")
    print(f"[Whisper 모델] 사용자 로그인 음성 → 텍스트: {b}")
    print(f"[정답문장] 렌덤 제시 문장: {a}")
    print("------------------------------------------------------------------------------")

    embs = sbert.encode([a, b], convert_to_tensor=True)
    sim = torch.nn.functional.cosine_similarity(embs[0], embs[1], dim=0).item()
    print(f"[의미유사도] cos={sim:.3f}")
    return sim >= threshold

def send_nodemcu(cmd: str, host=NODEMCU_HOST, port=NODEMCU_PORT, timeout=1.5, read_reply=False, retries=3, backoff=0.3):
    """
    NodeMCU TCP 서버(7777)에 한 줄 명령을 보냅니다. 예외/타임아웃에 대해 재시도(backoff)하며,
    read_reply=True면 첫 라인을 반환(없으면 공백 문자열).
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with socket.create_connection((host, port), timeout=timeout) as s:
                s.sendall((cmd.strip() + "\n").encode("utf-8"))
                if read_reply:
                    s.settimeout(timeout)
                    try:
                        data = s.recv(1024).decode("utf-8", errors="ignore").strip()
                        return data
                    except socket.timeout:
                        logging.warning("send_nodemcu reply timeout: %s", cmd)
                        return ""
                return ""
        except (ConnectionRefusedError, TimeoutError, OSError, socket.error) as e:
            last_err = e
            logging.warning("send_nodemcu attempt %d/%d failed: %s", attempt, retries, e)
            time.sleep(backoff * attempt)
    raise RuntimeError(f"{last_err}")

# ───────── 통합 인증 워커 (녹음 + STT + 의미유사도 + 화자검증 + nodeMCU 전송) ─────────
class UnifiedAuthWorker(QThread):
    finished = pyqtSignal(bool, str, str)   # success, user(best match or ""), message
    recording_done = pyqtSignal()           # 🔔 "녹음 종료" 알림 신호 추가
    profile_register_requested = pyqtSignal()  # 🆕 추가


    def __init__(self, expected_sentence: str, profiles: list, attempts_left: int, parent=None):
        super().__init__(parent)
        self.expected_sentence = expected_sentence
        self.profiles = profiles
        self.attempts_left = attempts_left

    # 모든 인증 과정 종류 후 화면 초기화
    def reset_to_main_scene(self):
        # MainScene.gif의 첫 프레임을 띄운 채 정지
        main_movie = QMovie("gif/MainScene.gif")
        main_movie.jumpToFrame(0)
        # QMovie객체가 아닌 현재 프레임만 표시하려면 setPixmap
        self.label.setPixmap(main_movie.currentPixmap())
        # 다음번 재생을 위해 self.movie에도 저장
        self.movie = main_movie

        # 상태 초기화
        self.auth_worker = None                  # 혹시 남아 있는 워커 제거

    def run(self):
        # 1) 녹음
        try:
            ok = record_until_silence("auth.wav", RECORD_SECONDS_LOGIN)
        except Exception as e:
            logging.exception("UnifiedAuthWorker record error: %s", e)
            self.finished.emit(False, "", "녹음 실패")
            return
        if not ok:
            self.finished.emit(False, "", "음성 미감지")
            return

        # 🔔 녹음이 "정상 종료"되면 즉시 UI에 알림 → Find people.gif로 전환
        self.recording_done.emit()

        # 2) Whisper STT
        try:
            result = whisper_model.transcribe(
                "auth.wav",
                language="ko",
                temperature=0.0,
                beam_size=1, best_of=1,
                condition_on_previous_text=False,
                fp16=USE_FP16,
                initial_prompt="스마트 도어락, 인증, 날씨, 프로필, 추가, 등록, 생성"
            )
            spoken = result["text"].strip().lower()

            # 🆕 음성명령: "프로필 등록" 감지 시 즉시 이벤트 발생
            if "프로필 등록" in spoken or "프로필 추가" in spoken or "프로필 등록" in spoken:
                print("🎤 음성 명령: 프로필 등록 감지 → 팝업 요청")
                self.profile_register_requested.emit()
                return  # 인증 절차 중단
        except Exception as e:
            logging.exception("Whisper error: %s", e)
            self.finished.emit(False, "", "음성 인식 실패")
            return

        # 3) 의미 유사도
        try:
            sem_ok = semantic_similarity(self.expected_sentence.lower(), spoken)
        except Exception as e:
            logging.exception("Semantic compare error: %s", e)
            sem_ok = False

        # 4) 화자 유사도(등록 프로필 중 최고값)
        best_match, best_score = None, -1e9
        try:
            for p in self.profiles:
                profile_dir = os.path.join(PROFILES_DIR, p)
                score = compare_with_ensemble(profile_dir, "auth.wav", alpha=ALPHA)
                print(f"[유사도] {p}: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_match = p
            spk_ok = (best_score >= SIMILARITY_THRESHOLD)
        except Exception as e:
            logging.exception("Embedding compare error: %s", e)
            self.finished.emit(False, "", "프로필 비교 실패")
            return

        success = sem_ok and spk_ok
        if success:
            msg = f"{best_match}님 안녕하세요! 도어락이 열렸습니다"
            self.finished.emit(True, best_match, msg)
        else:
            if not sem_ok and not spk_ok:
                reason = "문장 불일치 + 등록되지 않은 음성\n"
            elif not sem_ok:
                reason = "문장 의미 불일치\n"
            else:
                reason = "등록되지 않은 음성"
            left = max(0, self.attempts_left - 1)
            msg = f"인증 실패: {reason} (남은 시도 {left}회)"
            self.finished.emit(False, best_match or "", msg)

class NodeMCUSeqWorker(QThread):
    error = pyqtSignal(str)

    def __init__(self, on1=1000, gap=5000, on2=1000,
                host=NODEMCU_HOST, port=NODEMCU_PORT, parent=None):
        super().__init__(parent)
        self.on1, self.gap, self.on2 = int(on1), int(gap), int(on2)
        self.host, self.port = host, port

    def run(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=3.0) as s:
                s.settimeout(1.5)

                def _send_and_try_read(cmd):
                    s.sendall((cmd.strip() + "\n").encode("utf-8"))
                    try:
                        _ = s.recv(256)  # 한 줄 정도만 받아 확인
                    except socket.timeout:
                        pass

                # 1) 가벼운 핸드셰이크
                _send_and_try_read("PING")

                # 2) 단일 SEQ 전송
                _send_and_try_read(f"SEQ {self.on1} {self.gap} {self.on2}")

        except Exception as e:
            self.error.emit(str(e))

# ========== 녹음 및 VAD 함수 ==========
def record_until_silence(path,
                        max_duration,
                        block_duration=0.5,
                        silence_blocks_thresh=2):
    """
    • with InputStream: 블록 종료 시점에 스트림이 자동 close → semaphore 누수 방지
    • block_duration 초씩 읽어서 VAD 검사
    • speech_started 후 silence_blocks_thresh 연속 무음 시 녹음 종료
    """
    blocks = []
    speech_started = False
    silence_blocks = 0
    max_blocks = int(max_duration / block_duration)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            for _ in range(max_blocks):
                audio, _ = stream.read(int(SAMPLE_RATE * block_duration))
                if audio is None or len(audio) == 0:
                    continue
                blocks.append(audio)

                wav = torch.from_numpy(audio.squeeze()).float()
                ts = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLE_RATE)
                if ts:
                    speech_started = True
                    silence_blocks = 0
                elif speech_started:
                    silence_blocks += 1

                if speech_started and silence_blocks >= silence_blocks_thresh:
                    break
    except Exception as e:
        logging.exception("Audio input error: %s", e)
        return False

    if not blocks:
        return False

    try:
        full = np.concatenate(blocks, axis=0)
        wav_full = torch.from_numpy(full.squeeze()).float()
        speech_ts = get_speech_timestamps(wav_full, vad_model, sampling_rate=SAMPLE_RATE)
        if not speech_ts:
            return False
        voiced = collect_chunks(speech_ts, wav_full)
        sf.write(path, voiced.numpy(), SAMPLE_RATE)
        return True
    except Exception as e:
        logging.exception("Post-record/VAD error: %s", e)
        return False

# UDP 리스너 워커 클래스
class UDPListenerWorker(QThread):
    """
    ESP가 보내는 'EVENT PIR 1/0' UDP 메시지를 수신해 motion(int) 시그널로 내보낸다.
    """
    motion = pyqtSignal(int)  # 1=감지, 0=해제

    def __init__(self, port=UDP_LISTEN_PORT, parent=None):
        super().__init__(parent)
        self.port = port
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.settimeout(0.5)  # 폴링 종료를 위해 짧은 타임아웃
        except Exception as e:
            logging.exception(f"UDP bind 실패: {e}")
            try: sock.close()
            except: pass
            return

        while not self._stop:
            try:
                data, _addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            except Exception as e:
                logging.warning(f"UDP recv 오류: {e}")
                continue

            try:
                line = data.decode("utf-8", errors="ignore").strip()
                # 기대형식: "EVENT PIR 1" or "EVENT PIR 0"
                if line.startswith("EVENT PIR"):
                    parts = line.split()
                    if len(parts) >= 3 and parts[2].isdigit():
                        level = int(parts[2])
                        self.motion.emit(level)
            except Exception as e:
                logging.warning(f"UDP 파싱 오류: {e}")

        try: sock.close()
        except: pass

class RecordingDialog_(QDialog):
    def __init__(self, sentence, record_func, path, duration):
        super().__init__()
        self.setWindowTitle("🎙 녹음 안내")
        self.setFixedSize(400, 200)
        self.record_func = record_func
        self.path = path
        self.duration = duration

        lay = QVBoxLayout()
        lbl = QLabel(f"📢 사용자의 이름만 또박또박 말해주세요.:\n예시) 박.성.재\n『 {sentence} 』")
        lbl.setStyleSheet("color: #000000; font-size: 20px;")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        btn = QPushButton("🎤 녹음 시작")
        btn.clicked.connect(self._do_record)
        lay.addWidget(btn)
        self.setLayout(lay)

    def _do_record(self):
        self.record_func(self.path, self.duration)
        self.accept()

# ========== 프로필 생성 녹음 안내 다이얼로그 ==========
class RecordingDialog(QDialog):
    def __init__(self, sentence, record_func, path, duration):
        super().__init__()
        self.setWindowTitle("🎙 녹음 안내")
        self.setFixedSize(400, 200)
        self.record_func = record_func
        self.path = path
        self.duration = duration

        lay = QVBoxLayout()
        lbl = QLabel(f"📢 다음 문장을 또박또박 읽어주세요:\n\n『 {sentence} 』")
        lbl.setStyleSheet("color: #000000; font-size: 20px;")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        btn = QPushButton("🎤 녹음 시작")
        btn.clicked.connect(self._do_record)
        lay.addWidget(btn)

        self.setLayout(lay)

    def _do_record(self):
        self.record_func(self.path, self.duration)
        self.accept()

# ========== 메인 UI ==========
class SmartDoorlockUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔐 스마트 도어락 시뮬레이터")
        self.setGeometry(1800, 0, 480, 750)    # 실제 소형 LCD 화면 비율과 유사하게
        self.setStyleSheet("background-color: white;")
        self.auth_fail_count = 0    # 실패 횟수 초기화
        self.attempts_left = 3      # 통합 인증 총 시도 횟수

        # ── 메인 레이아웃 ──
        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0,0,0,0)

        # ── 애니메이션 라벨 & 녹음중 레이블 준비 ──
        self.label = QLabel(self)
        self.label.setFixedSize(480, 480)
        self.label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie("gif/MainScene.gif")
        self.label.setFixedHeight(500)
        # 최초엔 재생하지 않고 첫 프레임만
        self.movie.jumpToFrame(0)
        self.label.setPixmap(self.movie.currentPixmap())
        main_lay.addWidget(self.label)

        # ── 통합 인증 문장 표시용 레이블 ──
        self.challenge_label = QLabel("", self)
        self.challenge_label.setAlignment(Qt.AlignCenter)
        self.challenge_label.setStyleSheet("color: #000000; font-size: 23px;")
        self.challenge_label.setFixedHeight(100)
        self.challenge_label.hide()
        main_lay.addWidget(self.challenge_label)

        # ── 미디어 플레이어 준비 ──
        self.player = QMediaPlayer(self)

        # ── 상태 메시지 레이블 추가 ──
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            color: #000000;
            font-size: 23px;
            padding: 8px;
        """)
        self.status_label.setFixedHeight(100)
        main_lay.addWidget(self.status_label)

        # ── “녹음중” 텍스트 레이블 ──
        self.recording_label = QLabel("녹음중…", self)
        self.recording_label.setAlignment(Qt.AlignCenter)
        self.recording_label.setStyleSheet("color: #000000; font-size: 23px;")
        self.recording_label.setFixedHeight(60)
        self.recording_label.hide()               # 초기에는 숨김
        main_lay.addWidget(self.recording_label)

        # ── 위젯들 사이의 남는 공간을 전부 차지할 스트레치 추가 ──
        main_lay.addStretch()

        # ── 버튼 레이아웃 ──
        hbtn = QHBoxLayout()
        self.detect_btn = QPushButton("🚶 사용자 접근 감지"); 
        self.detect_btn.clicked.connect(self.on_user_detected)
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
        """)
        hbtn.addWidget(self.detect_btn)
        main_lay.addLayout(hbtn)

        self.profiles = []
        self.load_profiles()

        self.lockdown_timer = QTimer(self)
        self.lockdown_timer.setInterval(1000)  # 1초
        self.lockdown_timer.timeout.connect(self._tick_lockdown)
        self.lockdown_remaining = 0

        self.is_auth_running = False
        self._last_auto_trigger_ms = 0
        self.auto_trigger_enabled = AUTO_TRIGGER
        self._auto_cooldown_ms = AUTO_COOLDOWN_MS

        self.udp_worker = UDPListenerWorker(port=UDP_LISTEN_PORT)
        self.udp_worker.motion.connect(self._on_motion_event)
        self.udp_worker.start()

    # 프로필 생성 (ecapa 임베딩 파일, wav2vec 임베딩 파일, 음성 녹음본 5개 생성하여 프로필에 저장)
    def create_profile(self):
        try:
            # --- 관리자 인증 ---
            pwd, ok = QInputDialog.getText(
                self, "관리자 인증", "관리자 비밀번호를 입력하세요:",
                QLineEdit.Password
            )
            if not ok:
                return
            if pwd != ADMIN_PASSWORD:
                QMessageBox.critical(self, "인증 실패", "관리자 비밀번호가 올바르지 않습니다.")
                self.recording_label.hide()
                self.reset_to_main_scene()
                return
            # --------------------
            
            # --- 음성으로 이름 입력 ---
            tmp_path = os.path.join(PROFILES_DIR, "tmp_name.wav")
            dialog = RecordingDialog_(
                "이름을 말해주세요!",
                record_until_silence,
                tmp_path,
                5  # 5초 녹음
            )
            dialog.exec_()

            # STT 변환
            try:
                result = whisper_model.transcribe(
                    tmp_path, language="ko",
                    temperature=0.0, beam_size=1, best_of=1,
                    condition_on_previous_text=False,
                    fp16=USE_FP16
                )
                name = result["text"].strip()
            except Exception as e:
                QMessageBox.critical(self, "오류", f"이름 음성 인식 실패:\n{e}")
                self.reset_to_main_scene()
                return

            # 이름 유효성 검사
            name = "".join(c for c in name if c.isalnum())  # 한글/영문/숫자만 허용
            if not name:
                QMessageBox.warning(self, "경고", "유효한 이름을 인식하지 못했습니다.")
                self.reset_to_main_scene()
                return
            profile_dir = os.path.join(PROFILES_DIR, name)
            if os.path.exists(profile_dir):
                QMessageBox.warning(self, "중복", "이미 존재하는 이름입니다.")
                self.reset_to_main_scene()
                return
            os.makedirs(profile_dir, exist_ok=True)

            # --- 프로필 녹음 문장 ---
            sentences = [
                "지금 들리는 이 목소리는 저만 사용할 수 있는 보안 열쇠입니다.",
                "스마트 도어 시스템을 통해 집에 안전하게 들어가고 싶습니다.",
                "이제 제 음성으로 문을 열 수 있는 시대가 왔습니다. 열어주세요."
            ]

            ecapa_embs, wav_embs = [], []
            for i, sentence in enumerate(sentences):
                rec_path = os.path.join(profile_dir, f"rec{i+1}.wav")
                dialog = RecordingDialog(
                    sentence,
                    record_until_silence,
                    rec_path,
                    RECORD_DURATION
                )
                dialog.exec_()

                ecapa_embs.append(get_ecapa_embedding(rec_path).squeeze(0))
                wav_embs.append(get_wav2vec_pitch_embedding(rec_path).squeeze(0))

            # --- 임베딩 파일 저장 ---
            torch.save(torch.stack(ecapa_embs), os.path.join(profile_dir, "ecapa.pt"))
            torch.save(torch.stack(wav_embs), os.path.join(profile_dir, "wav2vec.pt"))

            QMessageBox.information(self, "완료", f"{name} 프로필이 생성되었습니다.")
            self.load_profiles()       # 새 프로필 목록 갱신
            self.reset_to_main_scene() # 생성 완료 후 메인 화면으로 복귀

        except Exception as e:
            QMessageBox.critical(self, "오류", f"프로필 생성 중 오류가 발생했습니다:\n{e}")
            self.reset_to_main_scene()


    def load_profiles(self):
        try:
            self.profiles = [d for d in os.listdir(PROFILES_DIR)
                            if os.path.isdir(os.path.join(PROFILES_DIR, d))]
        except Exception as e:
            logging.exception("load_profiles error: %s", e)
            self.profiles = []

    def on_recording_done(self):
        """녹음 종료 직후: '녹음중…' 감추고 Find people.gif로 전환"""
        self.recording_label.setText("음성 인증 중...")
        self.challenge_label.hide()
        self.movie = QMovie("gif/Find people.gif")
        self.label.setMovie(self.movie)
        self.movie.setSpeed(75)
        self.movie.start()

    def clear_status(self, delay=4000):
        """delay(ms) 뒤에 메시지 지우기."""
        QTimer.singleShot(delay, lambda: self.status_label.setText(""))

    def _on_motion_event(self, level: int):
        """
        PIR 1(감지) 수신 시 on_user_detected() 자동 호출.
        락다운 중/진행 중/쿨다운 중이면 무시.
        """
        if not self.auto_trigger_enabled:
            return
        if level != 1:
            return
        if self.is_auth_running:
            return
        if self.lockdown_timer.isActive():
            return

        now_ms = int(time.time() * 1000)
        if (now_ms - self._last_auto_trigger_ms) < self._auto_cooldown_ms:
            return

        self._last_auto_trigger_ms = now_ms
        # 실제 인증 시작
        self.on_user_detected()


    # 사용자 감지 (도어락 인증 과정 시작)
    def on_user_detected(self):
        # 중복 방지
        if self.is_auth_running:
            return
        self.is_auth_running = True
        # 1) MP3 재생 ──
        mp3_path = os.path.abspath("mp3/Apple Intelligence Sound Effect.mp3")  # 재생할 파일 경로
        url = QUrl.fromLocalFile(mp3_path)
        media = QMediaContent(url)
        self.player.setMedia(media)
        self.player.play()

        # 2) 랜덤 문장 준비 (이미 있다면 생략)
        # sentence = random.choice([
        #     "서울의 중심은 광화문입니다.",
        #     "열려라참깨",
        #     "오늘도 좋은 하루 되세요.",
        #     "봄에는 꽃이 피고 새가 날아요.",
        #     "푸른 하늘 아래 바람이 시원하게 붑니다.",
        #     "빨간 우체통 앞에서 사진을 한 장 찍습니다.",
        #     "초콜릿 케이크 한 조각이 오늘을 달콤하게 만듭니다.",
        #     "고양이는 창가에서 따뜻한 햇살을 즐깁니다.",
        #     "사과 다섯 개와 배 세 개를 장바구니에 담습니다.",
        #     "지하철은 잠시 후 오른쪽 문이 열립니다.",
        #     "비가 오면 노란 우산을 천천히 펼칩니다.",
        #     "맑은 종소리가 골목 끝까지 퍼져 나갑니다.",
        #     "작은 씨앗이 쑥쑥 자라 숲이 됩니다.",
        #     "책장 넘기는 소리에 집중하며 한 줄씩 읽습니다.",
        #     "탁자 위 유리컵에 얼음이 차갑게 달그락거립니다.",
        #     "천천히 또렷하게 이 문장을 끝까지 읽어 주세요."
        # ])
        sentence = random.choice([
            "오늘의 나는 충분히 멋져",
            "나는 나를 믿어",
            "작아도 꾸준히, 난 하고 있어",
            "지금 이 순간도 소중해",
            "내 속도대로 가면 돼",
            "고마워, 나 스스로에게도",
            "나는 매일 조금씩 성장해",
            "괜찮아, 잘 해오고 있어",
            "실수해도 나는 사랑받을 가치가 있어",
            "오늘의 작은 한 걸음이 내일을 만든다",
            "푸른 하늘 아래 바람이 시원하게 붑니다",
            "내 가능성은 생각보다 넓어",
            "내 마음을 내가 따뜻하게 안아줄래",
            "나는 충분히 해낼 수 있어",
            "나에게 친절할수록 세상도 부드러워져",
            "오늘도 좋은 하루 되세요"
        ])
        self.current_sentence = sentence

        # ✅ 문장 라벨에 표시 + 보이기
        self.challenge_label.setText(f"아래 문장을 읽어주세요!\n\n「{self.current_sentence}」")
        self.challenge_label.show() 

        # 3) 녹음 "중"에는 MainScene.gif 재생
        self.movie = QMovie("gif/MainScene.gif")
        self.label.setMovie(self.movie)
        self.movie.setSpeed(75)
        self.movie.start()
        self.recording_label.setText("녹음중…")
        self.recording_label.show()
        #self.detect_btn.setEnabled(False)

        # 4) 통합 인증 워커 시작 + 신호 연결
        self.auth_worker = UnifiedAuthWorker(
            expected_sentence=self.current_sentence,
            profiles=self.profiles,
            attempts_left=self.attempts_left  # 또는 self.attempts_left
        )

        # === 기존 연결 ===
        self.auth_worker.recording_done.connect(self.on_recording_done)  # 🔗 녹음 종료
        self.auth_worker.finished.connect(self.on_auth_finished)          # 🔗 인증 완료

        # === 🆕 추가: "프로필 등록" 음성 감지 시 프로필 생성 함수 실행 ===
        self.auth_worker.profile_register_requested.connect(self.create_profile)

        # === 워커 실행 ===
        self.auth_worker.start()

    # 인증 과정 종료 시점
    def on_auth_finished(self, success: bool, user: str, message: str):
        # 문장 숨김 & 녹음중 숨김
        self.challenge_label.hide()
        self.recording_label.hide()

        # 결과 애니메이션 + 사운드
        if success:
            # ✅ 인증에 사용된 음성을 해당 사용자 프로필에 업데이트
            profile_dir = os.path.join(PROFILES_DIR, user)
            update_profile_embeddings(profile_dir, "auth.wav")
        
            movie = QMovie("gif/Success.gif")
            self.label.setMovie(movie)
            movie.setSpeed(75)
            movie.start()
            # 성공 시 NodeMCU 제어(백그라운드) → 한 번만 SEQ 전송
            on1, gap, on2 = 500, 5000, 500
            self.nodemcu_worker = NodeMCUSeqWorker(
                on1=on1,   # 0.5초 ON
                gap=gap,  # 5초 대기
                on2=on2    # 0.5초 ON
            )
            self.nodemcu_worker.error.connect(self._on_nodemcu_error)
            self.nodemcu_worker.start()

            # UI 리셋 시간을 시퀀스 길이에 맞춰 조정
            total_ms = on1 + gap + on2 + 500   # +500ms 여유
            QTimer.singleShot(total_ms, self.reset_to_main_scene)
            self.attempts_left = 3  # 성공 시 시도횟수 리셋
        else:
            # 오류 사운드
            mp3_path = os.path.abspath("mp3/Mac Error Sound Effect.mp3")
            url = QUrl.fromLocalFile(mp3_path)
            media = QMediaContent(url)
            self.player.setMedia(media)
            self.player.play()

            movie = QMovie("gif/Error animation.gif")
            self.label.setMovie(movie)
            movie.setSpeed(75)
            movie.start()

            # 시도 횟수 감소
            self.attempts_left = max(0, self.attempts_left - 1)

        self.status_label.setText(message)
        #self.detect_btn.setEnabled(True if self.attempts_left > 0 or success else False)

        # ▷ 음성 미감지 분기에서 바로 return 하기 전에도 False로!
        if "음성 미감지" in message:
            self.is_auth_running = False   # ★ 추가
            QTimer.singleShot(3000, self.reset_to_main_scene)
            if self.attempts_left > 0:
                self.clear_status(delay=3000)
                #self.detect_btn.setEnabled(True)
                return

        # 락다운 분기/기타 처리...
        if not success and self.attempts_left == 0:
            self.start_lockdown(5)  
        else:
            self.clear_status()

        self.is_auth_running = False        # ★ 함수 끝에서도 확실히 해제

    def _on_nodemcu_error(self, err: str):
        self.status_label.setText(f"NodeMCU 연결 실패: {err}")
        self.clear_status()

    # 모든 인증 과정 종류 후 화면 초기화
    def reset_to_main_scene(self):
        # MainScene.gif의 첫 프레임을 띄운 채 정지
        main_movie = QMovie("gif/MainScene.gif")
        main_movie.jumpToFrame(0)
        # QMovie객체가 아닌 현재 프레임만 표시하려면 setPixmap
        self.label.setPixmap(main_movie.currentPixmap())
        # 다음번 재생을 위해 self.movie에도 저장
        self.movie = main_movie

        # 상태 초기화
        self.recording_label.hide()
        #self.detect_btn.setEnabled(True)    # 접근 감지 버튼 활성화
        self.is_auth_running = False        # ★ 인증 중 상태 초기화
        self.auth_worker = None                  # 혹시 남아 있는 워커 제거

    # 인증 3회 실패 시 락다운
    def start_lockdown(self, seconds=5):
        """락다운 시작: seconds 동안 카운트다운 표시"""
        self.lockdown_remaining = int(seconds)
        #self.detect_btn.setEnabled(False)
        # 안내 문장 숨김(있다면)
        self.challenge_label.hide()
        # 즉시 1회 갱신
        self.status_label.setText(f"연속 실패로 잠시 후 다시 시도하세요 \n\n({self.lockdown_remaining}초)")
        # 1초 주기 카운트다운 시작
        self.lockdown_timer.start()

    def _tick_lockdown(self):
        """1초마다 호출되어 남은 시간 갱신"""
        self.lockdown_remaining -= 1
        if self.lockdown_remaining > 0:
            self.status_label.setText(f"연속 실패로 잠시 후 다시 시도하세요 \n\n({self.lockdown_remaining}초)")
        else:
            self.lockdown_timer.stop()
            self._unlock_after_lockdown()

    def _unlock_after_lockdown(self):
        """락다운 해제: 버튼 활성화 및 메시지 정리"""
        # 혹시라도 남아있으면 정지
        if self.lockdown_timer.isActive():
            self.lockdown_timer.stop()
        self.attempts_left = 3
        self.reset_to_main_scene()
        self.status_label.setText("다시 시도할 수 있습니다.")
        #self.detect_btn.setEnabled(True)
        self.clear_status()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SmartDoorlockUI()
    w.show()
    sys.exit(app.exec_())
