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

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QMessageBox, QInputDialog, QDialog, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl, QSize
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QMovie
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from speechbrain.pretrained import SpeakerRecognition
import whisper
from sentence_transformers import SentenceTransformer

import socket, time, logging


# ========== 설정 ==========
SAMPLE_RATE = 16000
RECORD_DURATION = 10
RECORD_SECONDS_LOGIN = 10
PROFILES_DIR = "profiles"
SIMILARITY_THRESHOLD = 0.5
ALPHA = 0.5
os.makedirs(PROFILES_DIR, exist_ok=True)

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

# AP 모드라면 기본 IP, STA 모드라면 시리얼 모니터에 뜬 IP로 바꾸세요.
NODEMCU_HOST = "192.168.123.110"   # 또는 예: "192.168.0.37"
NODEMCU_PORT = 7777

# ========== 모델 불러오기 ==========
ecapa_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="ecapa_model"
)
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.eval()

sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Whisper는 MPS 말고 CPU/CUDA만 사용 (MPS에서 sparse 에러 방지)
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()  # CUDA일 때만 fp16

whisper_model = whisper.load_model("small", device=WHISPER_DEVICE)

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


def semantic_similarity(a: str, b: str, threshold: float = 0.7) -> bool:
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
    raise RuntimeError(f"NodeMCU 연결 실패: {last_err}")

# 녹음 전용 워커 스레드 정의
class RecordWorker(QThread):
    finished = pyqtSignal(bool)   # 녹음 성공 여부를 전달

    def __init__(self, path, duration):
        super().__init__()
        self.path = path
        self.duration = duration

    def run(self):
        try:
            ok = record_until_silence(self.path, self.duration)
        except Exception as e:
            logging.exception("RecordWorker error: %s", e)
            ok = False
        self.finished.emit(ok)


# 2차 인증 전용 워커 스레드 정의
class SecondAuthWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, user, sentence, attempts):
        super().__init__()
        self.user = user
        self.expected_sentence = sentence
        self.attempts = attempts

    def run(self):
        # 1) 녹음
        try:
            ok = record_until_silence("second.wav", RECORD_SECONDS_LOGIN)
        except Exception as e:
            logging.exception("SecondAuthWorker record error: %s", e)
            self.finished.emit(False, "녹음 실패")
            return
        if not ok:
            self.finished.emit(False, "음성 미감지")
            return

        # 2) Whisper STT & 의미 비교
        try:
            result = whisper_model.transcribe(
                "second.wav",
                language="ko",
                temperature=0.0,
                beam_size=1, best_of=1,
                condition_on_previous_text=False,
                fp16=USE_FP16,
                initial_prompt="스마트 도어락, 인증, 광화문, 날씨"
            )
            spoken = result["text"].strip().lower()
        except Exception as e:
            logging.exception("Whisper error: %s", e)
            self.finished.emit(False, "음성 인식 실패")
            return

        if not semantic_similarity(self.expected_sentence.lower(), spoken):
            self.finished.emit(False, "문장 의미 불일치")
            return

        # 3) 화자 유사도…
        try:
            score = compare_with_ensemble(os.path.join(PROFILES_DIR, self.user),
                                        "second.wav", alpha=ALPHA)
        except Exception as e:
            logging.exception("Embedding compare error: %s", e)
            self.finished.emit(False, "프로필 비교 실패")
            return
        print(f"[화자유사도] cos={score:.3f}")
        success = score >= SIMILARITY_THRESHOLD
        msg = f"{self.user}님 안녕하세요! 도어락이 열렸습니다" if success else f"남은 시도 {self.attempts-1}회"
        self.finished.emit(success, msg)

class NodeMCUWorker(QThread):
    error = pyqtSignal(str)

    def __init__(self, open_ms=7000, polls=8, interval=0.4, parent=None):
        super().__init__(parent)
        self.open_ms = open_ms
        self.polls = polls
        self.interval = interval
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            send_nodemcu(f"OPEN {self.open_ms}")
            for _ in range(self.polls):
                if self._stop:
                    break
                send_nodemcu("STATUS")
                time.sleep(self.interval)
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


# ========== 녹음 안내 다이얼로그 ==========
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
        self.setGeometry(1800, 0, 500, 1285)    # 실제 소형 LCD 화면 비율과 유사하게
        self.setStyleSheet("background-color: white;")
        self.auth_fail_count = 0    # 실패 횟수 초기화
        self.second_attempts = 3    # 2차 인증 총 시도 횟수

        # ── 메인 레이아웃 (한 번만!) ──
        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0,0,0,0)

        # ── 애니메이션 라벨 & 녹음중 레이블 준비 ──
        self.label = QLabel(self)
        self.label.setFixedSize(500, 500)
        self.label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie("gif/MainScene.gif")
        self.movie.setScaledSize(QSize(250, 250))
        self.label.setFixedHeight(500)
        # 최초엔 재생하지 않고 첫 프레임만
        self.movie.jumpToFrame(0)
        self.label.setPixmap(self.movie.currentPixmap())
        main_lay.addWidget(self.label)

        # ── 2차 인증 문장 표시용 레이블 ──
        self.challenge_label = QLabel("", self)
        self.challenge_label.setAlignment(Qt.AlignCenter)
        self.challenge_label.setStyleSheet("color: #bbbbbf; font-size: 23px;")
        self.challenge_label.setFixedHeight(100)
        self.challenge_label.hide()
        main_lay.addWidget(self.challenge_label)

        # ── 미디어 플레이어 준비 ──
        self.player = QMediaPlayer(self)

        # ── 상태 메시지 레이블 추가 ──
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            color: #bbbbbf;
            font-size: 23px;
            padding: 8px;
        """)
        self.status_label.setFixedHeight(100)
        main_lay.addWidget(self.status_label)

        # ── “녹음중” 텍스트 레이블 ──
        self.recording_label = QLabel("녹음중…", self)
        self.recording_label.setAlignment(Qt.AlignCenter)
        self.recording_label.setStyleSheet("color: #bbbbbf; font-size: 23px;")
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
        self.create_btn = QPushButton("➕ 프로필 생성");     
        self.create_btn.clicked.connect(self.create_profile)
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
        """)
        hbtn.addWidget(self.detect_btn); hbtn.addWidget(self.create_btn)
        main_lay.addLayout(hbtn)

        self.profiles = []
        self.load_profiles()
        
    def create_profile(self):
        name, ok = QInputDialog.getText(self, "프로필 생성", "사용자 이름을 입력하세요:")
        if not ok or not name.strip():
            return
        name = name.strip()
        profile_dir = os.path.join(PROFILES_DIR, name)
        if os.path.exists(profile_dir):
            QMessageBox.warning(self, "중복", "이미 존재하는 프로필입니다.")
            return
        os.makedirs(profile_dir, exist_ok=True)

        sentences = [
            "음성으로 문을 열겠습니다. 지금부터 인증을 시작합니다.",
            "이 문장을 정확히 말하면 잠금장치가 해제됩니다.",
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

        # 평균 임베딩 저장
        torch.save(torch.stack(ecapa_embs).mean(0), os.path.join(profile_dir, "ecapa.pt"))
        torch.save(torch.stack(wav_embs).mean(0),  os.path.join(profile_dir, "wav2vec.pt"))

        QMessageBox.information(self, "완료", "프로필 생성이 완료되었습니다!")
        self.load_profiles()

    def load_profiles(self):
        try:
            self.profiles = [d for d in os.listdir(PROFILES_DIR)
                            if os.path.isdir(os.path.join(PROFILES_DIR, d))]
        except Exception as e:
            logging.exception("load_profiles error: %s", e)
            self.profiles = []

    def clear_status(self, delay=3000):
        """delay(ms) 뒤에 메시지 지우기."""
        QTimer.singleShot(delay, lambda: self.status_label.setText(""))

    def on_user_detected(self):
        # 1) MP3 재생 ──
        mp3_path = os.path.abspath("mp3/Apple Intelligence Sound Effect.mp3")  # 재생할 파일 경로
        url = QUrl.fromLocalFile(mp3_path)
        media = QMediaContent(url)
        self.player.setMedia(media)
        self.player.play()

        # 2) 사용자 접근 감지 → 애니메이션 재생 + 녹음중 텍스트 보이기
        self.label.setMovie(self.movie)
        self.movie.start()
        self.recording_label.setText("녹음중…")
        self.recording_label.show()
        self.detect_btn.setEnabled(False)

        # 3) 워커 스레드로 녹음 시작
        self.recorder = RecordWorker("login.wav", RECORD_SECONDS_LOGIN)
        self.recorder.finished.connect(self.on_record_finished)
        self.recorder.start()

    def on_record_finished(self, ok: bool):
        # 이 슬롯은 메인 스레드에서 실행됩니다.
        self.movie.stop()
        self.recording_label.hide()
        self.detect_btn.setEnabled(True)

        if not ok:
            # 오류 MP3 재생
            mp3_path = os.path.abspath("mp3/Mac Error Sound Effect.mp3")  # 재생할 파일 경로
            url = QUrl.fromLocalFile(mp3_path)
            media = QMediaContent(url)
            self.player.setMedia(media)
            self.player.play()

            # 오류용 GIF 교체
            error_movie = QMovie("gif/Error animation.gif")
            self.label.setMovie(error_movie)
            error_movie.setSpeed(75)  # 기본 속도의 75% → 절반 속도로 재생
            error_movie.start()

            self.status_label.setText("실패: 음성이 감지되지 않았습니다.")
            self.clear_status()
            return

        # 녹음이 성공했으니 1차 인증 로직 실행
        self.process_first_auth()

    def process_first_auth(self):
        if not self.profiles:
            # 오류 MP3 재생
            mp3_path = os.path.abspath("mp3/Mac Error Sound Effect.mp3")  # 재생할 파일 경로
            url = QUrl.fromLocalFile(mp3_path)
            media = QMediaContent(url)
            self.player.setMedia(media)
            self.player.play()

            # 오류용 GIF 교체
            error_movie = QMovie("gif/Error animation.gif")
            self.label.setMovie(error_movie)
            error_movie.setSpeed(75)  # 기본 속도의 75% → 절반 속도로 재생
            error_movie.start()

            self.status_label.setText("오류: 등록된 프로필이 없습니다.")
            self.clear_status()
            return

        # 녹음 끝났으니 “녹음중” 텍스트 숨기고 완료 애니메이션으로 교체
        self.movie.stop()

        best_match, best_score = None, 0.0

        print("1차 인증 결과")
        for p in self.profiles:
            profile_dir = os.path.join(PROFILES_DIR, p)
            combined = compare_with_ensemble(profile_dir, "login.wav")
            print(f"[유사도] {p}: {combined:.4f}")

            if combined > best_score:
                best_score = combined
                best_match = p

        if best_score >= SIMILARITY_THRESHOLD:
            success_movie = QMovie("gif/Success.gif")
            self.label.setMovie(success_movie)
            success_movie.setSpeed(75)
            success_movie.start()
            self.status_label.setText(f"1차 인증 성공: {best_match}님, 안녕하세요!")
            self.start_second_auth(best_match)
            try:
                send_nodemcu("PING")
                send_nodemcu("STATUS")
            except Exception as e:
                self.status_label.setText(f"NodeMCU 연결 실패: {e}")
                self.clear_status()
        else:
            # 오류 MP3 재생
            mp3_path = os.path.abspath("mp3/Mac Error Sound Effect.mp3")  # 재생할 파일 경로
            url = QUrl.fromLocalFile(mp3_path)
            media = QMediaContent(url)
            self.player.setMedia(media)
            self.player.play()

            # 오류용 GIF 교체
            error_movie = QMovie("gif/Error animation.gif")
            self.label.setMovie(error_movie)
            error_movie.setSpeed(75)
            error_movie.start()

            self.status_label.setText("인증 실패: 등록되지 않은 음성입니다.")
        self.clear_status()

    def start_second_auth(self, user):
        # 1) 랜덤 문장 선택
        sentence = random.choice([
            "서울의 중심은 광화문입니다.",
            "오늘도 좋은 하루 되세요.",
            "봄에는 꽃이 피고 새가 날아요."
        ])
        # 2) 화면에 보여주기
        self.challenge_label.setText(f"2차 인증\n\n「{sentence}」")
        self.challenge_label.show()

        # 3) 애니메이션 재생
        auth_movie = QMovie("gif/Find people.gif")
        self.label.setMovie(auth_movie)
        auth_movie.setSpeed(75)
        auth_movie.start()

        # 4) 버튼 잠금
        self.detect_btn.setEnabled(False)

        # 5) 워커에 sentence도 넘기기
        self.second_worker = SecondAuthWorker(
            user=user,
            sentence=sentence,
            attempts=self.second_attempts
        )
        self.second_worker.finished.connect(self.on_second_auth_finished)
        self.second_worker.start()

    def on_second_auth_finished(self, success, message):
        # 챌린지 문장 숨기기
        self.challenge_label.hide()
        # 기존 성공/실패 처리…
        movie = QMovie("gif/Success.gif" if success else "gif/Error animation.gif")
        self.label.setMovie(movie)
        movie.start()
        self.status_label.setText(message)
        self.detect_btn.setEnabled(True)
        if not success:
            self.clear_status()
            self.second_attempts -= 1
            return
        self.clear_status(delay=5000)

        # ✅ 인증 성공: NodeMCU에 개방 펄스 전송 (백그라운드 스레드)
        self.nodemcu_worker = NodeMCUWorker(open_ms=7000, polls=8, interval=0.4)
        self.nodemcu_worker.error.connect(self._on_nodemcu_error)
        self.nodemcu_worker.start()

        # 성공일 때만 5초 뒤에 메인 화면으로 리셋
        QTimer.singleShot(5000, self.reset_to_main_scene)

    def _on_nodemcu_error(self, err: str):
        self.status_label.setText(f"NodeMCU 연결 실패: {err}")
        self.clear_status()

    def reset_to_main_scene(self):
        # MainScene.gif의 첫 프레임을 띄운 채 정지
        main_movie = QMovie("gif/MainScene.gif")
        main_movie.setScaledSize(QSize(250, 250))
        main_movie.jumpToFrame(0)
        # QMovie객체가 아닌 현재 프레임만 표시하려면 setPixmap
        self.label.setPixmap(main_movie.currentPixmap())
        # 다음번 재생을 위해 self.movie에도 저장
        self.movie = main_movie


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SmartDoorlockUI()
    w.show()
    sys.exit(app.exec_())
