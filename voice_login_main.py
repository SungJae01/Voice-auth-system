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
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QHBoxLayout, QMessageBox, QGridLayout, QInputDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap

from transformers import Wav2Vec2Model, Wav2Vec2Processor
from speechbrain.pretrained import SpeakerRecognition

# ========== 설정 ==========
SAMPLE_RATE = 16000
RECORD_SECONDS_PROFILE = 10
RECORD_SECONDS_LOGIN = 5
SIMILARITY_THRESHOLD = 0.6
PROFILES_DIR = "profiles"
ALPHA = 0.5
os.makedirs(PROFILES_DIR, exist_ok=True)

# ========== 모델 로드 ==========
ecapa_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="ecapa_model")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model.eval()

# ========== Pitch 추출 ==========
def extract_pitch(y, sr, target_len=512):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1])]
    pitch_values = np.nan_to_num(pitch_values)
    pitch_values = librosa.util.fix_length(pitch_values, size=target_len)
    return torch.tensor(pitch_values).float().unsqueeze(0)

# ========== 임베딩 생성 ==========
def get_wav2vec_pitch_embedding(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    inputs = wav2vec_processor(y, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        output = wav2vec_model(inputs)
    wav_emb = torch.mean(output.last_hidden_state, dim=1)
    pitch_emb = torch.mean(extract_pitch(y, sr), dim=1, keepdim=True) / 300.0
    return torch.cat((wav_emb, pitch_emb), dim=1)

def get_ecapa_embedding(file_path):
    waveform, sr = torchaudio.load(file_path)
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    emb = ecapa_model.encode_batch(waveform).squeeze(0)
    return emb.unsqueeze(0)

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b).mean().item()

def compare_with_ensemble(emb1_dir, test_audio, alpha=ALPHA):
    emb1_ecapa = torch.load(os.path.join(emb1_dir, "ecapa.pt")).unsqueeze(0)
    emb1_wav = torch.load(os.path.join(emb1_dir, "wav2vec.pt")).unsqueeze(0)

    emb2_ecapa = get_ecapa_embedding(test_audio)
    emb2_wav = get_wav2vec_pitch_embedding(test_audio)

    sim_ecapa = cosine_similarity(emb1_ecapa, emb2_ecapa)
    sim_wav = cosine_similarity(emb1_wav, emb2_wav)
    print(f"[유사도] ECAPA: {sim_ecapa:.4f}, Wav2Vec2+Pitch: {sim_wav:.4f}")
    return alpha * sim_wav + (1 - alpha) * sim_ecapa

# ========== PyQt5 UI ==========
class VoiceLoginApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 Voice Login (Ensemble)")
        self.setGeometry(300, 150, 600, 400)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.profile_buttons_layout = QGridLayout()
        self.layout.addLayout(self.profile_buttons_layout)

        self.create_buttons()
        self.load_profiles()

    def create_buttons(self):
        btn_layout = QHBoxLayout()

        self.create_btn = QPushButton("➕ 프로필 생성")
        self.create_btn.clicked.connect(self.create_profile)
        btn_layout.addWidget(self.create_btn)

        self.delete_btn = QPushButton("🗑️ 프로필 삭제")
        self.delete_btn.clicked.connect(self.delete_profile)
        btn_layout.addWidget(self.delete_btn)

        self.login_btn = QPushButton("🔐 자동 로그인")
        self.login_btn.clicked.connect(self.login)
        btn_layout.addWidget(self.login_btn)

        self.layout.addLayout(btn_layout)

    def load_profiles(self):
        for i in reversed(range(self.profile_buttons_layout.count())):
            widget = self.profile_buttons_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.profiles = []
        row = col = 0
        for name in os.listdir(PROFILES_DIR):
            profile_path = os.path.join(PROFILES_DIR, name)
            if os.path.isdir(profile_path):
                btn = QPushButton(name)
                btn.setFixedSize(100, 100)
                btn.clicked.connect(lambda _, n=name: self.login_profile(n))
                self.profile_buttons_layout.addWidget(btn, row, col)
                self.profiles.append(name)
                col += 1
                if col >= 4:
                    row += 1
                    col = 0

    def record_audio_login(self, path):
        audio = sd.rec(int(SAMPLE_RATE * RECORD_SECONDS_LOGIN), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        sf.write(path, audio, SAMPLE_RATE)

    def record_audio_profile(self, path):
        audio = sd.rec(int(SAMPLE_RATE * RECORD_SECONDS_PROFILE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        sf.write(path, audio, SAMPLE_RATE)

    def create_profile(self):
        name, ok = QInputDialog.getText(self, "프로필 생성", "사용자 이름을 입력하세요:")
        if not ok or not name.strip():
            return
        path = os.path.join(PROFILES_DIR, name.strip())
        if os.path.exists(path):
            QMessageBox.warning(self, "중복", "이미 존재하는 프로필입니다.")
            return

        os.makedirs(path, exist_ok=True)

        # long_sentences = [
        #     "오늘은 날씨가 맑고 기분 좋은 하루가 될 것 같아요. 이 문장을 또박또박 읽어주세요.",
        #     "인공지능 기술은 우리의 삶을 혁신적으로 변화시키고 있으며 앞으로 더 많은 발전이 기대됩니다.",
        #     "파이썬은 배우기 쉽고 강력한 기능을 갖춘 프로그래밍 언어입니다. 천천히 정확하게 읽어주세요.",
        #     "커피 한 잔을 마시며 여유로운 시간을 보내는 것은 일상 속의 소소한 행복입니다.",
        #     "기술은 문제를 해결하는 도구이며, 우리는 그 도구를 잘 활용해야 합니다. 이 문장을 읽어주세요."
        # ]

        long_sentences = [
            "음성으로 문을 열겠습니다. 지금부터 인증을 시작합니다.",
            "이 문장을 정확히 말하면 잠금장치가 해제됩니다.",
            "지금 들리는 이 목소리는 저만 사용할 수 있는 보안 열쇠입니다.",
            "스마트 도어 시스템을 통해 집에 안전하게 들어가고 싶습니다.",
            "이제 제 음성으로 문을 열 수 있는 시대가 왔습니다. 열어주세요."
        ]

        ecapa_embs, wav2vec_embs = [], []

        for i, sentence in enumerate(long_sentences):
            QMessageBox.information(self, f"녹음 {i+1}/5", f"📢 다음 문장을 읽어주세요:『 {sentence} 』")
            rec_path = os.path.join(path, f"record_{i+1}.wav")
            self.record_audio_profile(rec_path)

            ecapa_emb = get_ecapa_embedding(rec_path).squeeze(0)
            wav2vec_emb = get_wav2vec_pitch_embedding(rec_path).squeeze(0)

            ecapa_embs.append(ecapa_emb)
            wav2vec_embs.append(wav2vec_emb)

        torch.save(torch.stack(ecapa_embs).mean(dim=0), os.path.join(path, "ecapa.pt"))
        torch.save(torch.stack(wav2vec_embs).mean(dim=0), os.path.join(path, "wav2vec.pt"))

        QMessageBox.information(self, "완료", "🎉 프로필 생성이 완료되었습니다!")
        self.load_profiles()


    def second_auth(self, profile_dir):
        long_sentences = [
            "서울의 중심은 광화문입니다.",
            "오늘도 좋은 하루 되세요.",
            "봄에는 꽃이 피고 새가 날아요.",
            "학교에 가는 길은 즐겁습니다.",
            "나는 오늘 파이썬을 공부합니다."
        ]
        sentence = random.choice(long_sentences)
        QMessageBox.information(self, "2차 인증", f"📢 다음 문장을 말해주세요:『 {sentence} 』")
        self.record_audio_login("second.wav")
        score = compare_with_ensemble(profile_dir, "second.wav", alpha=0.3)
        print(f"🔐 2차 인증 유사도: {score:.4f}")
        if score >= SIMILARITY_THRESHOLD:
            QMessageBox.information(self, "로그인 완료", f"✅ 최종 인증 성공! (2차 유사도: {score:.4f})")
        else:
            QMessageBox.warning(self, "2차 인증 실패", f"❌ 유사도 부족 (2차 유사도: {score:.4f})")

    def delete_profile(self):
        name, ok = QInputDialog.getItem(self, "프로필 삭제", "삭제할 프로필을 선택하세요:", self.profiles, editable=False)
        if not ok or not name:
            return
        path = os.path.join(PROFILES_DIR, name)
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        os.rmdir(path)
        QMessageBox.information(self, "삭제 완료", f"{name} 프로필이 삭제되었습니다.")
        self.load_profiles()

    def login_profile(self, name):
        profile_dir = os.path.join(PROFILES_DIR, name)
        self.record_audio_login("login.wav")
        score = compare_with_ensemble(profile_dir, "login.wav", alpha=ALPHA)
        print(f"👤 {name} 유사도: {score:.4f}")
        if score >= SIMILARITY_THRESHOLD:
            QMessageBox.information(self, "1차 인증 성공", f"{name}님 1차 인증 성공! (유사도: {score:.4f})")
            self.second_auth(profile_dir)
        else:
            QMessageBox.warning(self, "로그인 실패", f"❌ 인증 실패 (유사도: {score:.4f})")

    def login(self):
        self.record_audio_login("login.wav")
        best_match, best_score = None, 0.0
        for name in self.profiles:
            profile_dir = os.path.join(PROFILES_DIR, name)
            score = compare_with_ensemble(profile_dir, "login.wav", alpha=ALPHA)
            print(f"👥 {name} 유사도: {score:.4f}")
            if score > best_score:
                best_score = score
                best_match = name
        if best_match and best_score >= SIMILARITY_THRESHOLD:
            QMessageBox.information(self, "1차 인증 성공", f"{best_match}님 1차 인증 성공! (유사도: {best_score:.4f})")
            self.second_auth(os.path.join(PROFILES_DIR, best_match))
        else:
            QMessageBox.warning(self, "로그인 실패", "❌ 일치하는 프로필이 없습니다.")

# ========== 실행 ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VoiceLoginApp()
    win.show()
    sys.exit(app.exec_())
