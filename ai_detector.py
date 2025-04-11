# ai_detector.py
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class SimpleResNetDetector(torch.nn.Module):
    def __init__(self, input_size=201, num_classes=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(32 * (input_size//4) * (input_size//4), 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_detector(model_path="ai_voice_detector.pth"):
    model = SimpleResNetDetector()
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("✅ 학습된 모델 불러오기 완료")
    else:
        print("⚠️ 학습된 모델이 없습니다. 새 모델을 초기화합니다.")
    model.eval()
    return model

def wav_to_spec(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = T.Resample(sr, 16000)(waveform)
    spec = T.MelSpectrogram(sample_rate=16000, n_fft=400)(waveform)
    spec_db = T.AmplitudeToDB()(spec)
    return spec_db.unsqueeze(0)

def is_ai_voice(file_path, model):
    spec = wav_to_spec(file_path)
    with torch.no_grad():
        output = model(spec)
        prob = F.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
    print(f"🔍 AI 탐지 결과: {'FAKE' if pred == 1 else 'REAL'} (score={prob[0][1].item():.4f})")
    return pred == 1  # 1: FAKE
