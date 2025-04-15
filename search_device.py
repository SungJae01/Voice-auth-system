import sounddevice as sd

print("🎧 사용 가능한 오디오 장치 목록:")
devices = sd.query_devices()

for i, device in enumerate(devices):
    print(f"[{i}] {device['name']} - Input Channels: {device['max_input_channels']}, Output Channels: {device['max_output_channels']}")

print("🎙️ 현재 기본 장치 (input/output):", sd.default.device)