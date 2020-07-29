import torch

import torchaudio
import pyaudio
import wave

from wakeword.model import WakeWordDetector

checkpoint = torch.load('./models/wakeword.pt', map_location='cpu')
model = WakeWordDetector(40, 64, num_layers=4)
model.load_state_dict(checkpoint)
model.eval()
tr = torchaudio.transforms.MFCC(8000)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

with torch.no_grad():
    w, _ = torchaudio.load('output.wav')
    w = tr(w)
    w = w.transpose(1, -1)
    print(torch.round(torch.sigmoid(model(w))))
