import os

import torch

import torchaudio
import pyaudio
import wave

from wakeword.model import WakeWordDetector

checkpoint = torch.load('./models/wakeword_checkpoint.pt', map_location='cpu')
model = WakeWordDetector(40, 64, num_layers=4)
model.load_state_dict(checkpoint)
model.eval()
tr = torchaudio.transforms.MFCC(8000)

with torch.no_grad():
    path = './sounds/test/0/'
    for f in os.listdir(path):
        w, _ = torchaudio.load(path + f)
        w = tr(w)
        w = w.transpose(1, -1)
        print((torch.sigmoid(model(w))))
