import argparse
import os
import subprocess
import threading
import time
import wave
from pathlib import Path

import logger
import pyaudio
import torch
import torchaudio

from actions.greeting import Greeting
from actions.simple_math import SimpleMath
from actions.time_now import TimeNow
from asr.google_asr_wrapper import GoogleASRWrapper
from tts.google_tts_wrapper import GoogleTTSWrapper
from wakeword.model import WakeWordDetector

logger = logger.logger


class Dispatcher:

    def __init__(self):
        self._actions = [SimpleMath(), Greeting(), TimeNow()]

    def execute(self, query: str) -> str:
        for action in self._actions:
            result = action.get_result(query)
            if result:
                return result

        return 'Я вас не понял'


class QueryListener:

    def __init__(self, sample_rate=16000, record_seconds=4):
        self.chunk = 1024
        self.file = 'tmp.wav'
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def record(self):
        logger.info('started rec')
        frames = []

        for i in range(0, int(self.sample_rate / self.chunk * self.record_seconds)):
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)

        wf = wave.open(self.file, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()
        logger.info('stop rec')

    def flush(self):
        os.remove(self.file)


class WakeWordListener:

    def __init__(self, sample_rate=8000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        logger.info("\nWake Word Engine is now listening... \n")


class WakeWordEngine:

    def __init__(self):
        self.listener = WakeWordListener(sample_rate=8000, record_seconds=2)
        checkpoint = torch.load(Path('./models/wakeword_checkpoint.pt'), map_location='cpu')
        self.model = WakeWordDetector(40, 64, num_layers=4)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.mfcc = torchaudio.transforms.MFCC(8000)
        self.audio_q = list()

    def save(self, waveforms, filename="wakeword_temp"):
        wf = wave.open(filename, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(8000)
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return filename

    def predict(self, audio):
        with torch.no_grad():
            filename = self.save(audio)
            waveform, _ = torchaudio.load(filename)
            mfcc = self.mfcc(waveform).transpose(1, -1)
            out = self.model(mfcc)
            prediction = torch.round(torch.sigmoid(out))
            return prediction.item()

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 15:  # remove part of stream
                diff = len(self.audio_q) - 15
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.predict(self.audio_q))
            elif len(self.audio_q) == 15:
                action(self.predict(self.audio_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                  args=(action,), daemon=True)
        thread.start()


class WakeWordAction:
    def __init__(self, asr, tts, sensitivity=5):
        self.detect_in_row = 0
        self.sensitivity = sensitivity
        self.activation_sound = Path('./activated.wav')
        self.asr = asr
        self.tts = tts
        self.query_listener = QueryListener()
        self.dispatcher = Dispatcher()

    def __call__(self, prediction):
        if prediction:
            self.detect_in_row += 1
            if self.detect_in_row == self.sensitivity:
                self.play(self.activation_sound)
                query_listener.record()
                self.detect_in_row = 0
                text = self.asr.get_text(self.query_listener.file)
                self.query_listener.flush()
                logger.info(f'recognized query: {text}')
                dispatcher_result = self.dispatcher.execute(text)
                logger.info(f'Result of dispatcher: {dispatcher_result}')
                tts.get_audio(dispatcher_result)
                self.play('./output.wav')

        else:
            self.detect_in_row = 0

    def play(self, sound: str):
        subprocess.Popen(['play', '-v', '1', sound], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the wakeword engine")
    parser.add_argument('--sensitivity', type=int, default=5, required=False,
                        help='lower value is more sensitive to activations')

    args = parser.parse_args()
    asr = GoogleASRWrapper(Path('asr/credentials.json'))
    tts = GoogleTTSWrapper(Path('tts/credentials.json'))
    wakeword_engine = WakeWordEngine()
    action = WakeWordAction(sensitivity=5, asr=asr, tts=tts)
    query_listener = QueryListener()
    wakeword_engine.run(action)
    threading.Event().wait()
