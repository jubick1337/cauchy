import io
from pathlib import Path

from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from google.oauth2 import service_account


class GoogleASRWrapper:
    def __init__(self, credentials: Path):
        self._credentials = service_account.Credentials.from_service_account_file(credentials)
        self._client = speech_v1.SpeechClient(credentials=self._credentials)
        self._language_code = "ru-RU"
        self._sample_rate = 16000
        self._encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16

    def get_text(self, local_file_path):
        config = {
            "language_code": self._language_code,
            "sample_rate_hertz": self._sample_rate,
            "encoding": self._encoding,
        }
        with io.open(local_file_path, "rb") as f:
            content = f.read()
        audio = {"content": content}

        response = self._client.recognize(config, audio)

        for result in response.results:
            alternative = result.alternatives[0]
            return alternative.transcript
