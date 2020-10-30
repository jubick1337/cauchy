import io
from pathlib import Path

from google.cloud import speech
# from google.cloud.speech_v1 import enums
from google.oauth2 import service_account


class GoogleASRWrapper:
    def __init__(self, credentials: Path):
        self._credentials = service_account.Credentials.from_service_account_file(credentials)
        self._client = speech.SpeechClient(credentials=self._credentials)
        self._language_code = "ru-RU"
        self._sample_rate = 16000
        self._encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16

    def get_text(self, local_file_path):
        with io.open(local_file_path, "rb") as f:
            content = f.read()
            audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            language_code=self._language_code,
            sample_rate_hertz=self._sample_rate,
            encoding=self._encoding,
        )

        response = self._client.recognize(config=config, audio=audio)

        for result in response.results:
            alternative = result.alternatives[0]
            return alternative.transcript
