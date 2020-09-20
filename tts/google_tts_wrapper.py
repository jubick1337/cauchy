from pathlib import Path

from google.cloud import texttospeech
from google.oauth2 import service_account


class GoogleTTSWrapper:
    def __init__(self, credentials: Path):
        self._credentials = service_account.Credentials.from_service_account_file(credentials)
        self._client = texttospeech.TextToSpeechClient(credentials=self._credentials)
        self._language_code = "ru-RU"
        self._ssml_gender = texttospeech.SsmlVoiceGender.MALE
        self._audio_encoding = texttospeech.AudioEncoding.MP3
        self._audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

    def get_audio(self, text: str):
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self._client.synthesize_speech(
            input=synthesis_input, voice=texttospeech.VoiceSelectionParams(
                language_code=self._language_code, ssml_gender=self._ssml_gender
            ), audio_config=self._audio_config
        )

        with open("output.wav", "wb") as out:
            out.write(response.audio_content)
