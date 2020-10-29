import random
import re
from typing import Optional

from actions.action import Action


class Greeting(Action):
    def __init__(self):
        self._regex = re.compile(r'(Привет|Здравствуйте)')
        self._answers = ['Привет', 'Хай', 'Рад тебя видеть']

    def get_result(self, query: str) -> Optional[str]:
        if self._regex.search(query):
            return random.choice(self._answers)

        return None
