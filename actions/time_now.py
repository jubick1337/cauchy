import re
from datetime import datetime
from typing import Optional

from actions.action import Action


class TimeNow(Action):
    def __init__(self):
        self._regex = re.compile(r'врем')

    def get_result(self, query: str) -> Optional[str]:
        if self._regex.search(query):
            now = datetime.now()
            return f'{now.hour} {now.minute}'

        return None
