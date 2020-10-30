import re
from typing import Optional

from actions.action import Action


class AdvancedMath(Action):

    def __init__(self):
        self._regex = re.compile(r'[А-Яа-я ]*((\d+|\s?) ([+\-*/]|\s?) (\d+|\s?))+')

    def get_result(self, query: str) -> Optional[str]:
        if not query:
            return
        query = query.replace('х', '*').replace('x', '*')

        if self._regex.match(query):
            expression = ' '.join([c for c in query if c.isdigit() or c in ['+', '-', '*', '/']]).strip()
            return eval(expression)
