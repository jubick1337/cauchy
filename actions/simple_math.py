import re
from typing import Optional

from actions.action import Action


class SimpleMathAction(Action):
    def __init__(self):
        self._regex = re.compile(r'[А-Яа-я ]*(\d+) ([+\-*/]) (\d+)')

    def get_result(self, query: str) -> Optional[str]:
        if not query:
            return
        query = query.replace('х', '*').replace('x', '*')
        try:
            groups = self._regex.match(query).groups()
            first_operand, operator, second_operand = groups
            return str(eval(f'{first_operand}{operator}{second_operand}'))
        except:
            return
