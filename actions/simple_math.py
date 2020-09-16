import re
from typing import Optional


class SimpleMathAction:
    def __init__(self):
        self._regex = re.compile(r'[А-Яа-я ]*(\d+) ([+\-*/]) (\d+)')

    def get_result(self, query: str) -> Optional[int or float]:
        query = query.replace('х', '*').replace('x', '*')
        try:
            groups = self._regex.match(query).groups()
            first_operand, operator, second_operand = groups
            return eval(f'{first_operand}{operator}{second_operand}')
        except SyntaxError:
            return None
