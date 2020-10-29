from abc import ABC
from typing import Optional


class Action(ABC):
    def get_text(self, query: str) -> Optional[str]:
        pass
