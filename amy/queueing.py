from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    data_board: Any = field(compare=False)
    label_moves: Any = field(compare=False)
    label_value: Any = field(compare=False)
    label_wdl: Any = field(compare=False)
    label_moves_remaining: Any = field(compare=False)


# Maximum priority to assign an item in the position queue
MAX_PRIO = 1_000_000
