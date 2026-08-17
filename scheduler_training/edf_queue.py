"""Non-preemptive EDF waiting queue used by Phase4.5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models import Action, Task


@dataclass
class QueueItem:
    task: Task
    route: Action
    stage: str
    service_time_s: float
    remaining_service_time_s: float
    decision_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def absolute_deadline_s(self) -> float:
        return float(self.task.arrival_time) + float(self.task.deadline_s)


class EDFTaskQueue:
    """A small deterministic EDF queue.

    Running work is managed outside this class, therefore EDF is explicitly
    non-preemptive: a newly arrived earlier-deadline task only changes the order
    of *waiting* tasks.
    """

    def __init__(self, policy: str = "edf") -> None:
        if policy not in {"edf", "fifo"}:
            raise ValueError(f"非法queue policy：{policy}")
        self.policy = policy
        self._items: list[QueueItem] = []
        self._seq = 0
        self._sequence: dict[int, int] = {}

    def push(self, item: QueueItem) -> None:
        self._items.append(item)
        self._sequence[id(item)] = self._seq
        self._seq += 1

    def pop(self) -> QueueItem:
        if not self._items:
            raise IndexError("队列为空")
        if self.policy == "fifo":
            item = self._items.pop(0)
        else:
            idx = min(
                range(len(self._items)),
                key=lambda i: (
                    self._items[i].absolute_deadline_s,
                    float(self._items[i].task.arrival_time),
                    int(self._items[i].task.task_id),
                    self._sequence[id(self._items[i])],
                ),
            )
            item = self._items.pop(idx)
        self._sequence.pop(id(item), None)
        return item

    def remove(self, item: QueueItem) -> None:
        self._items.remove(item)
        self._sequence.pop(id(item), None)

    def clear(self) -> None:
        self._items.clear()
        self._sequence.clear()

    def items(self) -> list[QueueItem]:
        if self.policy == "fifo":
            return list(self._items)
        return sorted(
            self._items,
            key=lambda item: (
                item.absolute_deadline_s,
                float(item.task.arrival_time),
                int(item.task.task_id),
                self._sequence[id(item)],
            ),
        )

    def workload_s(self) -> float:
        return float(sum(max(0.0, i.remaining_service_time_s) for i in self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)
