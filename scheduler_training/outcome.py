"""Outcome contract separating environment transitions from reward calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models import Action, Task
from system_state import SystemState


@dataclass
class StepOutcome:
    task: Task | None
    action: Action
    state_before: SystemState
    state_after: SystemState
    timeout_count: int = 0

    selected_delay_s: float = 0.0
    expected_accuracy: float = 1.0
    upload_mb: float = 0.0
    edge_compute_s: float = 0.0
    cloud_compute_s: float = 0.0

    completed_task_ids: list[int] = field(default_factory=list)
    timeout_task_ids: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
