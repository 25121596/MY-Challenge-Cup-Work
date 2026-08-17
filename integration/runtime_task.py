from __future__ import annotations

"""Integration-owned task model for the real wall-clock runtime."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    TRANSMITTING = "transmitting"
    COMPLETED = "completed"
    TIMEOUT = "timeout"


@dataclass
class RuntimeTask:
    """A task parsed from edge input and executed by Integration.

    The confidence field is an input/profile prior retained for diagnostics only.
    It is not part of the 24D pre-route observation and is not used by Post-edge,
    which evaluates signals measured from the actual edge-model response.
    """

    task_id: int
    arrival_time: int
    size_mb: float
    deadline_s: float
    confidence: float

    external_task_id: str | None = None
    node_id: str | None = None
    source_timestamp: str | None = None
    source_arrival_offset_s: float | None = None
    task_type: str = "generic_anomaly"
    priority: str = "medium"
    device_id: str | None = None
    device_type: str | None = None

    complexity: float = 0.5
    edge_base_latency_s: float = 0.15
    cloud_base_latency_s: float = 0.15
    estimated_cloud_accuracy: float = 0.94
    collaborative_edge_latency_s: float = 0.08
    collaborative_cloud_latency_s: float = 0.10
    collaborative_accuracy: float = 0.97
    collaborative_payload_ratio: float = 0.20

    edge_input: dict[str, Any] = field(default_factory=dict)
    confidence_source: str = "profile_prior"
    size_source: str = "profile_prior"
    deadline_source: str = "profile_prior"
    performance_source: str = "profile_prior"
    source: str = "edge_input"

    status: TaskStatus = TaskStatus.WAITING
    execution_stage: str = "NOT_STARTED"
