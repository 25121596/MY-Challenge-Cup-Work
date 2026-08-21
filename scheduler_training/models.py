from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Action(IntEnum):
    """Pre-route actions plus the internal post-edge cloud-review route.

    Gym exposes only LOCAL_FIRST and DIRECT_CLOUD. CLOUD_REVIEW is created by
    PostEdgePolicy after the edge result exists and is never a PPO action.
    """

    IDLE = 0
    LOCAL_FIRST = 1
    DIRECT_CLOUD = 2
    CLOUD_REVIEW = 3


# Gym/PPO indices are deliberately separate from internal lifecycle route values.
PPO_ACTION_TO_ROUTE = {
    0: Action.LOCAL_FIRST,
    1: Action.DIRECT_CLOUD,
}
ROUTE_TO_PPO_ACTION = {route: index for index, route in PPO_ACTION_TO_ROUTE.items()}


class TaskStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    TRANSMITTING = "transmitting"
    COMPLETED = "completed"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """Industrial inference task used by Phase4.5.

    Phase4.5 changes time fields from discrete slot indices to continuous seconds,
    while keeping all task/profile numeric parameters unchanged.
    """

    task_id: int
    arrival_time: float
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

    # Numeric priors are intentionally unchanged from Phase4.
    complexity: float = 0.5
    edge_base_latency_s: float = 0.15
    cloud_base_latency_s: float = 0.15
    estimated_cloud_accuracy: float = 0.94

    collaborative_edge_latency_s: float = 0.08
    collaborative_cloud_latency_s: float = 0.10
    collaborative_accuracy: float = 0.97
    collaborative_payload_ratio: float = 0.20

    edge_input: dict[str, Any] = field(default_factory=dict)
    edge_result_evidence: dict[str, Any] | None = None
    edge_result_source: str = "simulated_profile_result"
    edge_latency_source: str = "profile_prior"
    confidence_source: str = "profile_prior"
    size_source: str = "profile_prior"
    deadline_source: str = "profile_prior"
    performance_source: str = "profile_prior"
    source: str = "typed_simulation"

    route: Action | None = None
    pre_route_action: Action | None = None
    final_route: str | None = None
    post_edge_decision: str | None = None
    post_edge_reason: str | None = None
    post_edge_rule_version: str | None = None
    status: TaskStatus = TaskStatus.WAITING
    start_time: float | None = None
    finish_time: float | None = None

    execution_stage: str = "NOT_STARTED"
    edge_stage_start_time: float | None = None
    edge_stage_finish_time: float | None = None
    transmission_start_time: float | None = None
    transmission_ready_time: float | None = None
    cloud_stage_start_time: float | None = None

    edge_stage_done: bool = False
    uploaded_to_cloud: bool = False
    edge_draft_result: dict[str, Any] | None = None
    collaborative_request: dict[str, Any] | None = None

    @property
    def absolute_deadline_s(self) -> float:
        return float(self.arrival_time) + float(self.deadline_s)
