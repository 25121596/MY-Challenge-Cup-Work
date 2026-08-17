"""Phase4.5 unified system-state contract.

Training and real deployment should expose the same semantic state fields.  The
simulator uses :class:`SystemState`; Integration v0.3 can map its
RealStateProvider output to the same keys before calling build_observation().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class SystemState(Mapping[str, Any]):
    current_time_s: float
    edge_queue: int
    cloud_queue: int
    edge_workload_s: float
    cloud_workload_s: float
    edge_resource: float
    cloud_resource: float
    network_quality: float
    bandwidth_mbps: float

    edge_busy: bool = False
    cloud_busy: bool = False
    cloud_ready_queue: int = 0
    transmission_queue: int = 0
    cloud_ready_workload_s: float = 0.0
    reserved_cloud_workload_s: float = 0.0
    edge_active_task: int | None = None
    cloud_active_task: int | None = None
    network_state: str = "GOOD"
    queue_policy: str = "edf"
    network_mode: str = "markov"
    resource_mode: str = "consume_recover"
    reward_mode: str = "deadline_aware"
    task_source: str = "typed_simulation"
    external_tasks_remaining: int = 0
    edge_inputs_loaded: int = 0

    @property
    def current_time(self) -> float:
        """Compatibility alias. Phase4.5 time is seconds, not slot index."""
        return self.current_time_s

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["current_time"] = self.current_time_s
        return data

    # Mapping interface keeps old code using state["..."] / state.get(...) valid.
    def __getitem__(self, key: str) -> Any:
        if key == "current_time":
            return self.current_time_s
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        yield from self.to_dict()

    def __len__(self) -> int:
        return len(self.to_dict())
