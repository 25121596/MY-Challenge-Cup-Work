from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Action(IntEnum):
    """仿真器内部动作枚举。

    Stable-Baselines3 的离散动作索引仍是 0/1/2，并在 ppo_core.py 中映射为
    EDGE/CLOUD/COLLABORATIVE。IDLE 只供没有新任务的物理时隙使用。
    """

    IDLE = 0
    EDGE = 1
    CLOUD = 2
    COLLABORATIVE = 3


class TaskStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    TRANSMITTING = "transmitting"
    COMPLETED = "completed"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """工业推理任务。

    前五个字段保持 Phase1/Phase2 的构造兼容。Phase3 新增协同路径参数和
    阶段时间戳，但仍不要求 hhk input 提供当前任务执行后的真实模型输出。
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

    # 当前阶段用于模拟 1.5B 边端模型和 7B 云端模型的先验。
    complexity: float = 0.5
    edge_base_latency_s: float = 0.15
    cloud_base_latency_s: float = 0.15
    estimated_cloud_accuracy: float = 0.94

    # Phase3：边端初判 -> 传输 -> 云端复核路径的先验。
    collaborative_edge_latency_s: float = 0.08
    collaborative_cloud_latency_s: float = 0.10
    collaborative_accuracy: float = 0.97
    collaborative_payload_ratio: float = 0.20

    edge_input: dict[str, Any] = field(default_factory=dict)
    confidence_source: str = "profile_prior"
    size_source: str = "profile_prior"
    deadline_source: str = "profile_prior"
    performance_source: str = "profile_prior"
    source: str = "typed_simulation"

    route: Action | None = None
    status: TaskStatus = TaskStatus.WAITING
    start_time: int | None = None
    finish_time: int | None = None

    # 协同任务生命周期。时间均为离散 slot 索引。
    execution_stage: str = "NOT_STARTED"
    edge_stage_start_time: int | None = None
    edge_stage_finish_time: int | None = None
    transmission_start_time: int | None = None
    transmission_ready_time: int | None = None
    cloud_stage_start_time: int | None = None

    edge_stage_done: bool = False
    uploaded_to_cloud: bool = False
    edge_draft_result: dict[str, Any] | None = None
    collaborative_request: dict[str, Any] | None = None
