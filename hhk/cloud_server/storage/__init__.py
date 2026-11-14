"""Cloud server storage layer.

In-memory stores that can be replaced with Redis or a database later.
"""

from .device_state import device_state
from .task_queue import task_queue_manager
from .models import (
    HeartbeatPayload,
    HiddenStates,
    InferenceContext,
    SplitInferenceRequest,
    FeatureUploadRequest,
    ChatCompletionRequest,
    ChatMessage,
    ConflictArbitrationRequest,
    ConflictArbitrationResponse,
    ModelUpdateResponse,
    LoRAUpdate,
    RuleUpdate,
    NeighborInfo,
    HealthResponse,
    TaskItem,
    DeviceGpu,
    DeviceCpu,
    DeviceMemory,
    DeviceNetwork,
    DeviceInference,
)

__all__ = [
    "device_state",
    "task_queue_manager",
    "HeartbeatPayload",
    "HiddenStates",
    "InferenceContext",
    "SplitInferenceRequest",
    "FeatureUploadRequest",
    "ChatCompletionRequest",
    "ChatMessage",
    "ConflictArbitrationRequest",
    "ConflictArbitrationResponse",
    "ModelUpdateResponse",
    "LoRAUpdate",
    "RuleUpdate",
    "NeighborInfo",
    "HealthResponse",
    "TaskItem",
    "DeviceGpu",
    "DeviceCpu",
    "DeviceMemory",
    "DeviceNetwork",
    "DeviceInference",
]
