"""Pydantic models for request/response schemas and internal state."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


# ── Health ───────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    load: float = 0.0
    version: str = "1.2.0"
    max_split_layer: int = 28


# ── Heartbeat sub-models ─────────────────────────────────────────


class DeviceCpu(BaseModel):
    usage_pct: float = 0.0
    core_count: int = 1
    freq_mhz: float = 0.0


class DeviceMemory(BaseModel):
    total_mb: float = 0.0
    avail_mb: float = 0.0


class DeviceGpu(BaseModel):
    name: str = ""
    total_mb: float = 0.0
    free_mb: float = 0.0
    used_mb: float = 0.0
    device_type: int = 0


class DeviceNetwork(BaseModel):
    rtt_ms: float = 0.0
    packet_loss_pct: float = 0.0
    cloud_reachable: bool = True


class DeviceInference(BaseModel):
    current_tps: float = 0.0
    task_queue_len: int = 0


class HeartbeatPayload(BaseModel):
    device_id: str
    device_type: str = "unknown"
    timestamp_ms: int = 0
    is_fluctuating: bool = False
    cpu: DeviceCpu = Field(default_factory=DeviceCpu)
    memory: DeviceMemory = Field(default_factory=DeviceMemory)
    gpus: list[DeviceGpu] = Field(default_factory=list)
    network: DeviceNetwork = Field(default_factory=DeviceNetwork)
    inference: DeviceInference = Field(default_factory=DeviceInference)
    # Optional extra fields
    monitored_asset: str = ""
    camera_fov: str = ""
    model_version: int = 0
    current_lora_version: int = 0


# ── Task models ──────────────────────────────────────────────────


class TaskPayload(BaseModel):
    """Open-ended dict carried inside a task."""

    model_config = ConfigDict(extra="allow")


class TaskItem(BaseModel):
    type: str  # cloud_review | upload_features | model_update | rule_sync
    task_id: str
    description: str = ""
    priority: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Split Inference ──────────────────────────────────────────────


class HiddenStates(BaseModel):
    split_layer: int
    current_heads: int = 16
    head_dim: int = 128
    batch_size: int = 1
    seq_len: int = 512
    dtype: str = "float16"
    data_b64: str = ""
    data_bytes: int = 0
    checksum: str = ""


class InferenceContext(BaseModel):
    n_past: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 256
    prompt_tokens_b64: str = ""
    n_prompt_tokens: int = 0
    seed: int = 42
    stop_strings: list[str] = Field(default_factory=list)


class SplitDeviceInfo(BaseModel):
    device_id: str = ""
    avail_mem_mb: float = 0.0
    current_tps: float = 0.0
    network_rtt_ms: float = 0.0


class SplitInferenceRequest(BaseModel):
    model: str = ""
    hidden: HiddenStates
    context: InferenceContext = Field(default_factory=InferenceContext)
    device: SplitDeviceInfo = Field(default_factory=SplitDeviceInfo)


# ── Feature upload ──────────────────────────────────────────────


class FeatureMedia(BaseModel):
    type: str = "feature_vector"
    width: int = 224
    height: int = 224
    channels: int = 3
    mime: str = "image/jpeg"
    feature_shape: list[int] = Field(default_factory=list)
    features_b64: str = ""
    raw_bytes_b64: str = ""
    timestamp_us: int = 0


class FeatureUploadRequest(BaseModel):
    prompt: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = True
    media: FeatureMedia = Field(default_factory=FeatureMedia)


# ── Chat completions (OpenAI-compatible) ────────────────────────


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = Field(default_factory=list, min_length=1)
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512
    seed: int = 42
    stream: bool = True


# ── Model updates ───────────────────────────────────────────────


class LoRAUpdate(BaseModel):
    update_id: str
    url: str = ""
    local_path: str = ""
    scale: float = 1.0
    version: int = 0
    checksum: str = ""


class RuleUpdate(BaseModel):
    rule_id: str
    rule_type: str = "threshold"
    target: str = ""
    value: dict[str, Any] = Field(default_factory=dict)
    version: int = 0
    effective_at_ms: int = 0


class ModelUpdateResponse(BaseModel):
    lora_updates: list[LoRAUpdate] = Field(default_factory=list)
    rule_updates: list[RuleUpdate] = Field(default_factory=list)


# ── Neighbors ───────────────────────────────────────────────────


class NeighborInfo(BaseModel):
    node_id: str
    ip: str = ""
    udp_port: int = 15555
    tcp_port: int = 15556
    device_type: str = "unknown"
    asset: str = ""
    fov: str = ""


# ── Conflict arbitration ────────────────────────────────────────


class ConflictArbitrationRequest(BaseModel):
    conflict_id: str
    target_id: str
    target_type: str = "defect"
    our_node_id: str = ""
    our_decision: str = ""
    our_confidence: float = 0.0
    peer_node_id: str = ""
    peer_decision: str = ""
    peer_confidence: float = 0.0
    reason: str = ""
    severity: int = 0
    timestamp_ms: int = 0


class ConflictArbitrationResponse(BaseModel):
    decision: str
    reasoning: str = ""
    confidence: float = 0.0
