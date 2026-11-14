"""Pydantic models for request/response schemas and internal state."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Health ───────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    load: float = 0.0
    version: str = "1.2.0"
    max_split_layer: int = 28
    backend: str = "mock"
    inference_ready: bool = False
    split_inference_ready: bool = False


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
    deadline_at_ms: int = 0
    enqueued_at_ms: int = 0
    idempotency_key: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class EswEnvelope(BaseModel):
    """Identity-bearing HTTP envelope for one ESW-v1 4-byte payload."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["edge-esw-envelope/v1"] = "edge-esw-envelope/v1"
    device_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    audit_review_id: str | None = Field(default=None, min_length=1, max_length=128)
    esw_payload_b64: str = Field(min_length=1, max_length=32)


class EswEnvelopeResult(BaseModel):
    device_id: str
    request_id: str
    trace_id: str
    audit_review_id: str | None = None
    esw_status: Literal["ESW_OK", "ESW_INVALID", "ESW_UNSUPPORTED"]
    raw_word_hex: str | None = None
    rejection_reason: str | None = None
    cloud_action: Literal["unknown", "accept_local", "direct_cloud", "cloud_review"]
    policy_reason: str | None = None
    fields: dict[str, Any] | None = None


# ── Edge-to-cloud escalation contract ─────────────────────────────────────

class EscalationNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rtt_ms: float = 0.0
    bandwidth_kbps: float = 0.0
    packet_loss_pct: float = 0.0
    cloud_reachable: bool = True


class EscalationRequest(BaseModel):
    """Traceable request for model review, not a business-priority task."""
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["edge-cloud-review/v3"] = "edge-cloud-review/v3"
    review_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    from_node: str = Field(min_length=1, max_length=128)
    observed_at: str = Field(min_length=1, max_length=64)
    received_at: str = Field(min_length=1, max_length=64)
    decided_at: str = Field(min_length=1, max_length=64)
    deadline_at: str = Field(min_length=1, max_length=64)
    local_prediction: Literal["normal", "fault"] | None = None
    raw_logit: float | None = None
    failure_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_version: str | None = Field(default=None, max_length=128)
    feature_schema_id: str | None = Field(default=None, max_length=128)
    model_route: Literal["ai4i_tabular_moe_v1", "qwen_text_fallback_v1"]
    model_version: str = Field(min_length=1, max_length=128)
    policy_version: str = Field(min_length=1, max_length=128)
    threshold_used: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_review_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    review_state: Literal["required", "not_required"]
    review_reason: list[str] = Field(default_factory=list)
    safety_priority: Literal["P0", "P1", "P2"]
    safety_trigger: str | None = Field(default=None, max_length=64)
    action: str = Field(min_length=1, max_length=64)
    network: EscalationNetwork = Field(default_factory=EscalationNetwork)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_review_semantics(self) -> "EscalationRequest":
        hard_triggers = {
            "emergency_stop", "interlock_trip", "overtemperature_trip", "pressure_relief_trip",
        }
        if self.safety_trigger is not None and self.safety_trigger not in hard_triggers:
            raise ValueError("safety_trigger must be a recognized hard safety trigger")
        if self.safety_priority == "P0":
            if self.safety_trigger not in hard_triggers or self.action != "protect":
                raise ValueError("P0 requires a hard safety_trigger and action=protect")
        elif self.safety_trigger is not None:
            raise ValueError("a hard safety_trigger requires safety_priority=P0")
        elif self.action != "continue_with_review":
            raise ValueError("non-P0 cloud review requires action=continue_with_review")
        if self.review_state != "required":
            raise ValueError("the cloud-review endpoint accepts only review_state=required")
        if not self.review_reason:
            raise ValueError("a cloud-review request requires at least one review_reason")
        if len(set(self.review_reason)) != len(self.review_reason):
            raise ValueError("review_reason entries must not be duplicated")
        if self.model_route == "qwen_text_fallback_v1":
            if any(value is not None for value in (
                self.local_prediction, self.raw_logit, self.failure_probability, self.calibration_version,
                self.feature_schema_id, self.threshold_used, self.threshold_review_margin,
            )):
                raise ValueError("uncalibrated Qwen review must not claim a probability or local prediction")
            if "qwen_fallback_uncalibrated" not in self.review_reason:
                raise ValueError("uncalibrated Qwen review requires qwen_fallback_uncalibrated")
        if self.model_route == "ai4i_tabular_moe_v1":
            scored_fields = (
                self.local_prediction, self.failure_probability, self.calibration_version,
                self.feature_schema_id, self.threshold_used, self.threshold_review_margin,
            )
            if any(value is None for value in scored_fields):
                if any(value is not None for value in scored_fields):
                    raise ValueError("unscored AI4I review must null every model-score field")
                allowed_unscored = {"feature_schema_mismatch", "model_exception"}
                if not set(self.review_reason).issubset(allowed_unscored):
                    raise ValueError("unscored AI4I review requires a closed failure reason")
                expected_safety = "P0" if self.safety_trigger is not None else "P1"
                if self.safety_priority != expected_safety:
                    raise ValueError("unscored AI4I review safety priority is invalid")
                return self
            expected_prediction = "normal" if self.failure_probability <= self.threshold_used else "fault"
            if self.local_prediction != expected_prediction:
                raise ValueError("AI4I local_prediction does not match calibrated failure_probability")
            near_threshold = (
                self.threshold_review_margin > 0.0 and
                abs(self.failure_probability - self.threshold_used) <= self.threshold_review_margin
            )
            reason_set = set(self.review_reason)
            if near_threshold != ("near_normal_accept_threshold" in reason_set):
                raise ValueError("AI4I threshold-boundary review reason does not match policy margin")
            expected_safety = "P0" if self.safety_trigger is not None else (
                "P1" if expected_prediction == "fault" or near_threshold else "P2"
            )
            if self.safety_priority != expected_safety:
                raise ValueError("AI4I safety_priority does not match hard-safety and review policy")
            expected_review = self.safety_priority == "P0" or expected_prediction == "fault" or near_threshold
            if (self.review_state == "required") != expected_review:
                raise ValueError("AI4I review_state does not match the model and hard-safety policy")
        return self


class EscalationAck(BaseModel):
    review_id: str
    trace_id: str
    status: str  # accepted | duplicate | expired | rejected
    queue_position: int | None = None
    accepted_at: str
    deadline_at: str
    execution_status: str = "queued"
    storage: str = "in_memory"


class ReviewFeedback(BaseModel):
    """Post-review observation; cloud output is a teacher signal, not truth."""
    model_config = ConfigDict(extra="forbid")

    feedback_schema_version: Literal["edge-cloud-feedback/v1"] = "edge-cloud-feedback/v1"
    sample_id: str = Field(min_length=1, max_length=128)
    trace_id: str | None = Field(default=None, max_length=128)
    timestamp: str = Field(min_length=1, max_length=64)
    edge_prediction: Literal["normal", "fault"] | None = None
    cloud_prediction: Literal["normal", "fault"] | None = None
    disagreement: bool | None = None
    cloud_changed_decision: bool | None = None
    review_useful: bool | None = None
    human_label: str | None = Field(default=None, max_length=128)
    confirmed_outcome: str | None = Field(default=None, max_length=128)
    label_source: str | None = Field(default=None, max_length=64)
    available_at: str | None = Field(default=None, max_length=64)
    versions: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_comparison(self) -> "ReviewFeedback":
        if self.edge_prediction is not None and self.cloud_prediction is not None:
            expected = self.edge_prediction != self.cloud_prediction
            if self.disagreement is not None and self.disagreement != expected:
                raise ValueError("disagreement does not match edge/cloud predictions")
        if self.review_useful is not None and not (self.human_label or self.confirmed_outcome or self.label_source):
            raise ValueError("review_useful requires an explicit outcome source")
        return self


class ReviewFeedbackAck(BaseModel):
    sample_id: str
    trace_id: str | None = None
    status: str = "accepted"
    storage: str = "in_memory"


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
    cache_prompt: bool = False
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


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
