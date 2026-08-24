"""Auditable edge decision and cloud-review contracts.

The model reports risk; code owns routing and protocol construction.  In
particular, no field named ``confidence`` is emitted: it had mixed three
incompatible meanings in earlier revisions.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any


AI4I_FEATURE_SCHEMA_ID = "ai4i_8d_v1"
AI4I_MODEL_ROUTE = "ai4i_tabular_moe_v1"
HARD_SAFETY_TRIGGERS = {
    "emergency_stop", "interlock_trip", "overtemperature_trip", "pressure_relief_trip",
}
AI4I_UNSCORED_REVIEW_REASONS = {"feature_schema_mismatch", "model_exception"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _valid_hard_trigger(task: dict[str, Any]) -> str | None:
    trigger = task.get("data", {}).get("safety_trigger")
    return trigger if trigger in HARD_SAFETY_TRIGGERS else None


def safety_fields(task: dict[str, Any], review_required: bool) -> dict[str, Any]:
    """Safety is independent from classifier probability and business value."""
    trigger = _valid_hard_trigger(task)
    if trigger:
        return {"safety_priority": "P0", "safety_trigger": trigger, "action": "protect"}
    if review_required:
        return {"safety_priority": "P1", "safety_trigger": None, "action": "continue_with_review"}
    return {"safety_priority": "P2", "safety_trigger": None, "action": "continue"}


def ai4i_decision_fields(task: dict[str, Any], failure_probability: float,
                         normal_accept_threshold: float, model_version: str,
                         calibration_version: str, policy_version: str,
                         threshold_review_margin: float = 0.0) -> dict[str, Any]:
    """Create the complete, explicit result for calibrated AI4I inference."""
    p = min(1.0, max(0.0, float(failure_probability)))
    threshold = min(1.0, max(0.0, float(normal_accept_threshold)))
    review_margin = min(1.0, max(0.0, float(threshold_review_margin)))
    local_prediction = "normal" if p <= threshold else "fault"
    hard_trigger = _valid_hard_trigger(task)
    # The margin is an explicit operational policy, not a substitute for
    # calibration.  Zero preserves the frozen baseline behaviour.
    near_threshold = review_margin > 0.0 and abs(p - threshold) <= review_margin
    review_required = local_prediction == "fault" or near_threshold or hard_trigger is not None
    review_reason = []
    if local_prediction == "fault":
        review_reason.append("failure_probability_above_normal_accept_threshold")
    if near_threshold:
        review_reason.append("near_normal_accept_threshold")
    if hard_trigger is not None:
        review_reason.append("hard_safety_trigger")
    result = {
        "local_prediction": local_prediction,
        "failure_probability": round(p, 6),
        "calibration_version": calibration_version,
        "feature_schema_id": AI4I_FEATURE_SCHEMA_ID,
        "model_route": AI4I_MODEL_ROUTE,
        "model_version": model_version,
        "policy_version": policy_version,
        "threshold_used": round(threshold, 6),
        "threshold_review_margin": round(review_margin, 6),
        "review_state": "required" if review_required else "not_required",
        "review_reason": review_reason,
    }
    result.update(safety_fields(task, review_required))
    return result


def ai4i_unscored_review_fields(task: dict[str, Any], reason: str,
                                model_version: str = "ai4i_tabular_moe:unavailable") -> dict[str, Any]:
    """Fail closed when the frozen AI4I contract or runtime is unavailable.

    There is deliberately no synthetic probability here.  The record remains
    traceable as an AI4I-route review, but it cannot be mistaken for a model
    result.  ``reason`` is intentionally a small closed vocabulary so that a
    caller cannot silently turn arbitrary diagnostics into routing policy.
    """
    if reason not in AI4I_UNSCORED_REVIEW_REASONS:
        raise ValueError(f"unsupported unscored AI4I review reason: {reason}")
    result = {
        "local_prediction": None,
        "raw_logit": None,
        "failure_probability": None,
        "calibration_version": None,
        "feature_schema_id": None,
        "model_route": AI4I_MODEL_ROUTE,
        "model_version": model_version,
        "policy_version": "ai4i_fail_closed_review_v1",
        "threshold_used": None,
        "threshold_review_margin": None,
        "review_state": "required",
        "review_reason": [reason],
    }
    result.update(safety_fields(task, review_required=True))
    return result


def qwen_fallback_fields(task: dict[str, Any]) -> dict[str, Any]:
    """Uncalibrated Qwen output is never promoted to a probability."""
    result = {
        "local_prediction": None,
        "raw_logit": None,
        "failure_probability": None,
        "calibration_version": None,
        "feature_schema_id": None,
        "model_route": "qwen_text_fallback_v1",
        "model_version": "Qwen3-1.7B-IQ4_XS",
        "policy_version": "qwen_fallback_requires_review_v1",
        "threshold_used": None,
        "threshold_review_margin": None,
        "review_state": "required",
        "review_reason": ["qwen_fallback_uncalibrated"],
    }
    result.update(safety_fields(task, review_required=True))
    return result


def compact_cloud_payload(task: dict[str, Any], output: dict[str, Any],
                          network: dict[str, Any] | None = None) -> dict[str, Any]:
    """Construct a traceable cloud-review request, not a business-priority task."""
    now = datetime.now(timezone.utc)
    task_id = str(task.get("task_id", "auto"))
    node_id = str(task.get("node_id", "edge-01"))
    observed_at = str(task.get("observed_at", task.get("timestamp", "")))
    decided_at = str(output.get("decided_at", output.get("timestamp", utc_now())))
    stable = json.dumps({"task_id": task_id, "node_id": node_id, "observed_at": observed_at,
                         "failure_probability": output.get("failure_probability"),
                         "raw_logit": output.get("raw_logit"),
                         "review_reason": output.get("review_reason", [])}, sort_keys=True)
    digest = hashlib.sha256(stable.encode()).hexdigest()
    deadline = now + timedelta(seconds=30 if output.get("safety_priority") == "P0" else 120)
    features = task.get("data", {}).get("features", [])
    evidence_hash = hashlib.sha256(json.dumps(features, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": "edge-cloud-review/v3",
        "review_id": "review-" + digest[:20],
        "trace_id": "trace-" + digest[20:40],
        "idempotency_key": "idem-" + digest[:24],
        "task_id": task_id,
        "from_node": node_id,
        "observed_at": observed_at,
        "received_at": output.get("received_at", decided_at),
        "decided_at": decided_at,
        "deadline_at": deadline.isoformat().replace("+00:00", "Z"),
        "local_prediction": output.get("local_prediction"),
        "failure_probability": output.get("failure_probability"),
        "raw_logit": output.get("raw_logit"),
        "calibration_version": output.get("calibration_version"),
        "feature_schema_id": output.get("feature_schema_id"),
        "model_route": output.get("model_route"),
        "model_version": output.get("model_version"),
        "policy_version": output.get("policy_version"),
        "threshold_used": output.get("threshold_used"),
        "threshold_review_margin": output.get("threshold_review_margin"),
        "review_state": output.get("review_state"),
        "review_reason": output.get("review_reason", []),
        "safety_priority": output.get("safety_priority"),
        "safety_trigger": output.get("safety_trigger"),
        "action": output.get("action"),
        "network": network or {"cloud_reachable": True},
        "evidence": {"feature_count": len(features), "feature_hash": "sha256:" + evidence_hash},
    }
