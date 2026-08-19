import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge-io-protocol"))

from decision_contract import (  # noqa: E402
    ai4i_decision_fields,
    ai4i_unscored_review_fields,
    compact_cloud_payload,
    qwen_fallback_fields,
)
from edge_inference import parse_short_decision  # noqa: E402


def task() -> dict:
    return {"task_id": "T-1", "node_id": "edge-01", "observed_at": "2026-08-07T00:00:00Z",
            "task_type": "status_monitoring", "data": {"device_id": "milling-machine-01", "features": [0.0] * 8}}


def test_ai4i_probability_is_not_a_unified_confidence():
    normal = ai4i_decision_fields(task(), .001, .02751, "model", "cal", "policy")
    review = ai4i_decision_fields(task(), .2, .02751, "model", "cal", "policy")
    assert normal["local_prediction"] == "normal"
    assert normal["review_state"] == "not_required"
    assert normal["safety_priority"] == "P2"
    assert review["local_prediction"] == "fault"
    assert review["review_state"] == "required"
    assert review["safety_priority"] == "P1"
    assert "confidence" not in review


def test_p0_requires_hard_safety_trigger():
    t = task()
    t["data"]["safety_trigger"] = "interlock_trip"
    result = ai4i_decision_fields(t, .001, .02751, "model", "cal", "policy")
    assert result["safety_priority"] == "P0"
    assert result["action"] == "protect"
    assert result["review_state"] == "required"
    assert "hard_safety_trigger" in result["review_reason"]


def test_qwen_probability_is_explicitly_unavailable():
    result = qwen_fallback_fields(task())
    assert result["failure_probability"] is None
    assert result["review_state"] == "required"
    assert "confidence" not in result


def test_qwen_rejects_legacy_score_fields_instead_of_ignoring_them():
    category, error = parse_short_decision('{"c": 1, "confidence": 0.99}')
    assert category is None
    assert error == "FORMAT_ERROR"


def test_ai4i_threshold_review_margin_is_explicit_and_does_not_change_prediction():
    result = ai4i_decision_fields(task(), .027, .02751, "model", "cal", "policy",
                                  threshold_review_margin=.001)
    assert result["local_prediction"] == "normal"
    assert result["review_state"] == "required"
    assert result["review_reason"] == ["near_normal_accept_threshold"]
    assert result["safety_priority"] == "P1"
    assert result["threshold_review_margin"] == .001


def test_ai4i_schema_or_runtime_failure_is_unscored_and_fails_closed():
    result = ai4i_unscored_review_fields(task(), "feature_schema_mismatch")
    assert result["model_route"] == "ai4i_tabular_moe_v1"
    assert result["failure_probability"] is None
    assert result["local_prediction"] is None
    assert result["review_state"] == "required"
    assert result["safety_priority"] == "P1"
    assert result["review_reason"] == ["feature_schema_mismatch"]


def test_cloud_payload_is_traceable_and_not_business_priority():
    output = ai4i_decision_fields(task(), .2, .02751, "model", "cal", "policy")
    output.update({"received_at": "2026-08-07T00:00:01Z", "decided_at": "2026-08-07T00:00:02Z"})
    payload = compact_cloud_payload(task(), output)
    assert payload["failure_probability"] == .2
    assert payload["threshold_review_margin"] == 0.0
    assert payload["review_state"] == "required"
    assert payload["safety_priority"] == "P1"
    assert "priority" not in payload
    assert {"task_id", "trace_id", "idempotency_key", "deadline_at", "evidence"} <= payload.keys()
