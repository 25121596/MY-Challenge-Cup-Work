"""Contract tests for edge-to-cloud escalation admission."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import app
from storage.escalation_queue import EscalationQueue
from storage.models import EscalationRequest


def payload(key: str = "idem-1", safety_priority: str = "P1") -> dict:
    now = datetime.now(timezone.utc)
    return {"schema_version": "edge-cloud-review/v3", "review_id": "review-" + key,
            "trace_id": "trace-1", "idempotency_key": key, "task_id": "task-1", "from_node": "edge-1",
            "observed_at": now.isoformat(), "received_at": now.isoformat(), "decided_at": now.isoformat(),
            "deadline_at": (now + timedelta(seconds=60)).isoformat(), "local_prediction": "fault",
            "failure_probability": .62, "calibration_version": "ai4i_platt:test",
            "feature_schema_id": "ai4i_8d_v1", "model_route": "ai4i_tabular_moe_v1",
            "model_version": "ai4i_tabular_moe:test", "policy_version": "policy:test", "threshold_used": .02751,
            "threshold_review_margin": 0.0,
            "review_state": "required", "review_reason": ["failure_probability_above_normal_accept_threshold"],
            "safety_priority": safety_priority, "safety_trigger": None, "action": "continue_with_review",
            "network": {"rtt_ms": 120, "cloud_reachable": True}, "evidence": {"feature_hash": "sha256:test"}}


def test_escalation_admission_idempotency_and_lookup():
    with TestClient(app) as client:
        first = client.post("/api/v1/escalations", json=payload())
        assert first.status_code == 202
        assert first.json()["status"] == "accepted"
        second = client.post("/api/v1/escalations", json=payload())
        assert second.status_code == 202
        assert second.json()["status"] == "duplicate"
        queried = client.get("/api/v1/escalations/review-idem-1")
        assert queried.status_code == 200
        assert queried.json()["execution_status"] == "queued"


def test_escalation_rejects_expired_deadline():
    body = payload("expired")
    body["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with TestClient(app) as client:
        response = client.post("/api/v1/escalations", json=body)
    assert response.status_code == 202
    assert response.json()["status"] == "expired"


def test_p0_cannot_be_forged_from_a_model_claim():
    body = payload("forged-p0", "P0")
    with TestClient(app) as client:
        response = client.post("/api/v1/escalations", json=body)
    assert response.status_code == 422


def test_escalation_rejects_removed_unified_confidence_field():
    body = payload("forbidden-confidence")
    body["confidence"] = 0.99
    with TestClient(app) as client:
        response = client.post("/api/v1/escalations", json=body)
    assert response.status_code == 422


def test_cloud_accepts_explicit_ai4i_boundary_review():
    body = payload("near-boundary")
    body.update({
        "local_prediction": "normal",
        "failure_probability": .027,
        "threshold_review_margin": .001,
        "review_reason": ["near_normal_accept_threshold"],
        "safety_priority": "P1",
    })
    with TestClient(app) as client:
        response = client.post("/api/v1/escalations", json=body)
    assert response.status_code == 202


def test_cloud_accepts_unscored_ai4i_fail_closed_review():
    body = payload("schema-mismatch")
    body.update({
        "local_prediction": None,
        "failure_probability": None,
        "calibration_version": None,
        "feature_schema_id": None,
        "threshold_used": None,
        "threshold_review_margin": None,
        "policy_version": "ai4i_fail_closed_review_v1",
        "review_reason": ["feature_schema_mismatch"],
        "safety_priority": "P1",
    })
    with TestClient(app) as client:
        response = client.post("/api/v1/escalations", json=body)
    assert response.status_code == 202


def test_queue_orders_safety_urgency_before_deadline():
    queue = EscalationQueue()
    normal_review = EscalationRequest(**payload("queue-p1", "P1"))
    urgent_review_body = payload("queue-p0", "P0")
    urgent_review_body["safety_trigger"] = "interlock_trip"
    urgent_review_body["action"] = "protect"
    urgent_review_body["deadline_at"] = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    urgent_review = EscalationRequest(**urgent_review_body)
    queue.submit(normal_review)
    queue.submit(urgent_review)
    assert queue.get("review-queue-p0")["queue_position"] == 1
    assert queue.get("review-queue-p1")["queue_position"] == 2


def test_health_explicitly_marks_mock_as_not_inference_ready():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["backend"] == "mock"
    assert response.json()["inference_ready"] is False
    assert response.json()["split_inference_ready"] is False


def test_review_feedback_is_stored_without_promoting_cloud_to_truth():
    body = payload("feedback-1")
    with TestClient(app) as client:
        admitted = client.post("/api/v1/escalations", json=body)
        assert admitted.status_code == 202
        feedback = {
            "feedback_schema_version": "edge-cloud-feedback/v1",
            "sample_id": "task-1",
            "trace_id": body["trace_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "edge_prediction": "fault",
            "cloud_prediction": "normal",
            "disagreement": True,
            "cloud_changed_decision": True,
            "review_useful": None,
            "versions": {"cloud_model_version": "qwen-full:test"},
            "evidence": {"label_kind": "teacher_signal"},
        }
        response = client.post("/api/v1/escalations/review-feedback-1/feedback", json=feedback)
        assert response.status_code == 202
        stored = client.get("/api/v1/escalations/review-feedback-1")
        assert stored.json()["feedback"]["review_useful"] is None
