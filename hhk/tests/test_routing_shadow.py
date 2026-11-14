import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge-io-protocol"))

from routing_shadow import select_shadow, validate_routing_selection  # noqa: E402


def task() -> dict:
    return {
        "task_id": "routing-1",
        "task_type": "status_monitoring",
        "observed_at": "2026-08-23T00:00:00Z",
        "data": {"network": {"cloud_reachable": True}},
    }


def edge(review_state: str = "not_required", priority: str = "P2") -> dict:
    return {
        "task_id": "routing-1",
        "decided_at": "2026-08-23T00:00:01Z",
        "review_state": review_state,
        "safety_priority": priority,
        "failure_probability": .02,
        "raw_logit": -2.1,
        "decision": {"detail": {"gate_weights": [.4, .6]}},
    }


def test_shadow_selection_is_deterministic_and_records_propensity():
    first = select_shadow(task(), edge(), upload_probability=1.0, seed="fixed")
    second = select_shadow(task(), edge(), upload_probability=1.0, seed="fixed")
    assert first == second
    assert first["population_state"] == "randomized_shadow"
    assert first["upload_probability"] == 1.0
    validate_routing_selection(first)


def test_baseline_upload_is_not_reclassified_as_random_shadow():
    selected = select_shadow(task(), edge("required", "P1"), upload_probability=.5, seed="fixed")
    assert selected["population_state"] == "policy_uploaded"
    assert selected["baseline_upload"] is True
    assert selected["upload_probability"] == 1.0


def test_p0_and_network_failure_are_outside_router_population():
    p0 = select_shadow(task(), edge("required", "P0"), upload_probability=1.0, seed="fixed")
    assert p0["population_state"] == "structurally_ineligible"
    assert p0["selection_reason"] == "hard_safety_p0"
    offline_task = task()
    offline_task["data"]["network"] = {"cloud_reachable": False}
    offline = select_shadow(offline_task, edge(), upload_probability=1.0, seed="fixed")
    assert offline["selection_reason"] == "cloud_unreachable"


def test_expired_deadline_is_not_eligible():
    expired = task()
    expired["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    selection = select_shadow(expired, edge(), upload_probability=1.0, seed="fixed")
    assert selection["population_state"] == "structurally_ineligible"
    assert selection["selection_reason"] == "deadline_expired"


def test_deadline_below_controlled_cloud_budget_is_not_eligible():
    limited = task()
    limited["deadline_at"] = (datetime.now(timezone.utc) + timedelta(milliseconds=10)).isoformat()
    selection = select_shadow(
        limited, edge(), upload_probability=1.0, seed="fixed",
        network={"cloud_reachable": True, "minimum_cloud_completion_ms": 20},
    )
    assert selection["population_state"] == "structurally_ineligible"
    assert selection["selection_reason"] == "deadline_budget_insufficient"


def test_invalid_probability_is_rejected():
    with pytest.raises(ValueError, match="upload_probability"):
        select_shadow(task(), edge(), upload_probability=1.1, seed="fixed")
