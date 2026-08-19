import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge-io-protocol"))
sys.path.insert(0, str(ROOT))

from feedback_loop import build_observation, compare_predictions, summarize  # noqa: E402


def test_comparison_is_unlabeled_when_cloud_is_unavailable():
    result = compare_predictions({"local_prediction": "fault"}, None)
    assert result["disagreement"] is None
    assert result["cloud_changed_decision"] is None


def test_observation_retains_raw_logit_and_does_not_call_disagreement_truth():
    task = {"task_id": "t-1", "task_type": "milling", "data": {"features": [1, 2]}}
    edge = {"task_id": "t-1", "model_route": "ai4i_tabular_moe_v1", "model_version": "m-1",
            "raw_logit": -1.2, "failure_probability": .2, "local_prediction": "normal",
            "review_state": "not_required", "review_reason": [], "latency_ms": 2.0}
    cloud = {"model_version": "cloud-1", "cloud_prediction": "fault", "raw_output": "fault"}
    row = build_observation(task, edge, cloud_output=cloud)
    assert row["edge"]["raw_logit"] == -1.2
    assert row["comparison"]["disagreement"] is True
    assert row["outcome"]["review_useful"] is None
    assert summarize([row])["unlabeled_records"] == 1


def test_review_useful_requires_explicit_source():
    task = {"task_id": "t-2", "data": {}}
    edge = {"local_prediction": "normal"}
    with pytest.raises(ValueError, match="outcome source"):
        build_observation(task, edge, outcome={"review_useful": True})


def test_legacy_fields_are_rejected():
    task = {"task_id": "t-3", "data": {"priority": "P1"}}
    with pytest.raises(ValueError, match="legacy"):
        build_observation(task, {"local_prediction": "normal"})

