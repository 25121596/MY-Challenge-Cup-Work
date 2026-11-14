import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from ai4i_moe_inference import Ai4iMoeInference  # noqa: E402


def online_task() -> dict:
    return json.loads((ROOT / "data" / "standard_inputs" / "ai4i_milling_inference_v1.json")
                      .read_text(encoding="utf-8"))[0]


def test_ai4i_emits_calibrated_probability_not_unified_confidence():
    output = Ai4iMoeInference().infer(online_task())
    assert output["model_route"] == "ai4i_tabular_moe_v1"
    assert output["local_prediction"] in {"normal", "fault"}
    assert 0.0 <= output["failure_probability"] <= 1.0
    assert output["calibration_version"].startswith("ai4i_platt:")
    assert output["review_state"] in {"required", "not_required"}
    assert "confidence" not in output


def test_ai4i_rejects_duplicate_readings_source():
    task = copy.deepcopy(online_task())
    task["data"]["readings"] = {"not_allowed": 1.0}
    with pytest.raises(ValueError, match="features-only"):
        Ai4iMoeInference().infer(task)


def test_ai4i_rejects_non_one_hot_type_code():
    task = copy.deepcopy(online_task())
    task["data"]["features"][:3] = [1, 1, 0]
    with pytest.raises(ValueError, match="one-hot"):
        Ai4iMoeInference().infer(task)
