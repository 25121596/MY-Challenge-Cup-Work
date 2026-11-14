import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import edge_run  # noqa: E402


def test_declared_ai4i_schema_error_is_not_downgraded_to_qwen(tmp_path):
    source = ROOT / "data" / "standard_inputs" / "ai4i_milling_inference_v1.json"
    task = copy.deepcopy(json.loads(source.read_text(encoding="utf-8"))[0])
    task["data"]["features"] = task["data"]["features"][:-1]
    input_path = tmp_path / "invalid_ai4i.json"
    input_path.write_text(json.dumps([task]), encoding="utf-8")

    results, latencies = edge_run.run_inference(str(input_path), "http://127.0.0.1:9", max_samples=1)

    assert len(results) == 1
    assert len(latencies) == 1
    result = results[0]
    assert result["route"] == "ai4i_tabular_moe_fail_closed"
    assert result["failure_probability"] is None
    assert result["local_prediction"] is None
    assert result["review_state"] == "required"
    assert result["review_reason"] == ["feature_schema_mismatch"]
    assert result["cloud_payload"]["model_route"] == "ai4i_tabular_moe_v1"
