import copy
import json
import subprocess
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


def test_phase_a_routing_shadow_is_written_for_eligible_local_record(tmp_path):
    source = ROOT / "data" / "standard_inputs" / "ai4i_milling_inference_v1.json"
    task = json.loads(source.read_text(encoding="utf-8"))[0]
    input_path = tmp_path / "ai4i.json"
    feedback_path = tmp_path / "shadow.jsonl"
    input_path.write_text(json.dumps([task]), encoding="utf-8")

    results, _ = edge_run.run_inference(
        str(input_path), "http://127.0.0.1:9", max_samples=1,
        feedback_log=str(feedback_path), routing_shadow_probability=1.0,
        routing_network={"cloud_reachable": True, "rtt_ms": 20},
    )

    assert results[0]["routing_shadow"]["population_state"] == "randomized_shadow"
    observation = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert observation["routing"]["upload_probability"] == 1.0

    manifest_path = tmp_path / "shadow.manifest.json"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "summarize_routing_shadow.py"), str(feedback_path),
         "--output", str(manifest_path)],
        check=True,
        cwd=ROOT,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["population_state_counts"] == {"randomized_shadow": 1}
