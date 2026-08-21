from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hhk_v2_replay import load_hhk_v2_replay
from models import Task
from post_edge_policy import DEFAULT_RULES_PATH, PostEdgeDecision, PostEdgePolicy
from scheduler import AlwaysLocalFirstScheduler
from simulator import CloudEdgeSimulator


def _input(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "node_id": "edge-01",
        "timestamp": "2026-08-06T13:25:20",
        "task_type": "status_monitoring",
        "max_latency_ms": 10000,
        "data": {
            "features": [0.0] * 8,
            "feature_schema_id": "ai4i_8d_v1",
            "model_route": "ai4i_tabular_moe_v1",
        },
    }


def _result(task_id: str, probability: float, state: str) -> dict:
    return {
        "task_id": task_id,
        "latency_ms": 2.5,
        "failure_probability": probability,
        "local_prediction": "normal",
        "calibration_version": "ai4i_platt:test",
        "feature_schema_id": "ai4i_8d_v1",
        "model_route": "ai4i_tabular_moe_v1",
        "model_version": "ai4i_tabular_moe:test",
        "threshold_used": 0.03,
        "review_state": state,
        "safety_priority": "P2",
        "safety_trigger": False,
    }


class HhkV2ReplayTests(unittest.TestCase):
    def test_join_report_and_timestamp_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "inputs.json"
            result_path = root / "results.json"
            input_path.write_text(json.dumps([_input("a"), _input("b"), _input("c")]), encoding="utf-8")
            result_path.write_text(json.dumps({"results": [_result("a", 0.01, "not_required"), _result("b", 0.08, "required")]}), encoding="utf-8")
            tasks, _, report = load_hhk_v2_replay(input_path, result_path, replay_interval_s=0.2)
            self.assertEqual(len(tasks), 2)
            self.assertEqual(report.matched_count, 2)
            self.assertEqual(report.unmatched_input_count, 1)
            self.assertAlmostEqual(report.join_coverage, 2 / 3)
            self.assertEqual(report.arrival_timing_source, "configured_replay_interval")
            self.assertEqual([task.source_arrival_offset_s for task in tasks], [0.0, 0.2])
            self.assertEqual(tasks[0].edge_result_source, "hhk_v2_edge_result_replay")

    def test_failure_probability_rule_direction(self):
        policy = PostEdgePolicy.from_json(DEFAULT_RULES_PATH)
        task = Task(1, 0.0, 0.1, 10.0, 0.9, task_type="status_monitoring")
        accept, _, _ = load_hhk_v2_evidence_for_test(policy, task, _result("a", 0.01, "not_required"))
        review, _, _ = load_hhk_v2_evidence_for_test(policy, task, _result("b", 0.08, "required"))
        self.assertEqual(accept.decision, PostEdgeDecision.ACCEPT_EDGE)
        self.assertEqual(review.decision, PostEdgeDecision.CLOUD_REVIEW)
        self.assertEqual(review.reason, "HIGH_FAILURE_PROBABILITY")
        legacy = PostEdgePolicy.from_json(
            Path(__file__).resolve().parents[1] / "config" / "post_edge_rules_v1.json"
        )
        self.assertEqual(legacy.schema_version, "post-edge-rules/1.0")

    def test_simulator_consumes_replayed_result_after_local_first(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "inputs.json"
            result_path = root / "results.json"
            input_path.write_text(json.dumps([_input("a"), _input("b")]), encoding="utf-8")
            result_path.write_text(json.dumps({"results": [
                _result("a", 0.01, "not_required"),
                _result("b", 0.08, "required"),
            ]}), encoding="utf-8")
            simulator = CloudEdgeSimulator(
                scheduler=AlwaysLocalFirstScheduler(),
                task_source="hhk_v2_replay",
                edge_input_path=str(input_path),
                edge_result_path=str(result_path),
            )
            simulator.run_until_complete(max_steps=20)
            completed = simulator.get_completed_tasks().set_index("external_task_id")
            self.assertEqual(completed.loc["a", "route"], "EDGE")
            self.assertEqual(completed.loc["b", "route"], "COLLABORATIVE")
            self.assertEqual(completed.loc["b", "post_edge_reason"], "HIGH_FAILURE_PROBABILITY")
            self.assertEqual(simulator.get_metrics()["post_edge_producer_consistency_rate"], 1.0)


def load_hhk_v2_evidence_for_test(policy, task, row):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        input_path = root / "inputs.json"
        result_path = root / "results.json"
        input_path.write_text(json.dumps([_input(row["task_id"])]), encoding="utf-8")
        result_path.write_text(json.dumps({"results": [row]}), encoding="utf-8")
        tasks, _, report = load_hhk_v2_replay(input_path, result_path)
        evidence = tasks[0].edge_result_evidence
        result = policy.evaluate(task=task, edge_result=evidence, now_s=0.1, cloud_review_estimated_s=0.2)
        return result, tasks, report


if __name__ == "__main__":
    unittest.main()
