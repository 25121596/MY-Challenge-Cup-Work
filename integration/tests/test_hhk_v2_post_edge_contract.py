from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from edge_result_adapter import normalize_edge_result
from policy_bundle_loader import PostEdgeDecision, PostEdgePolicy


V2_RULES = {
    "schema_version": "post-edge-rules/2.0",
    "rule_version": "post-edge-ai4i-test-v2",
    "default_failure_probability_threshold": 0.027514,
    "honor_producer_threshold": True,
    "threshold_review_margin": 0.0,
    "require_calibration_version": True,
    "require_feature_schema_id": True,
    "always_review_model_routes": ["qwen_text_fallback_v1"],
    "legacy_confidence_fallback_threshold": 0.78,
    "task_confidence_thresholds": {},
    "force_review_on_parse_failure": True,
    "force_review_on_ood": True,
    "force_review_levels": ["critical"],
    "force_review_categories": ["unknown"],
    "require_deadline_feasibility": True,
    "cloud_review_deadline_margin_s": 0.0,
    "deadline_infeasible_action": "ACCEPT_EDGE",
}


class HHKV2PostEdgeContractTest(unittest.TestCase):
    def setUp(self):
        self.policy = PostEdgePolicy(V2_RULES, source_path=Path("rules.json"))

    @staticmethod
    def signals(probability: float) -> dict:
        return {
            "failure_probability": probability,
            "threshold_used": 0.027514,
            "threshold_review_margin": 0.0,
            "model_route": "ai4i_tabular_moe_v1",
            "calibration_version": "platt-test",
            "feature_schema_id": "ai4i_8d_v1",
            "review_state": "required" if probability > 0.027514 else "not_required",
            "output_valid": True,
            "source": "hhk_v2_edge_result",
        }

    def evaluate(self, signals: dict, *, deadline_s: float = 100.0):
        task = SimpleNamespace(task_type="fault_diagnosis", absolute_deadline_s=deadline_s)
        return self.policy.evaluate(
            task=task,
            edge_result=signals,
            now_s=1.0,
            cloud_review_estimated_s=2.0,
        )

    def test_calibrated_probability_drives_v2_review(self):
        accepted = self.evaluate(self.signals(0.01))
        reviewed = self.evaluate(self.signals(0.05))
        self.assertEqual(accepted.decision, PostEdgeDecision.ACCEPT_EDGE)
        self.assertEqual(reviewed.decision, PostEdgeDecision.CLOUD_REVIEW)
        self.assertEqual(reviewed.raw_failure_probability, 0.05)
        self.assertEqual(reviewed.failure_probability_threshold, 0.027514)
        self.assertEqual(reviewed.evidence_source, "hhk_v2_edge_result")

    def test_missing_calibration_is_reviewed(self):
        signals = self.signals(0.01)
        signals.pop("calibration_version")
        result = self.evaluate(signals)
        self.assertEqual(result.decision, PostEdgeDecision.CLOUD_REVIEW)
        self.assertEqual(result.reason, "EDGE_CALIBRATION_UNAVAILABLE")

    def test_deadline_infeasible_review_falls_back_to_edge(self):
        result = self.evaluate(self.signals(0.05), deadline_s=2.0)
        self.assertEqual(result.decision, PostEdgeDecision.ACCEPT_EDGE)
        self.assertTrue(result.review_recommended)
        self.assertFalse(result.cloud_review_feasible)
        self.assertIn("DEADLINE_INFEASIBLE", result.reason)

    def test_adapter_preserves_hhk_v2_fields_from_json_answer(self):
        payload = self.signals(0.05)
        payload.update(
            {
                "local_prediction": "fault",
                "safety_priority": "P1",
                "review_reason": ["high_failure_probability"],
            }
        )
        normalized = normalize_edge_result({"answer": json.dumps(payload)})
        self.assertEqual(normalized["failure_probability"], 0.05)
        self.assertEqual(normalized["model_route"], "ai4i_tabular_moe_v1")
        self.assertEqual(normalized["calibration_version"], "platt-test")
        self.assertEqual(normalized["feature_schema_id"], "ai4i_8d_v1")
        self.assertEqual(normalized["review_state"], "required")
        self.assertEqual(normalized["local_prediction"], "fault")

    def test_adapter_does_not_invent_v2_probability_for_legacy_output(self):
        normalized = normalize_edge_result(
            {"answer": json.dumps({"confidence": 0.9, "category": "known"})}
        )
        self.assertNotIn("failure_probability", normalized)
        self.assertNotIn("model_route", normalized)


if __name__ == "__main__":
    unittest.main()
