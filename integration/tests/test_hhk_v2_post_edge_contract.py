from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from edge_result_adapter import normalize_edge_result
from policy_bundle_loader import PostEdgeDecision, PostEdgePolicy
from tests.bundle_fixture import TEST_RULES


class HHKV2PostEdgeContractTest(unittest.TestCase):
    """Regression coverage for legacy HHK rows under the current V3 contract."""

    def setUp(self):
        self.policy = PostEdgePolicy(TEST_RULES, source_path=Path("rules.json"))

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

    def test_review_state_drives_review_without_reinterpreting_probability(self):
        accepted = self.evaluate(self.signals(0.01))
        reviewed = self.evaluate(self.signals(0.05))
        self.assertEqual(accepted.decision, PostEdgeDecision.ACCEPT_EDGE)
        self.assertEqual(reviewed.decision, PostEdgeDecision.CLOUD_REVIEW)
        self.assertEqual(reviewed.reason, "EDGE_REVIEW_REQUIRED")
        self.assertEqual(reviewed.failure_probability, 0.05)
        self.assertEqual(reviewed.edge_review_state, "required")

    def test_missing_review_state_uses_legacy_confidence_fallback(self):
        signals = self.signals(0.01)
        signals.pop("review_state")
        result = self.evaluate(signals)
        self.assertEqual(result.decision, PostEdgeDecision.CLOUD_REVIEW)
        self.assertEqual(result.reason, "EDGE_CONFIDENCE_UNAVAILABLE")

    def test_deadline_infeasible_review_falls_back_to_edge(self):
        result = self.evaluate(self.signals(0.05), deadline_s=2.0)
        self.assertEqual(result.decision, PostEdgeDecision.ACCEPT_EDGE)
        self.assertTrue(result.review_recommended)
        self.assertFalse(result.cloud_review_feasible)
        self.assertIn("DEADLINE_INFEASIBLE", result.reason)

    def test_adapter_preserves_hhk_review_fields_from_json_answer(self):
        payload = self.signals(0.05)
        payload.update(
            {
                "local_prediction": "fault",
                "safety_priority": "P1",
                "action": "continue_with_review",
                "review_reason": ["high_failure_probability"],
            }
        )
        normalized = normalize_edge_result({"answer": json.dumps(payload)})
        self.assertEqual(normalized["failure_probability"], 0.05)
        self.assertEqual(normalized["model_route"], "ai4i_tabular_moe_v1")
        self.assertEqual(normalized["review_state"], "required")
        self.assertEqual(normalized["local_prediction"], "fault")
        self.assertEqual(normalized["safety_priority"], "P1")
        self.assertEqual(normalized["edge_action"], "continue_with_review")

    def test_adapter_does_not_invent_probability_for_legacy_output(self):
        normalized = normalize_edge_result(
            {"answer": json.dumps({"confidence": 0.9, "category": "known"})}
        )
        self.assertIsNone(normalized["failure_probability"])
        self.assertIsNone(normalized["model_route"])


if __name__ == "__main__":
    unittest.main()
