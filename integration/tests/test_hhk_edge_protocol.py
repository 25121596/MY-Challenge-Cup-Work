from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from edge_result_adapter import normalize_edge_result
from policy_bundle_loader import PostEdgeDecision, PostEdgePolicy
from tests.bundle_fixture import TEST_RULES


class HhkEdgeProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PostEdgePolicy(TEST_RULES, source_path=Path(__file__))
        self.task = SimpleNamespace(task_type="fault_diagnosis", absolute_deadline_s=100.0)

    def evaluate(self, result: dict) -> dict:
        signals = normalize_edge_result({"output": result})
        decision = self.policy.evaluate(
            task=self.task,
            edge_result=signals,
            now_s=1.0,
            cloud_review_estimated_s=0.1,
        )
        return {"signals": signals, "decision": decision}

    def test_calibrated_ai4i_normal_accepts_without_confidence(self) -> None:
        row = self.evaluate({
            "model_route": "ai4i_tabular_moe_v1",
            "failure_probability": 0.021,
            "local_prediction": "normal",
            "review_state": "not_required",
            "safety_priority": "P2",
            "action": "continue",
            "decision": {"level": "info"},
        })
        self.assertIsNone(row["signals"]["confidence"])
        self.assertEqual(row["signals"]["failure_probability"], 0.021)
        self.assertEqual(row["decision"].decision, PostEdgeDecision.ACCEPT_EDGE)
        self.assertEqual(row["decision"].reason, "EDGE_RESULT_ACCEPTED")

    def test_edge_required_review_routes_to_cloud_without_confidence(self) -> None:
        row = self.evaluate({
            "model_route": "qwen_text_fallback_v1",
            "failure_probability": None,
            "local_prediction": None,
            "review_state": "required",
            "safety_priority": "P1",
            "action": "continue_with_review",
            "decision": {"level": "warning"},
        })
        self.assertIsNone(row["signals"]["confidence"])
        self.assertEqual(row["decision"].decision, PostEdgeDecision.CLOUD_REVIEW)
        self.assertEqual(row["decision"].reason, "EDGE_REVIEW_REQUIRED")

    def test_legacy_confidence_result_still_uses_legacy_threshold(self) -> None:
        row = self.evaluate({"confidence": 0.4, "decision": {"level": "warning"}})
        self.assertEqual(row["decision"].decision, PostEdgeDecision.CLOUD_REVIEW)
        self.assertEqual(row["decision"].reason, "LOW_EDGE_CONFIDENCE")


if __name__ == "__main__":
    unittest.main()
