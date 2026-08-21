from __future__ import annotations

import unittest

from ai4i_cloud_review import (
    BINARY_GRAMMAR,
    build_request_body,
    evaluate_results,
    parse_binary_cloud_output,
    select_edge_rows,
    validate_sources,
)


CUTOFF = 0.25


def edge_row(task_id: str, label: int, probability: float) -> dict:
    return {
        "task_id": task_id,
        "label": label,
        "failure_probability": probability,
        "cloud_review": probability > CUTOFF,
    }


def prompt_row(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "feature_schema_id": "ai4i_8d_v1",
        "qwen_input_encoding": "ai4i_qwen_8d_v1",
        "prompt": f"AI4I_QWEN_8D_V1. Output one digit only.\n{task_id}",
    }


def cloud_result(task_id: str, prediction: str | None, latency_ms: float = 10.0) -> dict:
    raw = {"normal": "0", "fault": "1"}.get(prediction)
    return {
        "schema_version": "ai4i-cloud-review-result/v1",
        "task_id": task_id,
        "timestamp": "2026-08-21T00:00:00Z",
        "request": {
            "model_id": "Qwen3-8B-cloud",
            "prompt_encoding": "ai4i_qwen_8d_v1",
        },
        "raw_output": raw,
        "cloud_prediction": prediction,
        "output_valid": prediction is not None,
        "parser_error": None if prediction is not None else "bad output",
        "latency_ms": latency_ms,
        "error": None,
    }


class AI4ICloudReviewTest(unittest.TestCase):
    def test_binary_parser_is_strict(self):
        self.assertEqual(parse_binary_cloud_output("0"), ("normal", None))
        self.assertEqual(parse_binary_cloud_output(" 1\n"), ("fault", None))
        self.assertIsNone(parse_binary_cloud_output("answer: 1")[0])
        self.assertIsNone(parse_binary_cloud_output(1)[0])

    def test_request_is_deterministic_and_disables_thinking(self):
        body = build_request_body("cloud", "prompt", seed=42)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["max_tokens"], 1)
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["grammar"], BINARY_GRAMMAR)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(body["stream"])

    def test_sources_reject_label_leakage_and_route_mismatch(self):
        rows = [edge_row("t1", 0, 0.1)]
        prompts = [prompt_row("t1")]
        indexed = validate_sources(rows, prompts, CUTOFF)
        self.assertIn("t1", indexed)

        leaked = [dict(prompt_row("t1"), label=0)]
        with self.assertRaisesRegex(ValueError, "泄漏字段"):
            validate_sources(rows, leaked, CUTOFF)

        inconsistent = [dict(rows[0], cloud_review=True)]
        with self.assertRaisesRegex(ValueError, "cloud_review"):
            validate_sources(inconsistent, prompts, CUTOFF)

    def test_stratified_pilot_selects_both_labels(self):
        rows = [
            edge_row("n1", 0, 0.3),
            edge_row("n2", 0, 0.4),
            edge_row("p1", 1, 0.5),
            edge_row("p2", 1, 0.6),
        ]
        selected = select_edge_rows(rows, review_only=True, pilot_per_label=1)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["label"] for row in selected}, {0, 1})

    def test_evaluation_separates_disagreement_correction_and_regression(self):
        rows = [
            edge_row("accepted", 0, 0.1),
            edge_row("corrected-fp-1", 0, 0.3),
            edge_row("corrected-fp-2", 0, 0.4),
            edge_row("regressed", 1, 0.5),
            edge_row("agreed-tp", 1, 0.6),
        ]
        targets = [row for row in rows if row["cloud_review"]]
        results = {
            "corrected-fp-1": cloud_result("corrected-fp-1", "normal", 10.0),
            "corrected-fp-2": cloud_result("corrected-fp-2", "normal", 20.0),
            "regressed": cloud_result("regressed", "normal", 30.0),
            "agreed-tp": cloud_result("agreed-tp", "fault", 40.0),
        }
        metrics, feedback = evaluate_results(
            rows,
            targets,
            results,
            cutoff=CUTOFF,
            edge_model_version="edge-test",
            batch_scope="post_edge_review",
            source_hashes={},
        )

        comparison = metrics["edge_cloud_comparison"]
        self.assertEqual(comparison["disagreement_count"], 3)
        self.assertEqual(comparison["review_useful_count"], 2)
        self.assertEqual(comparison["cloud_regression_count"], 1)
        self.assertEqual(comparison["net_corrections"], 1)
        self.assertTrue(metrics["final_system_available"])
        self.assertAlmostEqual(metrics["edge_only_baseline"]["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["edge_model_classifier_reference"]["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["final_system"]["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["cloud_execution"]["latency_ms"]["p50"], 25.0)
        self.assertEqual(len(feedback), 4)
        self.assertEqual(feedback[0]["feedback_schema_version"], "edge-cloud-feedback/v1")
        self.assertEqual(feedback[0]["label_source"], "AI4I held-out test label")

    def test_partial_pilot_does_not_claim_final_system_metrics(self):
        rows = [
            edge_row("a", 0, 0.3),
            edge_row("b", 1, 0.4),
        ]
        metrics, _ = evaluate_results(
            rows,
            [rows[0]],
            {"a": cloud_result("a", "normal")},
            cutoff=CUTOFF,
            edge_model_version="edge-test",
            batch_scope="pilot",
            source_hashes={},
        )
        self.assertFalse(metrics["final_system_available"])
        self.assertIsNone(metrics["final_system"])


if __name__ == "__main__":
    unittest.main()
