from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from distillation.accuracy_metrics import evaluate_prediction_rows
from distillation.dataset_builder import build_dataset
from distillation.io_utils import read_jsonl
from distillation.output_validation import validate_output
from distillation.schema import TeacherConfig
from distillation.teacher_pipeline import generate_teacher_outputs


class DistillationPipelineTests(unittest.TestCase):
    def test_simulated_teacher_dataset_and_accuracy_contracts(self):
        root = Path(__file__).resolve().parents[1]
        sample_input = root / "data" / "sample_edge_inputs.json"
        self.assertTrue(sample_input.is_file())

        with tempfile.TemporaryDirectory(prefix="distillation_validation_") as temp_dir:
            temp = Path(temp_dir)
            teacher_path = temp / "teacher_outputs.jsonl"
            report = generate_teacher_outputs(
                sample_input,
                teacher_path,
                TeacherConfig(backend="simulated", model="simulated_rule"),
                limit=16,
            )
            self.assertEqual(report.input_count, 16)
            self.assertEqual(report.valid_count, 16)

            teacher_rows = read_jsonl(teacher_path)
            self.assertTrue(
                all(row["label_source"] == "teacher_pseudo_label" for row in teacher_rows)
            )
            self.assertTrue(
                all(isinstance(row.get("student_input"), dict) for row in teacher_rows)
            )

            train_path = temp / "train.jsonl"
            test_path = temp / "test.jsonl"
            manifest_path = temp / "manifest.json"
            dataset_report = build_dataset(
                teacher_path,
                train_path,
                test_path,
                manifest_path,
                target_policy="prefer-ground-truth",
                test_ratio=0.25,
                seed=42,
            )
            self.assertGreater(dataset_report.train_count, 0)
            self.assertGreater(dataset_report.test_count, 0)
            train_ids = {row["task_id"] for row in read_jsonl(train_path)}
            test_ids = {row["task_id"] for row in read_jsonl(test_path)}
            self.assertFalse(train_ids.intersection(test_ids))

            truth = teacher_rows[0]["teacher_output"]
            self.assertTrue(validate_output(truth).valid)
            self.assertFalse(validate_output({"action": "alarm"}).valid)

            accuracy = evaluate_prediction_rows(
                [
                    {"prediction": truth, "ground_truth": truth},
                    {"prediction": truth, "ground_truth": truth},
                ]
            )
            self.assertEqual(accuracy.category_accuracy, 1.0)
            no_truth = evaluate_prediction_rows(
                [{"prediction": truth, "ground_truth": None}]
            )
            self.assertIsNone(no_truth.category_accuracy)

    def test_unknown_student_model_is_not_activated(self):
        root = Path(__file__).resolve().parents[1]
        registry = json.loads(
            (root / "models" / "student_registry.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(registry["active_model"])


if __name__ == "__main__":
    unittest.main()
