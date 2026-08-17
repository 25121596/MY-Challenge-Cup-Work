"""不加载模型，直接对 prediction + ground_truth JSONL 计算准确率指标。"""
from __future__ import annotations

import argparse
import json

from distillation.accuracy_metrics import evaluate_prediction_rows
from distillation.io_utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="计算Phase4真实标签准确率指标")
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--ground-truth-field", default="ground_truth")
    parser.add_argument("--output", default="results/distillation/accuracy_report.json")
    args = parser.parse_args()
    source = read_jsonl(args.input)
    rows = [
        {
            "prediction": row.get(args.prediction_field),
            "ground_truth": row.get(args.ground_truth_field),
        }
        for row in source
    ]
    report = evaluate_prediction_rows(rows).to_dict()
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
