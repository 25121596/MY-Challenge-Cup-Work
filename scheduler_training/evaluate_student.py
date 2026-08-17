"""原始学生与LoRA学生对照评估入口。"""
from __future__ import annotations

import argparse
import json

from distillation.student_evaluation import compare_base_and_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="比较base学生和LoRA学生")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--test-file", default="data/distillation/distillation_test.jsonl")
    parser.add_argument("--output", default="results/distillation/student_predictions.jsonl")
    parser.add_argument("--report", default="results/distillation/student_accuracy_report.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    report = compare_base_and_lora(
        args.model_path,
        args.lora_path,
        args.test_file,
        args.output,
        args.report,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
