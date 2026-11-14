"""Phase4.5蒸馏数据集构建入口。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from distillation.dataset_builder import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="构建Phase4蒸馏训练集和测试集")
    parser.add_argument(
        "--teacher-outputs",
        default="data/distillation/teacher_outputs.jsonl",
    )
    parser.add_argument("--train-output", default="data/distillation/distillation_train.jsonl")
    parser.add_argument("--test-output", default="data/distillation/distillation_test.jsonl")
    parser.add_argument("--manifest", default="data/distillation/dataset_manifest.json")
    parser.add_argument(
        "--target-policy",
        choices=["prefer-ground-truth", "ground-truth-only", "teacher-only"],
        default="prefer-ground-truth",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-teacher-confidence", type=float, default=0.0)
    parser.add_argument("--require-ground-truth-for-test", action="store_true")
    args = parser.parse_args()

    report = build_dataset(
        args.teacher_outputs,
        args.train_output,
        args.test_output,
        args.manifest,
        target_policy=args.target_policy,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_teacher_confidence=args.min_teacher_confidence,
        require_ground_truth_for_test=args.require_ground_truth_for_test,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
