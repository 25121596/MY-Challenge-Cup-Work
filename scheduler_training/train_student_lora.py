"""参数化学生LoRA训练入口；不写死边端模型。"""
from __future__ import annotations

import argparse
import json

from distillation.student_training import StudentTrainingConfig, train_student_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase4参数化学生LoRA训练")
    parser.add_argument("--model-path", required=True, help="学生HF模型路径或仓库名")
    parser.add_argument(
        "--train-file", default="data/distillation/distillation_train.jsonl"
    )
    parser.add_argument("--output-dir", default="models/student_candidates/edge_lora_v1")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="",
        help="逗号分隔；留空则自动识别",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    config = StudentTrainingConfig(
        model_path=args.model_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=tuple(
            item.strip() for item in args.target_modules.split(",") if item.strip()
        ),
        load_in_4bit=args.load_in_4bit,
    )
    result = train_student_lora(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
