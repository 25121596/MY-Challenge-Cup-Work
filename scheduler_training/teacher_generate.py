"""Phase4.5教师结果生成入口。

保留原“蒸馏.zip”的脚本名称，但改为可插拔模拟教师/llama-server教师，
并与hhk标准input直接兼容。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from distillation.io_utils import read_json
from distillation.schema import TeacherConfig
from distillation.teacher_backends import build_teacher_backend
from distillation.teacher_pipeline import generate_teacher_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="生成Phase4教师伪标签JSONL")
    parser.add_argument("--input", required=True, help="hhk标准input JSON")
    parser.add_argument(
        "--output",
        default="data/distillation/teacher_outputs.jsonl",
        help="教师输出JSONL",
    )
    parser.add_argument(
        "--config",
        default="config/distillation.example.json",
        help="Phase4配置JSON",
    )
    parser.add_argument(
        "--teacher-backend",
        choices=["api", "simulated"],
        default=None,
        help="覆盖配置中的教师后端",
    )
    parser.add_argument("--api-base", default=None, help="覆盖教师API base URL")
    parser.add_argument("--model", default=None, help="覆盖教师模型别名")
    parser.add_argument("--ground-truth", default=None, help="可选真实标签JSONL")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--health-check-only", action="store_true", help="仅检查教师服务"
    )
    args = parser.parse_args()

    config_data = read_json(args.config) if Path(args.config).is_file() else {}
    config = TeacherConfig.from_dict(config_data)
    overrides = asdict(config)
    if args.teacher_backend:
        overrides["backend"] = args.teacher_backend
    if args.api_base:
        overrides["api_base"] = args.api_base
    if args.model:
        overrides["model"] = args.model
    config = TeacherConfig(**overrides)

    if args.health_check_only:
        result = build_teacher_backend(config).health()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.get("ok") else 1)

    report = generate_teacher_outputs(
        args.input,
        args.output,
        config,
        ground_truth_path=args.ground_truth,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
