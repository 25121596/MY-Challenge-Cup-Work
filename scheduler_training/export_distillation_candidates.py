"""根据Phase4.5最终调度路径导出教师标注候选任务。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from distillation.candidate_export import export_distillation_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="导出Phase4.5蒸馏候选任务")
    parser.add_argument("--input", default="data/sample_edge_inputs.json")
    parser.add_argument("--output", default="data/distillation/distillation_candidates.json")
    parser.add_argument(
        "--strategy",
        choices=[
            "rule", "fixed_cloud", "fixed_local_first", "ppo",
        ],
        default="rule",
    )
    parser.add_argument("--ppo-model", default=None)
    parser.add_argument(
        "--routes", default="CLOUD,COLLABORATIVE",
        help="逗号分隔的候选路由",
    )
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    routes = tuple(item.strip().upper() for item in args.routes.split(",") if item.strip())
    report = export_distillation_candidates(
        args.input,
        args.output,
        strategy=args.strategy,
        ppo_model=args.ppo_model,
        routes=routes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
