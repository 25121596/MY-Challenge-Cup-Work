"""登记学生候选模型；不会自动替换未知边端模型。"""
from __future__ import annotations

import argparse
import json

from distillation.model_registry import register_student_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="登记Phase4学生候选模型")
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--evaluation-report", default=None)
    parser.add_argument("--registry", default="models/student_registry.json")
    args = parser.parse_args()
    item = register_student_candidate(
        args.registry,
        model_name=args.name,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        evaluation_report=args.evaluation_report,
    )
    print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
