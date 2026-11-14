"""校验教师/学生结构化输出。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from distillation.io_utils import read_jsonl, write_json
from distillation.output_validation import validate_output


def main() -> None:
    parser = argparse.ArgumentParser(description="校验Phase4输出协议")
    parser.add_argument("--input", required=True, help="JSONL文件")
    parser.add_argument(
        "--field", default="teacher_output", help="待校验字段，默认teacher_output"
    )
    parser.add_argument("--report", default="results/distillation/output_validation.json")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    details = []
    for row in rows:
        result = validate_output(row.get(args.field, ""))
        details.append(
            {
                "task_id": row.get("task_id"),
                "valid": result.valid,
                "errors": list(result.errors),
            }
        )
    report = {
        "input": args.input,
        "field": args.field,
        "total": len(rows),
        "valid": sum(1 for item in details if item["valid"]),
        "invalid": sum(1 for item in details if not item["valid"]),
        "details": details,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
