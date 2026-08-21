"""Validate and summarize an HHK V2 input/result replay pair."""

from __future__ import annotations

import argparse
import json

from hhk_v2_replay import load_hhk_v2_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="检查HHK V2 AI4I输入/结果联结契约")
    parser.add_argument("--edge-input-path", required=True)
    parser.add_argument("--edge-result-path", required=True)
    parser.add_argument("--edge-input-interval-s", type=float, default=0.05)
    parser.add_argument("--require-full-coverage", action="store_true")
    args = parser.parse_args()
    _, _, report = load_hhk_v2_replay(
        args.edge_input_path,
        args.edge_result_path,
        replay_interval_s=args.edge_input_interval_s,
        require_full_coverage=args.require_full_coverage,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if report.join_coverage < 1.0:
        print(
            "[WARN] 当前仅回放具有HHK V2边端结果的匹配任务；"
            "这不是全量AI4I输入覆盖。"
        )


if __name__ == "__main__":
    main()
