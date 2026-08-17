"""Validate a published Phase4.5 policy bundle."""

from __future__ import annotations

import argparse
import json

from policy_bundle import validate_policy_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证Phase4.5 pre-route策略包")
    parser.add_argument("bundle", help="policy_bundle.json路径")
    parser.add_argument(
        "--load-ppo",
        action="store_true",
        help="同时加载Stable-Baselines3模型并检查24D/Discrete(2)空间",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_policy_bundle(args.bundle, load_ppo=args.load_ppo)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
