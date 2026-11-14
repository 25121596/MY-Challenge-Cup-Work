"""Scheduler bundle loading smoke check.

Simulation is owned by ``scheduler_training``. Integration no longer embeds a
second active 25D/Discrete(3) simulator; this validator checks that the
published 24D/Discrete(2) PPO can be loaded by the real runtime.
"""

from __future__ import annotations

import argparse

import numpy as np

from config import DEFAULT_POLICY_BUNDLE
from real_ppo_runtime import RealPPOPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证Scheduler 24D/Discrete(2) policy bundle可被Integration加载"
    )
    parser.add_argument("--policy-bundle", default=str(DEFAULT_POLICY_BUNDLE))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = RealPPOPolicy(args.policy_bundle, deterministic=True)
    action, info = policy.predict_with_info(np.zeros(24, dtype=np.float32))
    print(f"bundle={policy.bundle.manifest_path}")
    print(f"policy_version={policy.bundle.policy_version}")
    print(f"post_edge_rule_version={policy.bundle.rule_version}")
    print(f"24D zero-observation smoke action={action} info={info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
