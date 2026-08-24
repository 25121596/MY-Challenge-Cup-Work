#!/usr/bin/env python3
"""Validate and summarize Phase-A routing observations without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge-io-protocol"))
from feedback_loop import validate_observation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    states: Counter[str] = Counter()
    policy_versions: Counter[str] = Counter()
    total = 0
    routed = 0
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            validate_observation(record)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"invalid observation at line {line_number}: {exc}") from exc
        total += 1
        routing = record.get("routing")
        if routing is not None:
            routed += 1
            states[routing["population_state"]] += 1
            policy_versions[routing["policy_version"]] += 1

    result = {
        "routing_manifest_schema_version": "edge-cloud-routing-manifest/v1",
        "source": str(args.input),
        "source_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "records": total,
        "records_with_routing": routed,
        "population_state_counts": dict(sorted(states.items())),
        "policy_version_counts": dict(sorted(policy_versions.items())),
        "warning": (
            "This is an instrumentation manifest. It does not establish cloud quality, "
            "confirmed incremental value, a trained router, or a production threshold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
