from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import urllib.request

from config import (
    CLOUD_ENDPOINT,
    CLOUD_METRICS_URL,
    DEFAULT_EDGE_INPUT,
    DEFAULT_POLICY_BUNDLE,
    EDGE_ENDPOINT,
    EDGE_HHK_STATE_URL,
    EDGE_METRICS_URL,
    VERSION,
)
from model_client import ModelClient
from edge_input_loader import load_edge_inputs
from policy_bundle_loader import load_policy_bundle


def parse_args():
    parser = argparse.ArgumentParser(description="检查 Integration 对接环境")
    parser.add_argument("--policy-bundle", default=str(DEFAULT_POLICY_BUNDLE))
    parser.add_argument("--input", default=str(DEFAULT_EDGE_INPUT))
    parser.add_argument("--skip-servers", action="store_true")
    return parser.parse_args()


def mark(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def _check_url(label: str, url: str) -> None:
    if not url:
        print(f"[OPTIONAL] {label}: 未配置")
        return
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            response.read(256)
        print(f"[OK] {label}: {url}")
    except Exception as exc:
        print(f"[FAIL] {label}: {url} | {exc}")


def main():
    args = parse_args()
    policy_bundle = Path(args.policy_bundle).resolve()
    input_file = Path(args.input).resolve()

    print("=== Integration 文件与契约 ===")
    print(f"[INFO] software version: {VERSION}")
    print(f"[{mark(input_file.is_file())}] input:          {input_file}")
    print(f"[{mark(policy_bundle.is_file())}] policy bundle:  {policy_bundle}")

    print("\n=== Scheduler bundle契约 ===")
    try:
        bundle = load_policy_bundle(policy_bundle)
        print(f"[OK] observation dim = {len(bundle.observation_names)}")
        print("[OK] actions = LOCAL_FIRST/DIRECT_CLOUD (0/1 PPO mapping)")
        print(f"[OK] policy version = {bundle.policy_version}")
        print(f"[OK] post-edge rule = {bundle.rule_version}")
    except Exception as exc:
        print(f"[FAIL] policy bundle: {exc}")

    try:
        tasks, report = load_edge_inputs(input_file)
        print(f"[OK] Integration input loader: {len(tasks)} tasks ({report.input_count} inputs)")
    except Exception as exc:
        print(f"[FAIL] Integration input loader: {exc}")

    print("\n=== Python 依赖 ===")
    for module in ("numpy", "psutil", "torch", "gymnasium", "stable_baselines3"):
        ok = importlib.util.find_spec(module) is not None
        print(f"[{mark(ok)}] {module}")

    print("\n=== 可选真实指标源 ===")
    _check_url("EDGE HHK state", EDGE_HHK_STATE_URL)
    _check_url("EDGE metrics", EDGE_METRICS_URL.rstrip("/") + "/metrics" if EDGE_METRICS_URL else "")
    _check_url("CLOUD metrics", CLOUD_METRICS_URL.rstrip("/") + "/metrics" if CLOUD_METRICS_URL else "")

    if not args.skip_servers:
        print("\n=== llama-server ===")
        for name, endpoint in (("EDGE", EDGE_ENDPOINT), ("CLOUD", CLOUD_ENDPOINT)):
            try:
                client = ModelClient(endpoint)
                health = client.health()
                model_id = client.get_model_id()
                print(f"[OK] {name}: {endpoint} | {health} | {model_id}")
            except Exception as exc:
                print(f"[FAIL] {name}: {endpoint} | {exc}")


if __name__ == "__main__":
    main()
