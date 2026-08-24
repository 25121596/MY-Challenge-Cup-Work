from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from config import (
    CLOUD_ENDPOINT,
    CLOUD_METRICS_URL,
    DEFAULT_ASSET_ROUTING_POLICY,
    DEFAULT_EDGE_INPUT_INTERVAL_S,
    DEFAULT_EDGE_INPUT,
    DEFAULT_POLICY_BUNDLE,
    EDGE_ENDPOINT,
    EDGE_HHK_STATE_URL,
    EDGE_METRICS_URL,
    FALLBACK_BANDWIDTH_MBPS,
    METRICS_TIMEOUT_S,
    NETWORK_HISTORY_SIZE,
    NETWORK_PROBE_BYTES,
    PROFILE_SAMPLE_INTERVAL_S,
    REQUEST_TIMEOUT_S,
    ROOT,
    STATE_SAMPLE_TTL_S,
    VERSION,
)
from asset_routing_policy import AssetRoutingPolicy
from model_client import ModelClient
from profiling import RuntimeProfiler, TRACE_SCHEMA_VERSION
from real_ppo_runtime import RealPPOPolicy, RealPPORuntime
from real_state_provider import (
    HHKHeartbeatSource,
    HttpResourceSource,
    LocalResourceSource,
    NetworkMonitor,
    RealStateProvider,
)
from real_task_runtime import RealTaskManager
from service_time_estimator import ServiceTimeEstimator
from task_io import save_json
from trace_collector import TraceCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cloud-edge Integration: Scheduler bundle real runtime + profiling"
    )
    parser.add_argument("--policy-bundle", default=str(DEFAULT_POLICY_BUNDLE))
    parser.add_argument("--input", default=str(DEFAULT_EDGE_INPUT))
    parser.add_argument("--asset-routing-policy", default=str(DEFAULT_ASSET_ROUTING_POLICY))
    parser.add_argument("--limit", type=int, default=8, help="0=全部任务")
    parser.add_argument("--arrival-interval-s", type=float, default=DEFAULT_EDGE_INPUT_INTERVAL_S)
    parser.add_argument("--ignore-source-timestamps", action="store_true")
    parser.add_argument("--drain-timeout-s", type=float, default=float(REQUEST_TIMEOUT_S + 10))
    parser.add_argument("--edge-endpoint", default=EDGE_ENDPOINT)
    parser.add_argument("--cloud-endpoint", default=CLOUD_ENDPOINT)
    parser.add_argument("--edge-hhk-state-url", default=EDGE_HHK_STATE_URL)
    parser.add_argument("--edge-metrics-url", default=EDGE_METRICS_URL)
    parser.add_argument("--cloud-metrics-url", default=CLOUD_METRICS_URL)
    parser.add_argument("--skip-model-health-check", action="store_true")
    parser.add_argument("--profile-sample-interval-s", type=float, default=PROFILE_SAMPLE_INTERVAL_S)
    parser.add_argument("--output-json", default=str(ROOT / "outputs" / "runtime_records.json"))
    parser.add_argument("--output-csv", default=str(ROOT / "outputs" / "runtime_trace.csv"))
    parser.add_argument("--output-jsonl", default=str(ROOT / "outputs" / "runtime_trace.jsonl"))
    parser.add_argument("--summary-json", default=str(ROOT / "outputs" / "runtime_summary.json"))
    return parser.parse_args()


def _resource_source(*, hhk_state_url: str, metrics_url: str, label: str) -> Any:
    if hhk_state_url:
        return HHKHeartbeatSource(hhk_state_url, timeout_s=METRICS_TIMEOUT_S)
    if metrics_url:
        return HttpResourceSource(metrics_url, timeout_s=METRICS_TIMEOUT_S, label=label)
    return LocalResourceSource(label=f"{label}:local")


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print("Cloud-edge Integration real PPO runtime + profiling")
    print(f"software version: {VERSION}")
    print("Scheduler bundle | 24D | Discrete(2) | Post-edge | EDF | trace schema v2")
    print("=" * 80)

    edge_client = ModelClient(args.edge_endpoint, timeout_s=REQUEST_TIMEOUT_S)
    cloud_client = ModelClient(args.cloud_endpoint, timeout_s=REQUEST_TIMEOUT_S)
    if not args.skip_model_health_check:
        try:
            print(f"EDGE  {args.edge_endpoint} -> {edge_client.health()}")
            print(f"CLOUD {args.cloud_endpoint} -> {cloud_client.health()}")
        except Exception as exc:
            print(f"真实模型health check失败: {exc}", file=sys.stderr)
            return 2

    try:
        policy = RealPPOPolicy(args.policy_bundle, deterministic=True)
    except Exception as exc:
        print(f"Scheduler policy bundle加载失败: {exc}", file=sys.stderr)
        return 3

    edge_source = _resource_source(
        hhk_state_url=args.edge_hhk_state_url,
        metrics_url=args.edge_metrics_url,
        label="edge_http",
    )
    cloud_source = _resource_source(
        hhk_state_url="",
        metrics_url=args.cloud_metrics_url,
        label="cloud_http",
    )
    network_monitor = NetworkMonitor(
        args.cloud_endpoint,
        cloud_metrics_base_url=(args.cloud_metrics_url or None),
        fallback_bandwidth_mbps=FALLBACK_BANDWIDTH_MBPS,
        timeout_s=METRICS_TIMEOUT_S,
        probe_bytes=NETWORK_PROBE_BYTES,
        history_size=NETWORK_HISTORY_SIZE,
    )
    state_provider = RealStateProvider(
        edge_source=edge_source,
        cloud_source=cloud_source,
        network_monitor=network_monitor,
        sample_ttl_s=STATE_SAMPLE_TTL_S,
    )
    manager = RealTaskManager(
        edge_client=edge_client,
        cloud_client=cloud_client,
        network_monitor=network_monitor,
        estimator=ServiceTimeEstimator(alpha=0.25),
    )
    profiler = RuntimeProfiler(
        state_provider,
        manager,
        interval_s=args.profile_sample_interval_s,
    )
    manager.set_profiler(profiler)
    runtime = RealPPORuntime(
        policy=policy,
        policy_bundle=policy.bundle,
        state_provider=state_provider,
        task_manager=manager,
        asset_routing_policy=AssetRoutingPolicy.from_json(args.asset_routing_policy),
    )
    collector = TraceCollector()

    try:
        tasks = runtime.load_tasks(args.input, strict=True)
        print(f"input={Path(args.input).resolve()} | loaded={len(tasks)}")
        profiler.start()
        result = runtime.run_tasks(
            tasks,
            limit=args.limit,
            arrival_interval_s=args.arrival_interval_s,
            use_source_timestamps=not args.ignore_source_timestamps,
            drain_timeout_s=args.drain_timeout_s,
            verbose=True,
        )
        payload = {
            "version": VERSION,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "observation_dim": len(runtime.observation_names),
            "observation_names": runtime.observation_names,
            "policy_bundle": str(policy.bundle.manifest_path),
            "policy_version": policy.bundle.policy_version,
            "asset_routing_policy": str(Path(args.asset_routing_policy).resolve()),
            "asset_routing_policy_version": runtime.asset_routing_policy.policy_version,
            "post_edge_rule_version": policy.bundle.rule_version,
            "input_count": result.input_count,
            "submitted_count": result.submitted_count,
            "completed_count": result.completed_count,
            "timeout_count": result.timeout_count,
            "service_time_estimates": result.service_time_estimates,
            "profiler_sample_interval_s": args.profile_sample_interval_s,
            "profiler_timeline": profiler.timeline(),
            "records": result.records,
        }
        save_json(args.output_json, payload)
        collector.write_jsonl(args.output_jsonl, result.records)
        collector.write_csv(args.output_csv, result.records)
        collector.write_summary(
            args.summary_json,
            result.records,
            integration_version=VERSION,
            policy_bundle=str(policy.bundle.manifest_path),
            policy_version=policy.bundle.policy_version,
            post_edge_rule_version=policy.bundle.rule_version,
            observation_dim=len(runtime.observation_names),
        )
        print("\n" + "=" * 80)
        print(
            f"DONE submitted={result.submitted_count} "
            f"completed={result.completed_count} timeout={result.timeout_count}"
        )
        print(f"JSON  -> {Path(args.output_json).resolve()}")
        print(f"JSONL -> {Path(args.output_jsonl).resolve()}")
        print(f"CSV   -> {Path(args.output_csv).resolve()}")
        print(f"SUM   -> {Path(args.summary_json).resolve()}")
        return 0
    except KeyboardInterrupt:
        print("用户中断。")
        return 130
    finally:
        profiler.stop()
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
