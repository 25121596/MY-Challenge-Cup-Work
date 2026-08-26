"""Evaluate pre-route and Post-edge ablations across versioned scenarios."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from experiment_scenarios import (
    AlwaysAcceptEdgePolicy,
    DEFAULT_SCENARIO_FILES,
    ExperimentScenario,
    load_experiment_scenarios,
    scenario_simulator_kwargs,
)
from post_edge_policy import PostEdgePolicy
from scheduler import AlwaysCloudScheduler, AlwaysLocalFirstScheduler, PPOScheduler, RuleScheduler
from simulator import CloudEdgeSimulator


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    display_name: str
    scheduler_factory: Callable[[], object]
    post_edge_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="24D/两动作多场景调度与消融评估")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--queue-policies", nargs="+", choices=["edf", "fifo"], default=["edf"])
    parser.add_argument("--ppo-model", default="models/releases/pre_route_v2/pre_route_ppo.zip")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--arrival-probability", type=float, default=0.10)
    parser.add_argument("--slot-duration", type=float, default=0.05)
    parser.add_argument("--network-mode", choices=["markov", "legacy"], default="markov")
    parser.add_argument("--resource-mode", choices=["consume_recover", "legacy"], default="consume_recover")
    parser.add_argument("--reward-mode", choices=["deadline_aware", "legacy", "lyapunov"], default="deadline_aware")
    parser.add_argument(
        "--task-source",
        choices=["typed_simulation", "edge_input", "hhk_v2_replay", "legacy_simulation"],
        default="typed_simulation",
    )
    parser.add_argument("--edge-input-path", default=None)
    parser.add_argument("--edge-result-path", default=None)
    parser.add_argument("--post-edge-rules", default=None)
    parser.add_argument("--edge-input-non-strict", action="store_true")
    parser.add_argument("--edge-input-interval-s", type=float, default=0.05)
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--standard-scenarios",
        action="store_true",
        help="评估normal_load/edge_burst/weak_network标准场景",
    )
    scenario_group.add_argument(
        "--scenario-files",
        nargs="+",
        help="显式指定一个或多个scheduler-experiment-scenario/1.0 JSON",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _scenario_paths(args: argparse.Namespace) -> tuple[str | Path, ...] | None:
    if args.standard_scenarios:
        return DEFAULT_SCENARIO_FILES
    if args.scenario_files:
        return tuple(args.scenario_files)
    return None


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("steps必须大于0")
    scenario_paths = _scenario_paths(args)
    if scenario_paths is not None and args.task_source != "typed_simulation":
        raise ValueError("场景矩阵评估只支持typed_simulation任务源")
    if args.task_source in {"edge_input", "hhk_v2_replay"}:
        if not args.edge_input_path:
            raise ValueError("edge_input模式必须提供--edge-input-path")
        if not Path(args.edge_input_path).is_file():
            raise FileNotFoundError(args.edge_input_path)
    if args.task_source == "hhk_v2_replay":
        if not args.edge_result_path:
            raise ValueError("hhk_v2_replay模式必须提供--edge-result-path")
        if not Path(args.edge_result_path).is_file():
            raise FileNotFoundError(args.edge_result_path)
    if scenario_paths is not None:
        load_experiment_scenarios(scenario_paths)
    if not args.skip_ppo:
        model_path = Path(args.ppo_model)
        if model_path.suffix == "":
            model_path = model_path.with_suffix(".zip")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"未找到二动作Pre-route PPO模型：{model_path}；可先使用--skip-ppo。"
            )


def _experiment_strategies(args: argparse.Namespace) -> tuple[StrategySpec, ...]:
    rows: list[StrategySpec] = [
        StrategySpec("EDGE_ONLY", "Edge-only基准", lambda: AlwaysLocalFirstScheduler(), "always_accept_edge"),
    ]
    if not args.skip_ppo:
        rows.append(StrategySpec(
            "PRE_ROUTE_ONLY", "仅Pre-route PPO", lambda: PPOScheduler(args.ppo_model), "always_accept_edge"
        ))
    rows.append(StrategySpec(
        "POST_EDGE_ONLY", "仅Post-edge规则", lambda: AlwaysLocalFirstScheduler(), "configured_rules"
    ))
    if not args.skip_ppo:
        rows.append(StrategySpec(
            "TWO_LAYER", "完整两层调度", lambda: PPOScheduler(args.ppo_model), "configured_rules"
        ))
    return tuple(rows)


def _legacy_strategies(args: argparse.Namespace) -> tuple[StrategySpec, ...]:
    rows: list[StrategySpec] = [
        StrategySpec("FIXED_LOCAL_FIRST", "固定本地优先", lambda: AlwaysLocalFirstScheduler(), "configured_rules"),
        StrategySpec("FIXED_CLOUD", "固定直接云端", lambda: AlwaysCloudScheduler(), "configured_rules"),
        StrategySpec("RULE", "Pre-route规则调度器", lambda: RuleScheduler(), "configured_rules"),
    ]
    if not args.skip_ppo:
        rows.append(StrategySpec("PPO", "二动作Pre-route PPO", lambda: PPOScheduler(args.ppo_model), "configured_rules"))
    return tuple(rows)


def _policy_for(spec: StrategySpec, args: argparse.Namespace):
    if spec.post_edge_mode == "always_accept_edge":
        return AlwaysAcceptEdgePolicy()
    return PostEdgePolicy.from_json(args.post_edge_rules) if args.post_edge_rules else None


def _source_metadata(scenario: ExperimentScenario | None, task_source: str) -> dict[str, str]:
    if scenario is not None:
        return {
            "scenario_id": scenario.scenario_id,
            "scenario_source": scenario.source_path,
            "metric_scope": "simulated_system",
            "latency_source": "configured_profile_simulation",
            "accuracy_source": "configured_expected_accuracy",
            "resource_source": "configured_scenario_simulation",
        }
    latency_source = (
        "hhk_v2_edge_latency_plus_simulated_cloud_network"
        if task_source == "hhk_v2_replay"
        else "configured_profile_simulation"
    )
    return {
        "scenario_id": "legacy_cli",
        "scenario_source": "command_line",
        "metric_scope": "simulated_system",
        "latency_source": latency_source,
        "accuracy_source": "configured_expected_accuracy",
        "resource_source": "legacy_simulated_dynamics",
    }


def _simulator_kwargs(
    args: argparse.Namespace,
    scenario: ExperimentScenario | None,
    *,
    seed: int,
) -> dict:
    if scenario is not None:
        values = scenario_simulator_kwargs(scenario, seed=seed)
    else:
        values = {
            "arrival_probability": args.arrival_probability,
            "slot_duration": args.slot_duration,
            "task_source": args.task_source,
            "edge_input_path": args.edge_input_path,
            "edge_result_path": args.edge_result_path,
            "edge_input_strict": not args.edge_input_non_strict,
            "edge_input_interval_s": args.edge_input_interval_s,
        }
    values.update({
        "network_mode": args.network_mode,
        "resource_mode": args.resource_mode,
        "reward_mode": args.reward_mode,
    })
    return values


def _aggregate(detailed: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric = [column for column in detailed.select_dtypes(include="number").columns if column != "seed"]
    grouped = detailed.groupby(group_columns, dropna=False)[numeric]
    means = grouped.mean()
    standard_deviations = grouped.std(ddof=0).add_suffix("_std")
    return pd.concat([means, standard_deviations], axis=1).reset_index()


def _edge_only_comparison(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (queue_policy, scenario_id), group in aggregate.groupby(["queue_policy", "scenario_id"], dropna=False):
        baseline_rows = group[group["strategy_id"] == "EDGE_ONLY"]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for _, current in group.iterrows():
            p95 = float(current.get("latency_p95_s", float("nan")))
            baseline_p95 = float(baseline.get("latency_p95_s", float("nan")))
            p95_reduction = (
                (baseline_p95 - p95) / baseline_p95
                if baseline_p95 == baseline_p95 and baseline_p95 > 0.0 and p95 == p95
                else float("nan")
            )
            rows.append({
                "queue_policy": queue_policy,
                "scenario_id": scenario_id,
                "strategy_id": current["strategy_id"],
                "strategy": current["strategy"],
                "completion_rate_delta": float(current["completion_rate"] - baseline["completion_rate"]),
                "timeout_rate_delta": float(current["timeout_rate"] - baseline["timeout_rate"]),
                "latency_p95_delta_s": p95 - baseline_p95,
                "latency_p95_reduction_rate": p95_reduction,
                "upload_mb_delta": float(current["upload_mb"] - baseline["upload_mb"]),
                "expected_accuracy_prior_delta": float(
                    current["average_expected_accuracy"] - baseline["average_expected_accuracy"]
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = _scenario_paths(args)
    scenarios: tuple[ExperimentScenario | None, ...] = (
        tuple(load_experiment_scenarios(paths)) if paths is not None else (None,)
    )
    strategies = _experiment_strategies(args) if paths is not None else _legacy_strategies(args)

    summary_rows: list[dict] = []
    type_rows: list[pd.DataFrame] = []
    collab_rows: list[pd.DataFrame] = []
    for queue_policy in args.queue_policies:
        for scenario in scenarios:
            source_metadata = _source_metadata(scenario, args.task_source)
            for spec in strategies:
                for seed in args.seeds:
                    kwargs = _simulator_kwargs(args, scenario, seed=seed)
                    kwargs.update({
                        "queue_policy": queue_policy,
                        "seed": seed,
                        "scheduler": spec.scheduler_factory(),
                    })
                    policy = _policy_for(spec, args)
                    if policy is not None:
                        kwargs["post_edge_policy"] = policy
                    simulator = CloudEdgeSimulator(**kwargs)
                    if scenario is None and args.task_source in {"edge_input", "hhk_v2_replay"}:
                        simulator.run_until_complete(max_steps=args.steps)
                    else:
                        simulator.run(args.steps)

                    common = {
                        "queue_policy": queue_policy,
                        "strategy_id": spec.strategy_id,
                        "strategy": spec.display_name,
                        "post_edge_mode": spec.post_edge_mode,
                        "seed": seed,
                        "task_source": simulator.task_source,
                        **source_metadata,
                    }
                    summary_rows.append({**common, **simulator.get_metrics()})
                    by_type = simulator.get_task_type_metrics()
                    if not by_type.empty:
                        for key, value in reversed(list(common.items())):
                            by_type.insert(0, key, value)
                        type_rows.append(by_type)

                    completed = simulator.get_completed_tasks()
                    if not completed.empty and "route" in completed:
                        collaborative = completed[completed["route"] == "COLLABORATIVE"].copy()
                        if not collaborative.empty:
                            for key, value in common.items():
                                collaborative[key] = value
                            collab_rows.append(collaborative)

    detailed_df = pd.DataFrame(summary_rows)
    group_columns = [
        "queue_policy", "scenario_id", "scenario_source", "strategy_id", "strategy",
        "post_edge_mode", "task_source", "metric_scope", "latency_source",
        "accuracy_source", "resource_source",
    ]
    aggregate_df = _aggregate(detailed_df, group_columns)
    comparison_df = _edge_only_comparison(aggregate_df)

    default_name = "results/scenario_strategy_comparison.csv" if paths is not None else "results/pre_route_strategy_comparison.csv"
    output = Path(args.output or default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    detailed_path = output.with_name(f"{output.stem}_detailed{output.suffix}")
    type_path = output.with_name(f"{output.stem}_by_task_type{output.suffix}")
    collab_path = output.with_name(f"{output.stem}_collaborative_task_records{output.suffix}")
    comparison_path = output.with_name(f"{output.stem}_vs_edge_only{output.suffix}")
    aggregate_df.to_csv(output, index=False, encoding="utf-8-sig")
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    if type_rows:
        pd.concat(type_rows, ignore_index=True).to_csv(type_path, index=False, encoding="utf-8-sig")
    if collab_rows:
        pd.concat(collab_rows, ignore_index=True).to_csv(collab_path, index=False, encoding="utf-8-sig")

    display_columns = [
        "scenario_id", "strategy_id", "generated_tasks", "completion_rate",
        "timeout_rate", "latency_p95_s", "max_edge_queue", "max_cloud_queue",
        "local_first_ratio", "direct_cloud_ratio", "cloud_review_ratio", "upload_mb",
        "average_expected_accuracy",
    ]
    print(aggregate_df[[c for c in display_columns if c in aggregate_df]].round(4).to_string(index=False))
    print(f"\n汇总均值/标准差：{output}")
    print(f"逐种子：{detailed_path}")
    print(f"相对Edge-only：{comparison_path}")
    if type_rows:
        print(f"按任务类型：{type_path}")
    if collab_rows:
        print(f"协同任务明细：{collab_path}")


if __name__ == "__main__":
    main()
