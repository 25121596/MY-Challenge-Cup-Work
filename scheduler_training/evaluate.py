"""Phase4.5 pre-route strategy evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scheduler import (
    AlwaysCloudScheduler,
    AlwaysLocalFirstScheduler,
    PPOScheduler,
    RuleScheduler,
)
from simulator import CloudEdgeSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase4.5二动作Pre-route策略评估")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--queue-policies", nargs="+", choices=["edf", "fifo"], default=["edf"])
    parser.add_argument("--ppo-model", default="models/releases/pre_route_v2/pre_route_ppo.zip")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--arrival-probability", type=float, default=0.10)
    parser.add_argument("--slot-duration", type=float, default=0.05)
    parser.add_argument("--network-mode", choices=["markov", "legacy"], default="markov")
    parser.add_argument(
        "--resource-mode",
        choices=["consume_recover", "legacy"],
        default="consume_recover",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["deadline_aware", "legacy", "lyapunov"],
        default="deadline_aware",
    )
    parser.add_argument(
        "--task-source",
        choices=["typed_simulation", "edge_input", "legacy_simulation"],
        default="typed_simulation",
    )
    parser.add_argument("--edge-input-path", default=None)
    parser.add_argument("--edge-input-non-strict", action="store_true")
    parser.add_argument("--edge-input-interval-s", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="results/pre_route_strategy_comparison.csv",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("steps必须大于0")
    if args.task_source == "edge_input":
        if not args.edge_input_path:
            raise ValueError("edge_input模式必须提供--edge-input-path")
        if not Path(args.edge_input_path).is_file():
            raise FileNotFoundError(args.edge_input_path)
    if not args.skip_ppo:
        model_path = Path(args.ppo_model)
        if model_path.suffix == "":
            model_path = model_path.with_suffix(".zip")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"未找到二动作Pre-route PPO模型：{model_path}；可先使用--skip-ppo。"
            )


def strategy_factories(args: argparse.Namespace) -> dict[str, object]:
    factories: dict[str, object] = {
        "固定本地优先": lambda: AlwaysLocalFirstScheduler(),
        "固定直接云端": lambda: AlwaysCloudScheduler(),
        "Pre-route规则调度器": lambda: RuleScheduler(),
    }
    if not args.skip_ppo:
        factories["二动作Pre-route PPO"] = lambda: PPOScheduler(args.ppo_model)
    return factories


def main() -> None:
    args = parse_args()
    validate_args(args)
    summary_rows: list[dict] = []
    type_rows: list[pd.DataFrame] = []
    collab_rows: list[pd.DataFrame] = []

    for queue_policy in args.queue_policies:
        for strategy_name, factory in strategy_factories(args).items():
            for seed in args.seeds:
                simulator = CloudEdgeSimulator(
                    seed=seed,
                    arrival_probability=args.arrival_probability,
                    slot_duration=args.slot_duration,
                    scheduler=factory(),
                    queue_policy=queue_policy,
                    network_mode=args.network_mode,
                    resource_mode=args.resource_mode,
                    reward_mode=args.reward_mode,
                    task_source=args.task_source,
                    edge_input_path=args.edge_input_path,
                    edge_input_strict=not args.edge_input_non_strict,
                    edge_input_interval_s=args.edge_input_interval_s,
                )
                if args.task_source == "edge_input":
                    simulator.run_until_complete(max_steps=args.steps)
                else:
                    simulator.run(args.steps)

                summary_rows.append(
                    {
                        "queue_policy": queue_policy,
                        "strategy": strategy_name,
                        "seed": seed,
                        "task_source": args.task_source,
                        **simulator.get_metrics(),
                    }
                )
                by_type = simulator.get_task_type_metrics()
                if not by_type.empty:
                    by_type.insert(0, "seed", seed)
                    by_type.insert(0, "strategy", strategy_name)
                    by_type.insert(0, "queue_policy", queue_policy)
                    type_rows.append(by_type)

                completed = simulator.get_completed_tasks()
                if not completed.empty and "route" in completed:
                    collaborative = completed[
                        completed["route"] == "COLLABORATIVE"
                    ].copy()
                    if not collaborative.empty:
                        collaborative["seed"] = seed
                        collaborative["strategy"] = strategy_name
                        collaborative["queue_policy"] = queue_policy
                        collab_rows.append(collaborative)

    detailed_df = pd.DataFrame(summary_rows)
    id_columns = {"queue_policy", "strategy", "seed", "task_source"}
    numeric_columns = [c for c in detailed_df.columns if c not in id_columns]
    aggregate_df = (
        detailed_df.groupby(["queue_policy", "strategy", "task_source"])[numeric_columns]
        .mean()
        .reset_index()
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detailed_path = output.with_name(f"{output.stem}_detailed{output.suffix}")
    type_path = output.with_name(f"{output.stem}_by_task_type{output.suffix}")
    collab_path = output.with_name(
        f"{output.stem}_collaborative_task_records{output.suffix}"
    )
    aggregate_df.to_csv(output, index=False, encoding="utf-8-sig")
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    if type_rows:
        pd.concat(type_rows, ignore_index=True).to_csv(
            type_path, index=False, encoding="utf-8-sig"
        )
    if collab_rows:
        pd.concat(collab_rows, ignore_index=True).to_csv(
            collab_path, index=False, encoding="utf-8-sig"
        )

    display_columns = [
        "queue_policy",
        "strategy",
        "generated_tasks",
        "completion_rate",
        "timeout_rate",
        "average_success_latency",
        "latency_p95_s",
        "average_expected_accuracy",
        "edge_ratio",
        "cloud_ratio",
        "collaborative_ratio",
        "local_first_ratio",
        "direct_cloud_ratio",
        "action_ratio_sum",
        "upload_mb",
        "average_reward",
    ]
    print(
        aggregate_df[[c for c in display_columns if c in aggregate_df]]
        .round(4)
        .to_string(index=False)
    )
    print(f"\n汇总：{output}")
    print(f"逐种子：{detailed_path}")
    if type_rows:
        print(f"按任务类型：{type_path}")
    if collab_rows:
        print(f"协同任务明细：{collab_path}")


if __name__ == "__main__":
    main()
