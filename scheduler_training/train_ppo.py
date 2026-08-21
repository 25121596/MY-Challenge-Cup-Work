"""Train and publish the Phase4.5 24D / Discrete(2) pre-route PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from observation import OBSERVATION_NAMES
from policy_bundle import publish_policy_bundle
from post_edge_policy import DEFAULT_RULES_PATH, PostEdgePolicy
from ppo_env import CloudEdgeGymEnv
from task_profiles import profiles_as_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练Phase4.5工业云边二动作Pre-route PPO")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--arrival-probability", type=float, default=0.10)
    parser.add_argument("--slot-duration", type=float, default=0.05)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", default="models/checkpoints/pre_route_v2/pre_route_ppo")
    parser.add_argument("--release-dir", default="models/releases/pre_route_v2")
    parser.add_argument("--policy-version", default="pre-route-ppo-v2")
    parser.add_argument("--post-edge-rules", default=str(DEFAULT_RULES_PATH))
    parser.add_argument("--queue-policy", choices=["edf", "fifo"], default="edf")
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
        choices=["typed_simulation", "edge_input", "hhk_v2_replay", "legacy_simulation"],
        default="typed_simulation",
    )
    parser.add_argument("--edge-input-path", default=None)
    parser.add_argument("--edge-result-path", default=None)
    parser.add_argument("--edge-input-non-strict", action="store_true")
    parser.add_argument("--edge-input-interval-s", type=float, default=0.05)
    parser.add_argument("--ppo-reward-scale", type=float, default=0.05)
    parser.add_argument("--communication-penalty-weight", type=float, default=0.25)
    parser.add_argument("--edge-compute-penalty-weight", type=float, default=0.05)
    parser.add_argument("--cloud-compute-penalty-weight", type=float, default=0.05)
    parser.add_argument("--eval-episodes", type=int, default=20)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.timesteps <= 0 or args.episode_steps <= 0:
        raise ValueError("timesteps和episode-steps必须大于0")
    if args.slot_duration <= 0 or args.edge_input_interval_s <= 0:
        raise ValueError("时间参数必须大于0")
    if args.eval_episodes <= 0:
        raise ValueError("eval-episodes必须大于0")
    if not args.policy_version.strip():
        raise ValueError("policy-version不能为空")
    PostEdgePolicy.from_json(args.post_edge_rules)
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


def create_env(seed: int, args: argparse.Namespace) -> CloudEdgeGymEnv:
    return CloudEdgeGymEnv(
        seed=seed,
        episode_steps=args.episode_steps,
        arrival_probability=args.arrival_probability,
        slot_duration=args.slot_duration,
        queue_policy=args.queue_policy,
        network_mode=args.network_mode,
        resource_mode=args.resource_mode,
        reward_mode=args.reward_mode,
        task_source=args.task_source,
        edge_input_path=args.edge_input_path,
        edge_result_path=args.edge_result_path,
        edge_input_strict=not args.edge_input_non_strict,
        edge_input_interval_s=args.edge_input_interval_s,
        ppo_reward_scale=args.ppo_reward_scale,
        communication_penalty_weight=args.communication_penalty_weight,
        edge_compute_penalty_weight=args.edge_compute_penalty_weight,
        cloud_compute_penalty_weight=args.cloud_compute_penalty_weight,
        post_edge_rules_path=args.post_edge_rules,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.torch_threads)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    checker = create_env(args.seed, args)
    check_env(checker, warn=True, skip_render_check=True)
    checker.close()

    monitor_path = results_dir / f"{output_path.stem}_monitor.csv"
    training_env = Monitor(create_env(args.seed, args), filename=str(monitor_path))
    model = PPO(
        "MlpPolicy",
        training_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        seed=args.seed,
        device="auto",
    )
    print(
        f"开始Phase4.5 PPO训练：actions=2, queue={args.queue_policy}, "
        f"task_source={args.task_source}, slot={args.slot_duration}s, "
        f"obs={len(OBSERVATION_NAMES)}维"
    )
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=True)
    actual_timesteps = int(model.num_timesteps)
    model.save(str(output_path))

    evaluation_env = Monitor(create_env(args.seed + 10_000, args))
    mean_reward, std_reward = evaluate_policy(
        model,
        evaluation_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    evaluation_env.close()
    training_env.close()

    replay_report = None
    report_env = create_env(args.seed, args)
    _, report_info = report_env.reset()
    replay_report = report_info.get("hhk_v2_replay")
    report_env.close()
    calibration_status = (
        "partial_hhk_v2_ai4i_edge_result_replay"
        if replay_report is not None
        else "legacy_parameters_not_real_calibrated"
    )
    metadata = {
        "phase": "4.5",
        "algorithm": "stable_baselines3_ppo",
        "time_engine": "continuous_event",
        "calibration_status": calibration_status,
        "observation_schema_version": "pre-route-observation/2.0",
        "action_schema_version": "pre-route-action/2.0",
        "action_mapping": {"0": "LOCAL_FIRST", "1": "DIRECT_CLOUD"},
        "legacy_25d_discrete3_compatible": False,
        "post_edge_rules": str(Path(args.post_edge_rules).resolve()),
        "mean_eval_reward": float(mean_reward),
        "std_eval_reward": float(std_reward),
        "requested_timesteps": args.timesteps,
        "actual_timesteps": actual_timesteps,
        "observation_names": OBSERVATION_NAMES,
        "environment": vars(args),
        "task_profiles": profiles_as_dict(),
        "hhk_v2_replay_report": replay_report,
    }
    with (results_dir / f"{output_path.stem}_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    pd.DataFrame(
        [{"mean_reward": mean_reward, "std_reward": std_reward}]
    ).to_csv(
        results_dir / f"{output_path.stem}_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest_path = publish_policy_bundle(
        model_path=output_path,
        release_dir=args.release_dir,
        rules_path=args.post_edge_rules,
        policy_version=args.policy_version,
        training_metadata={
            "seed": args.seed,
            "requested_timesteps": args.timesteps,
            "actual_timesteps": actual_timesteps,
            "episode_steps": args.episode_steps,
            "mean_eval_reward": float(mean_reward),
            "std_eval_reward": float(std_reward),
            "environment": vars(args),
        },
        calibration_metadata={
            "calibration_status": calibration_status,
            "scope": (
                "HHK V2 AI4I matched edge-result replay; network, cloud and resource dynamics remain simulated"
                if replay_report is not None
                else "legacy simulator parameters"
            ),
            "hhk_v2_replay_report": replay_report,
        },
    )
    print(f"模型已保存：{output_path.with_suffix('.zip')}")
    print(f"策略包已发布：{manifest_path}")
    print(
        f"{args.eval_episodes}个episode评估奖励："
        f"{mean_reward:.3f} ± {std_reward:.3f}"
    )


if __name__ == "__main__":
    main()
