from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np

from edge_input_loader import load_edge_inputs
from policy_bundle_loader import PolicyBundle, load_policy_bundle
from pre_route_observation import ACTION_MAPPING, build_pre_route_observation
from real_task_runtime import RealTaskManager, RealTaskEnvelope


class Policy(Protocol):
    def predict_action(self, observation: np.ndarray) -> int: ...


class RealPPOPolicy:
    """Load the 24D/Discrete(2) PPO contained in a validated bundle.

    Integration opportunistically records the policy distribution/value estimate for
    profiling.  If the installed SB3 version does not expose compatible policy
    internals, these fields remain None and action prediction still works.
    """

    def __init__(self, policy_bundle: str | Path | PolicyBundle, deterministic: bool = True) -> None:
        self.bundle = (
            policy_bundle
            if isinstance(policy_bundle, PolicyBundle)
            else load_policy_bundle(policy_bundle)
        )
        path = self.bundle.model_path
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "缺少stable-baselines3/gymnasium，请执行 pip install -r requirements.txt"
            ) from exc
        self.path = path
        self.deterministic = bool(deterministic)
        self.model = PPO.load(str(path))
        action_n = getattr(getattr(self.model, "action_space", None), "n", None)
        obs_shape = getattr(getattr(self.model, "observation_space", None), "shape", None)
        if action_n != 2:
            raise ValueError(
                f"PPO动作空间为{action_n}；旧Discrete(3)模型不兼容，当前需要Discrete(2)"
            )
        if obs_shape is not None and tuple(obs_shape) != (24,):
            raise ValueError(
                f"PPO观测维度为{obs_shape}；旧25D模型不兼容，当前需要24D"
            )

    def predict_action(self, observation: np.ndarray) -> int:
        return self.predict_with_info(observation)[0]

    def predict_with_info(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        value = int(np.asarray(action).item())
        if value not in (0, 1):
            raise ValueError(f"PPO输出非法动作: {value}")
        info: dict[str, Any] = {
            "deterministic": self.deterministic,
            "action_probabilities": None,
            "value_estimate": None,
        }
        try:
            import torch

            obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = getattr(getattr(distribution, "distribution", None), "probs", None)
                if probs is not None:
                    info["action_probabilities"] = [float(x) for x in probs.detach().cpu().numpy().reshape(-1)]
                predict_values = getattr(self.model.policy, "predict_values", None)
                if callable(predict_values):
                    values = predict_values(obs_tensor)
                    info["value_estimate"] = float(values.detach().cpu().numpy().reshape(-1)[0])
        except Exception as exc:
            # Profiling metadata is best-effort and must never change scheduling.
            info["policy_profile_error"] = f"{type(exc).__name__}: {exc}"
        return value, info


class CyclicTestPolicy:
    """Only for smoke tests; never used as a replacement for a trained PPO."""

    def __init__(self, sequence: tuple[int, ...] = (0, 1)) -> None:
        if not sequence or any(value not in (0, 1) for value in sequence):
            raise ValueError("test policy sequence只能包含0/1")
        self.sequence = tuple(sequence)
        self.index = 0

    def predict_action(self, observation: np.ndarray) -> int:
        return self.predict_with_info(observation)[0]

    def predict_with_info(self, observation: np.ndarray) -> tuple[int, dict[str, Any]]:
        if tuple(observation.shape) != (24,):
            raise ValueError(f"smoke policy收到非24维观测: {observation.shape}")
        value = self.sequence[self.index % len(self.sequence)]
        self.index += 1
        return value, {
            "deterministic": True,
            "action_probabilities": None,
            "value_estimate": None,
            "policy_type": "cyclic_test_policy",
        }


@dataclass
class RuntimeRunResult:
    input_count: int
    submitted_count: int
    completed_count: int
    timeout_count: int
    records: list[dict[str, Any]]
    service_time_estimates: dict[str, dict[str, float | int]]


class RealPPORuntime:
    def __init__(
        self,
        *,
        policy: Policy,
        policy_bundle: PolicyBundle | str | Path,
        state_provider: Any,
        task_manager: RealTaskManager,
    ) -> None:
        self.bundle = (
            policy_bundle
            if isinstance(policy_bundle, PolicyBundle)
            else load_policy_bundle(policy_bundle)
        )
        self.observation_names = list(self.bundle.observation_names)
        self.policy = policy
        self.state_provider = state_provider
        self.task_manager = task_manager
        self.task_manager.configure_policy_bundle(self.bundle)
        if len(self.observation_names) != 24:
            raise RuntimeError(f"bundle observation schema不是24维: {len(self.observation_names)}")

    def load_tasks(self, input_path: str | Path, *, strict: bool = True) -> list[Any]:
        tasks, _ = load_edge_inputs(input_path, strict=strict)
        return list(tasks)

    def _decision_for(
        self,
        env: RealTaskEnvelope,
    ) -> tuple[np.ndarray, dict[str, Any], int, int, dict[str, Any]]:
        state = self.state_provider.get_state(self.task_manager)
        observation = build_pre_route_observation(
            env.task,
            state,
            observation_names=self.observation_names,
        )
        predictor = getattr(self.policy, "predict_with_info", None)
        if callable(predictor):
            requested_action, policy_info = predictor(observation)
        else:
            requested_action = self.policy.predict_action(observation)
            policy_info = {}
        requested_action = int(requested_action)
        if requested_action not in (0, 1):
            raise ValueError(f"PPO输出非法pre-route动作: {requested_action}")

        effective_action = requested_action
        runtime_override = None
        network = state.get("real_network_metrics", {})
        cloud_reachable = network.get("reachable") if isinstance(network, dict) else None
        if requested_action == 1 and cloud_reachable is False:
            effective_action = 0
            runtime_override = "DIRECT_CLOUD_TO_LOCAL_FIRST_CLOUD_UNREACHABLE"
        policy_info = dict(policy_info or {})
        policy_info.update({
            "policy_version": self.bundle.policy_version,
            "requested_action_index": requested_action,
            "requested_action_name": ACTION_MAPPING[str(requested_action)],
            "effective_action_index": effective_action,
            "effective_action_name": ACTION_MAPPING[str(effective_action)],
            "runtime_override": runtime_override,
        })
        return observation, state, requested_action, effective_action, policy_info

    def run_tasks(
        self,
        tasks: list[Any],
        *,
        limit: int = 0,
        arrival_interval_s: float = 0.05,
        use_source_timestamps: bool = True,
        drain_timeout_s: float = 180.0,
        verbose: bool = True,
    ) -> RuntimeRunResult:
        if arrival_interval_s <= 0:
            raise ValueError("arrival_interval_s必须大于0")
        selected = tasks if limit <= 0 else tasks[:limit]
        self.task_manager.start()
        start = time.monotonic()

        for index, task in enumerate(selected):
            source_offset = getattr(task, "source_arrival_offset_s", None)
            if use_source_timestamps and source_offset is not None:
                offset = float(source_offset)
            else:
                offset = index * float(arrival_interval_s)
            due = start + max(0.0, offset)
            sleep_s = due - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

            env = self.task_manager.register_arrival(task)
            observation, state, requested_action, effective_action, policy_info = self._decision_for(env)
            self.task_manager.route(
                env,
                effective_action,
                requested_action_index=requested_action,
                state_before=state,
                observation=observation.tolist(),
                policy_info=policy_info,
            )
            if verbose:
                print(
                    f"[{index + 1:03d}] {env.task_id} "
                    f"{getattr(task, 'task_type', '')} -> "
                    f"{ACTION_MAPPING[str(effective_action)]} | "
                    f"q=({state['edge_queue']},{state['cloud_queue']}) "
                    f"res=({state['edge_resource']:.2f},{state['cloud_resource']:.2f}) "
                    f"net={state['network_quality']:.2f}"
                )

        drained = self.task_manager.wait_all(timeout_s=drain_timeout_s)
        if verbose and not drained:
            print("警告: drain timeout到达，仍有真实模型调用未返回。")

        records = self.task_manager.records()
        completed = sum(row.get("status") == "completed" for row in records)
        timeout = sum(row.get("status") == "timeout" for row in records)
        return RuntimeRunResult(
            input_count=len(tasks),
            submitted_count=len(selected),
            completed_count=int(completed),
            timeout_count=int(timeout),
            records=records,
            service_time_estimates=self.task_manager.estimator.snapshot(),
        )
