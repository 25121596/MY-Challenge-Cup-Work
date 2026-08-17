"""不依赖Gymnasium的事件驱动Phase3 PPO环境核心。

PPO只在新任务到达时决策一次。Gym动作索引：
0=EDGE，1=CLOUD，2=COLLABORATIVE。
没有新任务的物理时隙由环境自动推进，并把期间奖励累积到上一决策。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models import Action
from observation import build_observation
from scheduler import RuleScheduler, estimate_delays_for_task
from simulator import CloudEdgeSimulator


class CloudEdgePPOCore:
    def __init__(
        self,
        seed: int = 42,
        episode_steps: int = 300,
        arrival_probability: float = 0.10,
        slot_duration: float = 0.05,
        latency_weight: float = 2.0,
        queue_penalty_weight: float = 0.2,
        timeout_penalty: float = 20.0,
        accuracy_penalty_weight: float = 3.0,
        communication_penalty_weight: float = 0.25,
        edge_compute_penalty_weight: float = 0.05,
        cloud_compute_penalty_weight: float = 0.05,
        cloud_accuracy: float = 0.95,
        queue_policy: str = "edf",
        network_mode: str = "markov",
        resource_mode: str = "consume_recover",
        reward_mode: str = "deadline_aware",
        task_source: str = "typed_simulation",
        edge_input_path: str | None = None,
        edge_input_strict: bool = True,
        edge_input_interval_s: float = 0.05,
        ppo_reward_scale: float = 0.05,
    ) -> None:
        if episode_steps <= 0:
            raise ValueError("episode_steps必须大于0")
        if ppo_reward_scale <= 0:
            raise ValueError("ppo_reward_scale必须大于0")
        self.base_seed = seed
        self.episode_steps = episode_steps
        self.arrival_probability = arrival_probability
        self.slot_duration = slot_duration
        self.latency_weight = latency_weight
        self.queue_penalty_weight = queue_penalty_weight
        self.timeout_penalty = timeout_penalty
        self.accuracy_penalty_weight = accuracy_penalty_weight
        self.communication_penalty_weight = communication_penalty_weight
        self.edge_compute_penalty_weight = edge_compute_penalty_weight
        self.cloud_compute_penalty_weight = cloud_compute_penalty_weight
        self.cloud_accuracy = cloud_accuracy
        self.queue_policy = queue_policy
        self.network_mode = network_mode
        self.resource_mode = resource_mode
        self.reward_mode = reward_mode
        self.task_source = task_source
        self.edge_input_path = edge_input_path
        self.edge_input_strict = edge_input_strict
        self.edge_input_interval_s = edge_input_interval_s
        self.ppo_reward_scale = float(ppo_reward_scale)

        self.simulator: CloudEdgeSimulator | None = None
        self.context: dict | None = None
        self.decision_step = 0
        self.episode_index = 0

    def _create_simulator(self, seed: int) -> CloudEdgeSimulator:
        return CloudEdgeSimulator(
            seed=seed,
            arrival_probability=self.arrival_probability,
            slot_duration=self.slot_duration,
            scheduler=RuleScheduler(),
            latency_weight=self.latency_weight,
            queue_penalty_weight=self.queue_penalty_weight,
            timeout_penalty=self.timeout_penalty,
            accuracy_penalty_weight=self.accuracy_penalty_weight,
            communication_penalty_weight=self.communication_penalty_weight,
            edge_compute_penalty_weight=self.edge_compute_penalty_weight,
            cloud_compute_penalty_weight=self.cloud_compute_penalty_weight,
            cloud_accuracy=self.cloud_accuracy,
            queue_policy=self.queue_policy,
            network_mode=self.network_mode,
            resource_mode=self.resource_mode,
            reward_mode=self.reward_mode,
            task_source=self.task_source,
            edge_input_path=self.edge_input_path,
            edge_input_strict=self.edge_input_strict,
            edge_input_interval_s=self.edge_input_interval_s,
        )

    def _advance_to_decision(self) -> tuple[dict | None, float, int]:
        if self.simulator is None:
            raise RuntimeError("环境尚未创建")
        accumulated_reward = 0.0
        advanced_slots = 0
        for _ in range(100_000):
            if self.simulator.external_tasks_finished():
                return None, accumulated_reward, advanced_slots
            context = self.simulator.prepare_decision()
            if context["task"] is not None:
                return context, accumulated_reward, advanced_slots
            record = self.simulator.apply_decision(
                context,
                Action.IDLE,
                "事件驱动环境自动推进空闲时隙",
                None,
            )
            accumulated_reward += float(record["reward"])
            advanced_slots += 1
        raise RuntimeError("等待下一任务超过安全上限")

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.base_seed = int(seed)
            self.episode_index = 0
        episode_seed = self.base_seed + self.episode_index
        self.episode_index += 1
        self.simulator = self._create_simulator(episode_seed)
        self.decision_step = 0
        self.context, _, initial_idle_slots = self._advance_to_decision()
        observation = build_observation(
            None if self.context is None else self.context["task"],
            self.simulator.get_state()
            if self.context is None
            else self.context["state_before"],
        )
        report = self.simulator.edge_input_load_report
        return observation, {
            "episode_seed": episode_seed,
            "queue_policy": self.queue_policy,
            "task_source": self.task_source,
            "initial_idle_slots": initial_idle_slots,
            "edge_inputs_loaded": 0 if report is None else report.loaded_count,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.simulator is None or self.context is None:
            raise RuntimeError("环境没有可决策任务，请先reset")
        task = self.context["task"]
        if task is None:
            raise RuntimeError("事件驱动环境不应向PPO暴露空任务")

        mapping = {
            0: (Action.EDGE, "PPO训练动作：选择边端"),
            1: (Action.CLOUD, "PPO训练动作：选择云端"),
            2: (Action.COLLABORATIVE, "PPO训练动作：选择边端初判与云端复核"),
        }
        if int(action) not in mapping:
            raise ValueError(f"非法动作：{action}")
        selected_action, reason = mapping[int(action)]
        details = estimate_delays_for_task(task, self.context["state_before"])
        record = self.simulator.apply_decision(
            self.context,
            selected_action,
            reason,
            details,
        )
        transition_reward = float(record["reward"])
        self.decision_step += 1

        terminated = self.simulator.external_tasks_finished()
        truncated = not terminated and self.decision_step >= self.episode_steps
        idle_slots = 0
        if not terminated and not truncated:
            self.context, idle_reward, idle_slots = self._advance_to_decision()
            transition_reward += idle_reward
            terminated = (
                self.context is None and self.simulator.external_tasks_finished()
            )

        if terminated or truncated or self.context is None:
            next_observation = build_observation(None, self.simulator.get_state())
            self.context = None
        else:
            next_observation = build_observation(
                self.context["task"],
                self.context["state_before"],
            )

        info = {
            "time": record["time"],
            "task_id": record["task_id"],
            "task_type": record["task_type"],
            "action_name": record["action_name"],
            "timeout_count": record["timeout_count"],
            "selected_delay_s": record["selected_delay_s"],
            "expected_accuracy": record["expected_accuracy"],
            "selected_upload_mb": record["selected_upload_mb"],
            "idle_slots_advanced": idle_slots,
            "cumulative_reward": self.simulator.cumulative_reward,
        }
        if terminated or truncated:
            info["episode_summary"] = self.simulator.get_metrics()
        info["raw_transition_reward"] = transition_reward
        scaled_reward = float(
            np.clip(transition_reward * self.ppo_reward_scale, -10.0, 10.0)
        )
        return next_observation, scaled_reward, terminated, truncated, info
