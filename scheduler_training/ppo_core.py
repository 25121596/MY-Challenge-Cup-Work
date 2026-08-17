"""Gym-independent Phase4.5 PPO core.

PPO decides only at task-arrival decision events. Physical service completion,
transmission, deadline expiry, resource evolution and network evolution are
advanced by the continuous-time simulator between decisions.

Gym action indices are: 0=LOCAL_FIRST, 1=DIRECT_CLOUD.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models import Action, PPO_ACTION_TO_ROUTE
from observation import build_observation
from scheduler import RuleScheduler
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
        post_edge_rules_path: str | None = None,
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
        self.post_edge_rules_path = post_edge_rules_path

        self.simulator: CloudEdgeSimulator | None = None
        self.context: dict | None = None
        self.decision_step = 0
        self.episode_index = 0

    def _create_simulator(self, seed: int) -> CloudEdgeSimulator:
        simulator_kwargs = dict(
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
        if self.post_edge_rules_path is not None:
            simulator_kwargs["post_edge_rules_path"] = self.post_edge_rules_path
        return CloudEdgeSimulator(**simulator_kwargs)

    def _advance_to_decision(self) -> tuple[dict | None, float, dict[str, Any]]:
        if self.simulator is None:
            raise RuntimeError("环境尚未创建")
        return self.simulator.advance_to_next_decision()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.base_seed = int(seed)
            self.episode_index = 0
        episode_seed = self.base_seed + self.episode_index
        self.episode_index += 1
        self.simulator = self._create_simulator(episode_seed)
        self.decision_step = 0
        self.context, initial_event_reward, advance_info = self._advance_to_decision()
        self.simulator.cumulative_reward += initial_event_reward
        state = self.simulator.get_system_state() if self.context is None else self.context["state_before"]
        observation = build_observation(None if self.context is None else self.context["task"], state)
        report = self.simulator.edge_input_load_report
        return observation, {
            "episode_seed": episode_seed,
            "queue_policy": self.queue_policy,
            "task_source": self.task_source,
            "time_engine": "continuous_event",
            "initial_advanced_time_s": advance_info.get("advanced_time_s", 0.0),
            # Kept for old logging scripts; it is now only a reporting equivalent.
            "initial_idle_slots": int(round(advance_info.get("advanced_time_s", 0.0) / self.slot_duration)),
            "edge_inputs_loaded": 0 if report is None else report.loaded_count,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.simulator is None or self.context is None:
            raise RuntimeError("环境没有可决策任务，请先reset")
        task = self.context["task"]
        if task is None:
            raise RuntimeError("Phase4.5不会向PPO暴露空任务")

        action_index = int(action)
        if action_index not in PPO_ACTION_TO_ROUTE:
            raise ValueError(f"非法动作：{action}")
        selected_action = PPO_ACTION_TO_ROUTE[action_index]
        reason = (
            "PPO训练动作：选择本地优先"
            if selected_action == Action.LOCAL_FIRST
            else "PPO训练动作：选择直接云端"
        )
        details = self.simulator.service_time_model.estimate(task, self.context["state_before"])
        record = self.simulator.apply_decision(self.context, selected_action, reason, details)
        transition_reward = float(record["reward"])
        self.decision_step += 1

        terminated = self.simulator.external_tasks_finished()
        truncated = not terminated and self.decision_step >= self.episode_steps
        advance_info: dict[str, Any] = {"advanced_time_s": 0.0, "event_count": 0}

        if not terminated and not truncated:
            self.context, event_reward, advance_info = self._advance_to_decision()
            transition_reward += float(event_reward)
            self.simulator.cumulative_reward += float(event_reward)
            terminated = self.context is None and self.simulator.external_tasks_finished()
        else:
            self.context = None

        if terminated or truncated or self.context is None:
            next_observation = build_observation(None, self.simulator.get_system_state())
            self.context = None
        else:
            next_observation = build_observation(self.context["task"], self.context["state_before"])

        info = {
            "time": record["time"],
            "time_s": record["time_s"],
            "task_id": record["task_id"],
            "task_type": record["task_type"],
            "action_name": record["action_name"],
            "predicted_post_edge_decision": record.get("predicted_post_edge_decision"),
            "post_edge_rule_version": record.get("post_edge_rule_version"),
            "timeout_count": len(advance_info.get("timeout_task_ids", [])),
            "selected_delay_s": record["selected_delay_s"],
            "expected_accuracy": record["expected_accuracy"],
            "selected_upload_mb": record["selected_upload_mb"],
            "advanced_time_s": float(advance_info.get("advanced_time_s", 0.0)),
            "physical_event_count": int(advance_info.get("event_count", 0)),
            "completed_task_ids": advance_info.get("completed_task_ids", []),
            "timeout_task_ids": advance_info.get("timeout_task_ids", []),
            # Backward-compatible reporting field, not a simulation engine counter.
            "idle_slots_advanced": int(round(float(advance_info.get("advanced_time_s", 0.0)) / self.slot_duration)),
            "cumulative_reward": self.simulator.cumulative_reward,
        }
        if terminated or truncated:
            info["episode_summary"] = self.simulator.get_metrics()
        info["raw_transition_reward"] = transition_reward
        scaled_reward = float(np.clip(transition_reward * self.ppo_reward_scale, -10.0, 10.0))
        return next_observation, scaled_reward, terminated, truncated, info
