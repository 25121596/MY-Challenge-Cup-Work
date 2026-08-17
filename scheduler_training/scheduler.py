from __future__ import annotations

from pathlib import Path

import numpy as np

from models import Action, PPO_ACTION_TO_ROUTE, Task
from observation import OBSERVATION_NAMES, build_observation


def _task_factors(task: Task, state: dict) -> dict[str, float]:
    edge_resource = max(float(state["edge_resource"]), 0.10)
    cloud_resource = max(float(state["cloud_resource"]), 0.10)
    bandwidth = max(float(state["bandwidth_mbps"]), 1.0)
    network_quality = float(np.clip(state["network_quality"], 0.0, 1.0))
    complexity_factor = 0.80 + 0.40 * float(np.clip(task.complexity, 0.0, 1.0))
    return {
        "edge_resource": edge_resource,
        "cloud_resource": cloud_resource,
        "bandwidth_mbps": bandwidth,
        "network_quality": network_quality,
        "complexity_factor": complexity_factor,
        "edge_speed_factor": 0.50 + 0.50 * edge_resource,
        "cloud_speed_factor": 0.55 + 0.45 * cloud_resource,
    }


def estimate_local_first_review_path(
    task: Task,
    state: dict,
    *,
    edge_stage_s: float,
) -> dict[str, float]:
    """Estimate LOCAL_FIRST followed by a possible cloud review.

    The edge stage is the normal LOCAL_FIRST inference, not the former shorter
    pre-selected collaborative stage. Existing numeric priors remain unchanged.
    """

    factors = _task_factors(task, state)
    edge_wait_s = max(0.0, float(state.get("edge_workload_s", 0.0)))
    cloud_workload_s = max(0.0, float(state.get("cloud_workload_s", 0.0)))

    cloud_stage_s = (
        float(task.collaborative_cloud_latency_s)
        * factors["complexity_factor"]
        / factors["cloud_speed_factor"]
    )
    upload_mb = max(0.0, float(task.size_mb) * float(task.collaborative_payload_ratio))
    upload_time_s = upload_mb * 8.0 / factors["bandwidth_mbps"]
    network_rtt_s = 0.005 + (1.0 - factors["network_quality"]) * 0.075
    transfer_total_s = upload_time_s + network_rtt_s

    time_before_cloud_arrival_s = edge_wait_s + edge_stage_s + transfer_total_s
    cloud_wait_s = max(0.0, cloud_workload_s - time_before_cloud_arrival_s)
    total_s = (
        edge_wait_s
        + edge_stage_s
        + transfer_total_s
        + cloud_wait_s
        + cloud_stage_s
    )

    return {
        "local_first_review_delay": total_s,
        "local_first_review_edge_wait_s": edge_wait_s,
        "local_first_review_edge_stage_s": edge_stage_s,
        "local_first_review_upload_mb": upload_mb,
        "local_first_review_upload_time_s": upload_time_s,
        "local_first_review_network_rtt_s": network_rtt_s,
        "local_first_review_transfer_total_s": transfer_total_s,
        "local_first_review_cloud_wait_s": cloud_wait_s,
        "local_first_review_cloud_stage_s": cloud_stage_s,
        "local_first_review_remaining_s": transfer_total_s + cloud_wait_s + cloud_stage_s,
        "local_first_review_expected_accuracy": float(task.collaborative_accuracy),
    }


def estimate_delays_for_task(task: Task, state: dict) -> dict[str, float]:
    """Estimate the two pre-route actions and conditional review path."""

    factors = _task_factors(task, state)
    edge_service_time = (
        float(task.edge_base_latency_s)
        * factors["complexity_factor"]
        / factors["edge_speed_factor"]
    )
    cloud_inference_time = (
        float(task.cloud_base_latency_s)
        * factors["complexity_factor"]
        / factors["cloud_speed_factor"]
    )
    transmission_time = float(task.size_mb) * 8.0 / factors["bandwidth_mbps"]
    network_rtt_s = 0.005 + (1.0 - factors["network_quality"]) * 0.075
    cloud_service_time = transmission_time + network_rtt_s + cloud_inference_time

    edge_waiting_time = max(0.0, float(state.get("edge_workload_s", 0.0)))
    cloud_waiting_time = max(0.0, float(state.get("cloud_workload_s", 0.0)))
    details = {
        "edge_delay": edge_waiting_time + edge_service_time,
        "cloud_delay": cloud_waiting_time + cloud_service_time,
        "edge_waiting_time": edge_waiting_time,
        "cloud_waiting_time": cloud_waiting_time,
        "edge_service_time": edge_service_time,
        "cloud_service_time": cloud_service_time,
        "cloud_inference_time": cloud_inference_time,
        "transmission_time": transmission_time,
        "network_rtt_s": network_rtt_s,
    }
    details.update(
        estimate_local_first_review_path(
            task,
            state,
            edge_stage_s=edge_service_time,
        )
    )
    return details


class RuleScheduler:
    """Pre-route rule baseline using only information available on arrival."""

    name = "规则调度器（2动作）"

    def __init__(
        self,
        edge_resource_threshold: float = 0.25,
        cloud_resource_threshold: float = 0.20,
        network_threshold: float = 0.15,
        max_edge_workload_s: float = 2.0,
        max_cloud_workload_s: float = 2.0,
    ) -> None:
        self.edge_resource_threshold = edge_resource_threshold
        self.cloud_resource_threshold = cloud_resource_threshold
        self.network_threshold = network_threshold
        self.max_edge_workload_s = max_edge_workload_s
        self.max_cloud_workload_s = max_cloud_workload_s

    def select_action(self, task: Task, state: dict) -> tuple[Action, str, dict]:
        details = estimate_delays_for_task(task, state)
        edge_delay = details["edge_delay"]
        cloud_delay = details["cloud_delay"]
        edge_available = (
            state["edge_resource"] >= self.edge_resource_threshold
            and state["edge_workload_s"] <= self.max_edge_workload_s
        )
        cloud_available = (
            state["cloud_resource"] >= self.cloud_resource_threshold
            and state["network_quality"] >= self.network_threshold
            and state["cloud_workload_s"] <= self.max_cloud_workload_s
        )
        edge_can_finish = edge_delay <= task.deadline_s
        cloud_can_finish = cloud_delay <= task.deadline_s

        if edge_available and edge_can_finish and edge_delay <= cloud_delay:
            return Action.LOCAL_FIRST, "边端可用且预计更快完成", details
        if cloud_available and cloud_can_finish:
            return Action.DIRECT_CLOUD, "云端预计可按时完成", details
        if not cloud_available:
            return Action.LOCAL_FIRST, "云端或网络不可用，选择本地优先", details
        if edge_delay <= cloud_delay:
            return Action.LOCAL_FIRST, "两侧均有超时风险，选择较快的本地优先", details
        return Action.DIRECT_CLOUD, "两侧均有超时风险，选择较快的直接云端", details


class AlwaysLocalFirstScheduler:
    name = "固定本地优先"

    def select_action(self, task: Task, state: dict) -> tuple[Action, str, dict]:
        return Action.LOCAL_FIRST, "基准策略：全部本地优先", estimate_delays_for_task(task, state)


class AlwaysCloudScheduler:
    name = "固定直接云端"

    def select_action(self, task: Task, state: dict) -> tuple[Action, str, dict]:
        return Action.DIRECT_CLOUD, "基准策略：全部直接云端", estimate_delays_for_task(task, state)


AlwaysEdgeScheduler = AlwaysLocalFirstScheduler


class PPOScheduler:
    """Load a 24D / Discrete(2) pre-route Stable-Baselines3 PPO model."""

    name = "PPO调度器"

    def __init__(
        self,
        model_path: str = "models/releases/pre_route_v2/pre_route_ppo.zip",
        deterministic: bool = True,
    ) -> None:
        path = Path(model_path)
        if path.suffix == "":
            path = path.with_suffix(".zip")
        if path.suffix.lower() != ".zip":
            raise ValueError(f"仅支持Stable-Baselines3 .zip模型：{model_path}")
        if not path.is_file():
            raise FileNotFoundError(f"未找到PPO模型：{path}")
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError("请先安装requirements.txt中的stable-baselines3") from exc

        self.model_path = str(path)
        self.deterministic = deterministic
        self.model = PPO.load(str(path))
        action_n = getattr(getattr(self.model, "action_space", None), "n", None)
        if action_n != 2:
            raise ValueError(
                f"模型动作空间为{action_n}，需要Discrete(2) pre-route模型；"
                "旧Discrete(3) PPO明确不兼容"
            )
        obs_shape = getattr(getattr(self.model, "observation_space", None), "shape", None)
        if obs_shape is not None and tuple(obs_shape) != (len(OBSERVATION_NAMES),):
            raise ValueError(
                f"模型观测空间为{obs_shape}，需要24D pre-route observation；"
                "旧25D PPO明确不兼容"
            )

    def select_action(self, task: Task, state: dict) -> tuple[Action, str, dict]:
        observation = build_observation(task, state)
        action_value, _ = self.model.predict(observation, deterministic=self.deterministic)
        action_index = int(np.asarray(action_value).item())
        if action_index not in PPO_ACTION_TO_ROUTE:
            raise ValueError(f"PPO输出非法动作：{action_index}")
        action = PPO_ACTION_TO_ROUTE[action_index]
        reason = "PPO选择本地优先" if action == Action.LOCAL_FIRST else "PPO选择直接云端"
        return action, reason, estimate_delays_for_task(task, state)
