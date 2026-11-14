"""Phase3奖励函数：统一使用三条路径的同一份时延估计结果。"""

from __future__ import annotations

from dataclasses import dataclass

from models import Action, Task


VALID_REWARD_MODES = {"legacy", "lyapunov", "deadline_aware"}


@dataclass
class RewardConfig:
    latency_weight: float = 2.0
    queue_penalty_weight: float = 0.2
    timeout_penalty: float = 20.0
    accuracy_penalty_weight: float = 3.0
    communication_penalty_weight: float = 0.25
    edge_compute_penalty_weight: float = 0.05
    cloud_compute_penalty_weight: float = 0.05
    cloud_accuracy: float = 0.95
    reward_mode: str = "deadline_aware"

    def __post_init__(self) -> None:
        if self.reward_mode not in VALID_REWARD_MODES:
            raise ValueError(f"非法reward_mode：{self.reward_mode}")
        nonnegative = {
            "latency_weight": self.latency_weight,
            "queue_penalty_weight": self.queue_penalty_weight,
            "timeout_penalty": self.timeout_penalty,
            "accuracy_penalty_weight": self.accuracy_penalty_weight,
            "communication_penalty_weight": self.communication_penalty_weight,
            "edge_compute_penalty_weight": self.edge_compute_penalty_weight,
            "cloud_compute_penalty_weight": self.cloud_compute_penalty_weight,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name}不能为负数")


class RewardCalculator:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    @staticmethod
    def _lyapunov_value(edge_queue: int, cloud_queue: int) -> float:
        return 0.5 * (edge_queue**2 + cloud_queue**2)

    def calculate(
        self,
        *,
        task: Task | None,
        action: Action,
        details: dict,
        state_before: dict,
        state_after: dict,
        timeout_count: int,
    ) -> dict:
        selected_delay_s = 0.0
        expected_accuracy = 1.0
        upload_mb = 0.0
        edge_compute_s = 0.0
        cloud_compute_s = 0.0

        if task is not None and action != Action.IDLE:
            if action == Action.EDGE:
                selected_delay_s = float(details["edge_delay"])
                expected_accuracy = float(task.confidence)
                edge_compute_s = float(details["edge_service_time"])
            elif action == Action.CLOUD:
                selected_delay_s = float(details["cloud_delay"])
                expected_accuracy = float(
                    task.estimated_cloud_accuracy
                    if task.estimated_cloud_accuracy > 0
                    else self.config.cloud_accuracy
                )
                upload_mb = float(task.size_mb)
                cloud_compute_s = float(details["cloud_inference_time"])
            elif action == Action.COLLABORATIVE:
                selected_delay_s = float(details["collaborative_delay"])
                expected_accuracy = float(task.collaborative_accuracy)
                upload_mb = float(details["collaborative_upload_mb"])
                edge_compute_s = float(details["collaborative_edge_stage_s"])
                cloud_compute_s = float(details["collaborative_cloud_stage_s"])
            else:
                raise ValueError(f"奖励函数收到非法动作：{action}")

        deadline_ratio = (
            selected_delay_s / max(task.deadline_s, 1e-6)
            if task is not None and action != Action.IDLE
            else 0.0
        )
        total_queue_length = (
            int(state_after.get("edge_queue", 0))
            + int(state_after.get("cloud_queue", 0))
        )

        delay_basis = (
            min(deadline_ratio, 5.0)
            if self.config.reward_mode == "deadline_aware"
            else selected_delay_s
        )
        delay_penalty_value = self.config.latency_weight * delay_basis
        timeout_penalty_value = self.config.timeout_penalty * int(timeout_count)
        accuracy_penalty_value = self.config.accuracy_penalty_weight * (
            1.0 - expected_accuracy
        )
        communication_penalty_value = (
            self.config.communication_penalty_weight * upload_mb
        )
        resource_penalty_value = (
            self.config.edge_compute_penalty_weight * edge_compute_s
            + self.config.cloud_compute_penalty_weight * cloud_compute_s
        )

        lyapunov_before = self._lyapunov_value(
            int(state_before.get("edge_queue", 0)),
            int(state_before.get("cloud_queue", 0)),
        )
        lyapunov_after = self._lyapunov_value(
            int(state_after.get("edge_queue", 0)),
            int(state_after.get("cloud_queue", 0)),
        )
        lyapunov_drift = lyapunov_after - lyapunov_before
        queue_penalty_value = self.config.queue_penalty_weight * (
            lyapunov_drift
            if self.config.reward_mode == "lyapunov"
            else total_queue_length
        )

        reward = -(
            delay_penalty_value
            + queue_penalty_value
            + timeout_penalty_value
            + accuracy_penalty_value
            + communication_penalty_value
            + resource_penalty_value
        )
        return {
            "reward": reward,
            "selected_delay_s": selected_delay_s,
            "deadline_ratio": deadline_ratio,
            "expected_accuracy": expected_accuracy,
            "selected_upload_mb": upload_mb,
            "edge_compute_s": edge_compute_s,
            "cloud_compute_s": cloud_compute_s,
            "total_queue_length": total_queue_length,
            "delay_penalty": delay_penalty_value,
            "queue_penalty": queue_penalty_value,
            "timeout_penalty": timeout_penalty_value,
            "accuracy_penalty": accuracy_penalty_value,
            "communication_penalty": communication_penalty_value,
            "resource_penalty": resource_penalty_value,
            "lyapunov_before": lyapunov_before,
            "lyapunov_after": lyapunov_after,
            "lyapunov_drift": lyapunov_drift,
        }
