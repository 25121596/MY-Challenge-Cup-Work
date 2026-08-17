"""Phase4.5 reward calculation from StepOutcome.

Numeric weights and formulas are unchanged from Phase4 v1.8.5.  The structural
change is that RewardCalculator no longer estimates environment effects itself;
it consumes a StepOutcome produced by the simulator/runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from outcome import StepOutcome


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

    def calculate(self, outcome: StepOutcome) -> dict:
        task = outcome.task
        selected_delay_s = float(outcome.selected_delay_s)
        expected_accuracy = float(outcome.expected_accuracy)
        upload_mb = float(outcome.upload_mb)
        edge_compute_s = float(outcome.edge_compute_s)
        cloud_compute_s = float(outcome.cloud_compute_s)

        deadline_ratio = (
            selected_delay_s / max(task.deadline_s, 1e-6)
            if task is not None and outcome.action.name != "IDLE"
            else 0.0
        )
        total_queue_length = int(outcome.state_after.edge_queue) + int(outcome.state_after.cloud_queue)

        delay_basis = (
            min(deadline_ratio, 5.0)
            if self.config.reward_mode == "deadline_aware"
            else selected_delay_s
        )
        delay_penalty_value = self.config.latency_weight * delay_basis
        timeout_penalty_value = self.config.timeout_penalty * int(outcome.timeout_count)
        accuracy_penalty_value = self.config.accuracy_penalty_weight * (1.0 - expected_accuracy)
        communication_penalty_value = self.config.communication_penalty_weight * upload_mb
        resource_penalty_value = (
            self.config.edge_compute_penalty_weight * edge_compute_s
            + self.config.cloud_compute_penalty_weight * cloud_compute_s
        )

        lyapunov_before = self._lyapunov_value(
            int(outcome.state_before.edge_queue), int(outcome.state_before.cloud_queue)
        )
        lyapunov_after = self._lyapunov_value(
            int(outcome.state_after.edge_queue), int(outcome.state_after.cloud_queue)
        )
        lyapunov_drift = lyapunov_after - lyapunov_before
        queue_penalty_value = self.config.queue_penalty_weight * (
            lyapunov_drift if self.config.reward_mode == "lyapunov" else total_queue_length
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
