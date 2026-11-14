"""Phase2云边资源与网络动力学。

所有状态转移按slot_duration缩放，避免把Phase1的1秒步长参数直接用于50ms
时隙而造成资源和网络变化过快。
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np


VALID_RESOURCE_MODES = {"legacy", "consume_recover"}
VALID_NETWORK_MODES = {"legacy", "markov"}


class EnvironmentDynamics:
    def __init__(
        self,
        slot_duration: float = 0.05,
        resource_mode: str = "consume_recover",
        network_mode: str = "markov",
    ) -> None:
        if slot_duration <= 0:
            raise ValueError("slot_duration必须大于0")
        if resource_mode not in VALID_RESOURCE_MODES:
            raise ValueError(f"非法resource_mode：{resource_mode}")
        if network_mode not in VALID_NETWORK_MODES:
            raise ValueError(f"非法network_mode：{network_mode}")

        self.slot_duration = float(slot_duration)
        self.resource_mode = resource_mode
        self.network_mode = network_mode

        self.min_bandwidth_mbps = 5.0
        self.max_bandwidth_mbps = 50.0

        self.edge_idle_target = 0.85
        self.edge_busy_target = 0.55
        self.cloud_idle_target = 0.90
        self.cloud_busy_target = 0.65
        self.mean_reversion_speed_per_s = 0.15
        self.edge_noise_std_per_s = 0.015
        self.cloud_noise_std_per_s = 0.010
        self.network_noise_std_per_s = 0.07

        self.edge_resource_min = 0.20
        self.edge_resource_max = 1.00
        self.cloud_resource_min = 0.30
        self.cloud_resource_max = 1.00
        self.network_quality_min = 0.00
        self.network_quality_max = 1.00

        # 每秒消耗/恢复率，执行时按时隙长度缩放。
        self.edge_recovery_per_s = 0.045
        self.cloud_recovery_per_s = 0.0375
        self.edge_consumption_per_s = 0.090
        self.cloud_consumption_per_s = 0.075

        # 以1秒为参考的Markov状态转移概率。
        self.p_good_to_bad_per_s = 0.10
        self.p_bad_to_good_per_s = 0.30
        self.good_network_quality = 1.00
        self.bad_network_quality = 0.20

        self.reset()

    @property
    def bandwidth_mbps(self) -> float:
        return self.min_bandwidth_mbps + self.network_quality * (
            self.max_bandwidth_mbps - self.min_bandwidth_mbps
        )

    @staticmethod
    def node_workload(
        queue: deque,
        active: dict | None,
        slot_duration: float,
    ) -> float:
        workload_slots = sum(item["remaining_slots"] for item in queue)
        if active is not None:
            workload_slots += active["remaining_slots"]
        return float(workload_slots * slot_duration)

    def _step_noise(self, std_per_s: float) -> float:
        return std_per_s * math.sqrt(self.slot_duration)

    def _update_resources_legacy(
        self,
        *,
        edge_busy: bool,
        cloud_busy: bool,
        rng: np.random.Generator,
    ) -> None:
        alpha = 1.0 - math.exp(
            -self.mean_reversion_speed_per_s * self.slot_duration
        )
        edge_target = self.edge_busy_target if edge_busy else self.edge_idle_target
        cloud_target = self.cloud_busy_target if cloud_busy else self.cloud_idle_target

        self.edge_resource = float(
            np.clip(
                self.edge_resource
                + alpha * (edge_target - self.edge_resource)
                + rng.normal(0.0, self._step_noise(self.edge_noise_std_per_s)),
                self.edge_resource_min,
                self.edge_resource_max,
            )
        )
        self.cloud_resource = float(
            np.clip(
                self.cloud_resource
                + alpha * (cloud_target - self.cloud_resource)
                + rng.normal(0.0, self._step_noise(self.cloud_noise_std_per_s)),
                self.cloud_resource_min,
                self.cloud_resource_max,
            )
        )

    def _update_resources_consume_recover(
        self,
        *,
        edge_executed: bool,
        cloud_executed: bool,
        rng: np.random.Generator,
    ) -> None:
        edge_delta = (
            -self.edge_consumption_per_s
            if edge_executed
            else self.edge_recovery_per_s
        ) * self.slot_duration
        cloud_delta = (
            -self.cloud_consumption_per_s
            if cloud_executed
            else self.cloud_recovery_per_s
        ) * self.slot_duration

        self.edge_resource = float(
            np.clip(
                self.edge_resource
                + edge_delta
                + rng.normal(0.0, self._step_noise(self.edge_noise_std_per_s)),
                self.edge_resource_min,
                self.edge_resource_max,
            )
        )
        self.cloud_resource = float(
            np.clip(
                self.cloud_resource
                + cloud_delta
                + rng.normal(0.0, self._step_noise(self.cloud_noise_std_per_s)),
                self.cloud_resource_min,
                self.cloud_resource_max,
            )
        )

    def _update_network_legacy(self, rng: np.random.Generator) -> None:
        self.network_quality = float(
            np.clip(
                self.network_quality
                + rng.normal(0.0, self._step_noise(self.network_noise_std_per_s)),
                self.network_quality_min,
                self.network_quality_max,
            )
        )
        self.network_state = "GOOD" if self.network_quality >= 0.5 else "BAD"

    @staticmethod
    def _probability_for_step(probability_per_s: float, slot_duration: float) -> float:
        return 1.0 - (1.0 - probability_per_s) ** slot_duration

    def _update_network_markov(self, rng: np.random.Generator) -> None:
        if self.network_state == "GOOD":
            switch_probability = self._probability_for_step(
                self.p_good_to_bad_per_s,
                self.slot_duration,
            )
            if rng.random() < switch_probability:
                self.network_state = "BAD"
        else:
            switch_probability = self._probability_for_step(
                self.p_bad_to_good_per_s,
                self.slot_duration,
            )
            if rng.random() < switch_probability:
                self.network_state = "GOOD"

        base = (
            self.good_network_quality
            if self.network_state == "GOOD"
            else self.bad_network_quality
        )
        self.network_quality = float(
            np.clip(
                base + rng.normal(0.0, 0.015),
                self.network_quality_min,
                self.network_quality_max,
            )
        )

    def update(
        self,
        *,
        edge_busy: bool,
        cloud_busy: bool,
        edge_executed: bool,
        cloud_executed: bool,
        rng: np.random.Generator,
    ) -> None:
        if self.resource_mode == "legacy":
            self._update_resources_legacy(
                edge_busy=edge_busy,
                cloud_busy=cloud_busy,
                rng=rng,
            )
        else:
            self._update_resources_consume_recover(
                edge_executed=edge_executed,
                cloud_executed=cloud_executed,
                rng=rng,
            )

        if self.network_mode == "legacy":
            self._update_network_legacy(rng)
        else:
            self._update_network_markov(rng)

    def reset(self) -> None:
        self.edge_resource = 0.75
        self.cloud_resource = 0.85
        self.network_quality = 0.70
        self.network_state = "GOOD"
