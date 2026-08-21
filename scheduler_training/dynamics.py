"""Pluggable dynamics for Phase4.5.

The default Legacy* implementations preserve the numeric parameters and formulas
from Phase4 v1.8.5 while exposing real-ready interfaces.  No parameter
calibration is performed in Phase4.5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import math
from typing import Callable

import numpy as np

from models import Task
from system_state import SystemState


class TaskArrivalModel(ABC):
    @abstractmethod
    def reset(self, *, rng: np.random.Generator, now_s: float = 0.0) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_arrival_time_s(self) -> float | None:
        raise NotImplementedError

    @abstractmethod
    def consume_arrival(self, *, rng: np.random.Generator, now_s: float) -> None:
        raise NotImplementedError


class LegacyTaskArrivalModel(TaskArrivalModel):
    """Continuous-time event clock preserving Phase4 Bernoulli-per-slot arrivals.

    For simulated sources, geometric sampling is exactly the distribution of the
    number of 50ms Phase4 slots until the next Bernoulli success.  For edge-input
    mode, timestamps/intervals are consumed directly.
    """

    def __init__(
        self,
        *,
        arrival_probability: float,
        slot_duration: float,
        task_source: str,
        external_offsets_s: list[float] | None = None,
        edge_input_interval_s: float = 0.05,
    ) -> None:
        self.arrival_probability = float(arrival_probability)
        self.slot_duration = float(slot_duration)
        self.task_source = str(task_source)
        self.external_offsets_s = list(external_offsets_s or [])
        self.edge_input_interval_s = float(edge_input_interval_s)
        self._external_index = 0
        self._next: float | None = None

    def _sample_initial(self, rng: np.random.Generator) -> float | None:
        if self.arrival_probability <= 0.0:
            return None
        # Original simulator checks an arrival at slot 0, hence geometric-1.
        slots = int(rng.geometric(self.arrival_probability)) - 1
        return float(slots * self.slot_duration)

    def _sample_following(self, rng: np.random.Generator, now_s: float) -> float | None:
        if self.arrival_probability <= 0.0:
            return None
        slots = int(rng.geometric(self.arrival_probability))
        return float(now_s + slots * self.slot_duration)

    def _external_time(self, index: int) -> float | None:
        if index >= len(self.external_offsets_s):
            return None
        value = self.external_offsets_s[index]
        if value >= 0:
            return float(value)
        return float(index * self.edge_input_interval_s)

    def reset(self, *, rng: np.random.Generator, now_s: float = 0.0) -> None:
        self._external_index = 0
        if self.task_source in {"edge_input", "hhk_v2_replay"}:
            self._next = self._external_time(0)
        else:
            first = self._sample_initial(rng)
            self._next = None if first is None else now_s + first

    def next_arrival_time_s(self) -> float | None:
        return self._next

    def consume_arrival(self, *, rng: np.random.Generator, now_s: float) -> None:
        if self.task_source in {"edge_input", "hhk_v2_replay"}:
            self._external_index += 1
            self._next = self._external_time(self._external_index)
        else:
            self._next = self._sample_following(rng, now_s)


class ServiceTimeModel(ABC):
    @abstractmethod
    def estimate(self, task: Task, state: SystemState) -> dict[str, float]:
        raise NotImplementedError


class ProfileServiceTimeModel(ServiceTimeModel):
    """Legacy Phase4 scheduler formulas behind a replaceable service-time API."""

    def estimate(self, task: Task, state: SystemState) -> dict[str, float]:
        from scheduler import estimate_delays_for_task

        return estimate_delays_for_task(task, state)


class ResourceDynamics(ABC):
    edge_resource: float
    cloud_resource: float

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def advance(
        self,
        *,
        delta_t_s: float,
        edge_busy: bool,
        cloud_busy: bool,
        rng: np.random.Generator,
    ) -> None:
        raise NotImplementedError


class LegacyResourceDynamics(ResourceDynamics):
    """Phase4 resource parameters, expressed as continuous delta-time updates."""

    def __init__(self, mode: str = "consume_recover") -> None:
        if mode not in {"legacy", "consume_recover"}:
            raise ValueError(f"非法resource_mode：{mode}")
        self.mode = mode

        # Copied unchanged from Phase4 v1.8.5/environment.py.
        self.edge_idle_target = 0.85
        self.edge_busy_target = 0.55
        self.cloud_idle_target = 0.90
        self.cloud_busy_target = 0.65
        self.mean_reversion_speed_per_s = 0.15
        self.edge_noise_std_per_s = 0.015
        self.cloud_noise_std_per_s = 0.010
        self.edge_resource_min = 0.20
        self.edge_resource_max = 1.00
        self.cloud_resource_min = 0.30
        self.cloud_resource_max = 1.00
        self.edge_recovery_per_s = 0.045
        self.cloud_recovery_per_s = 0.0375
        self.edge_consumption_per_s = 0.090
        self.cloud_consumption_per_s = 0.075
        self.reset()

    def reset(self) -> None:
        self.edge_resource = 0.75
        self.cloud_resource = 0.85

    @staticmethod
    def _noise(std_per_s: float, delta_t_s: float, rng: np.random.Generator) -> float:
        return float(rng.normal(0.0, std_per_s * math.sqrt(max(delta_t_s, 0.0))))

    def advance(
        self,
        *,
        delta_t_s: float,
        edge_busy: bool,
        cloud_busy: bool,
        rng: np.random.Generator,
    ) -> None:
        dt = max(0.0, float(delta_t_s))
        if dt == 0.0:
            return
        if self.mode == "legacy":
            alpha = 1.0 - math.exp(-self.mean_reversion_speed_per_s * dt)
            edge_target = self.edge_busy_target if edge_busy else self.edge_idle_target
            cloud_target = self.cloud_busy_target if cloud_busy else self.cloud_idle_target
            self.edge_resource = float(np.clip(
                self.edge_resource
                + alpha * (edge_target - self.edge_resource)
                + self._noise(self.edge_noise_std_per_s, dt, rng),
                self.edge_resource_min,
                self.edge_resource_max,
            ))
            self.cloud_resource = float(np.clip(
                self.cloud_resource
                + alpha * (cloud_target - self.cloud_resource)
                + self._noise(self.cloud_noise_std_per_s, dt, rng),
                self.cloud_resource_min,
                self.cloud_resource_max,
            ))
            return

        edge_rate = -self.edge_consumption_per_s if edge_busy else self.edge_recovery_per_s
        cloud_rate = -self.cloud_consumption_per_s if cloud_busy else self.cloud_recovery_per_s
        self.edge_resource = float(np.clip(
            self.edge_resource + edge_rate * dt + self._noise(self.edge_noise_std_per_s, dt, rng),
            self.edge_resource_min,
            self.edge_resource_max,
        ))
        self.cloud_resource = float(np.clip(
            self.cloud_resource + cloud_rate * dt + self._noise(self.cloud_noise_std_per_s, dt, rng),
            self.cloud_resource_min,
            self.cloud_resource_max,
        ))


class NetworkDynamics(ABC):
    network_quality: float
    network_state: str

    @property
    @abstractmethod
    def bandwidth_mbps(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def advance(self, *, delta_t_s: float, rng: np.random.Generator) -> None:
        raise NotImplementedError


class LegacyNetworkDynamics(NetworkDynamics):
    """Phase4 network parameters behind a continuous-time adapter."""

    def __init__(self, mode: str = "markov") -> None:
        if mode not in {"legacy", "markov"}:
            raise ValueError(f"非法network_mode：{mode}")
        self.mode = mode

        # Copied unchanged from Phase4 v1.8.5/environment.py.
        self.min_bandwidth_mbps = 5.0
        self.max_bandwidth_mbps = 50.0
        self.network_noise_std_per_s = 0.07
        self.network_quality_min = 0.00
        self.network_quality_max = 1.00
        self.p_good_to_bad_per_s = 0.10
        self.p_bad_to_good_per_s = 0.30
        self.good_network_quality = 1.00
        self.bad_network_quality = 0.20
        self.reset()

    def reset(self) -> None:
        self.network_quality = 0.70
        self.network_state = "GOOD"

    @property
    def bandwidth_mbps(self) -> float:
        return float(self.min_bandwidth_mbps + self.network_quality * (
            self.max_bandwidth_mbps - self.min_bandwidth_mbps
        ))

    def advance(self, *, delta_t_s: float, rng: np.random.Generator) -> None:
        dt = max(0.0, float(delta_t_s))
        if dt == 0.0:
            return
        if self.mode == "legacy":
            self.network_quality = float(np.clip(
                self.network_quality
                + rng.normal(0.0, self.network_noise_std_per_s * math.sqrt(dt)),
                self.network_quality_min,
                self.network_quality_max,
            ))
            self.network_state = "GOOD" if self.network_quality >= 0.5 else "BAD"
            return

        # Convert the unchanged one-second transition probabilities to CTMC
        # hazards. This removes slot stepping while preserving their meaning.
        a = -math.log(1.0 - self.p_good_to_bad_per_s)
        b = -math.log(1.0 - self.p_bad_to_good_per_s)
        total = a + b
        decay = math.exp(-total * dt)
        if self.network_state == "GOOD":
            p_bad = (a / total) * (1.0 - decay)
            if rng.random() < p_bad:
                self.network_state = "BAD"
        else:
            p_good = (b / total) * (1.0 - decay)
            if rng.random() < p_good:
                self.network_state = "GOOD"

        base = self.good_network_quality if self.network_state == "GOOD" else self.bad_network_quality
        self.network_quality = float(np.clip(
            base + rng.normal(0.0, 0.015),
            self.network_quality_min,
            self.network_quality_max,
        ))
