"""Versioned experiment scenarios for multi-condition Scheduler evaluation.

The legacy simulator defaults remain unchanged.  This module builds fresh,
replaceable arrival/resource/network/service models from explicit JSON files so
that experimental conditions never silently rewrite the canonical profiles.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from dynamics import (
    LegacyNetworkDynamics,
    LegacyResourceDynamics,
    ProfileServiceTimeModel,
    ServiceTimeModel,
    TaskArrivalModel,
)
from models import Task
from post_edge_policy import (
    PostEdgeDecision,
    PostEdgePolicyResult,
)
from system_state import SystemState


SCENARIO_SCHEMA_VERSION = "scheduler-experiment-scenario/1.0"
DEFAULT_SCENARIO_DIR = Path(__file__).resolve().parent / "config" / "scenarios"
DEFAULT_SCENARIO_FILES = (
    DEFAULT_SCENARIO_DIR / "normal_load.json",
    DEFAULT_SCENARIO_DIR / "edge_burst.json",
    DEFAULT_SCENARIO_DIR / "weak_network.json",
)


def _number(mapping: Mapping[str, Any], key: str) -> float:
    try:
        return float(mapping[key])
    except KeyError as exc:
        raise ValueError(f"场景配置缺少字段：{key}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"场景字段{key}必须是数值") from exc


def _unit(mapping: Mapping[str, Any], key: str) -> float:
    value = _number(mapping, key)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"场景字段{key}必须位于[0,1]")
    return value


@dataclass(frozen=True)
class ArrivalSegment:
    start_s: float
    end_s: float
    probability: float


@dataclass(frozen=True)
class ExperimentScenario:
    scenario_id: str
    description: str
    task_source: str
    training_weight: float
    slot_duration_s: float
    arrival_period_s: float
    arrival_segments: tuple[ArrivalSegment, ...]
    edge_initial_resource: float
    cloud_initial_resource: float
    network_initial_quality: float
    network_initial_state: str
    network_good_quality: float
    network_bad_quality: float
    network_good_to_bad_per_s: float
    network_bad_to_good_per_s: float
    min_bandwidth_mbps: float
    max_bandwidth_mbps: float
    network_noise_std_per_s: float
    edge_service_scale: float
    cloud_service_scale: float
    source_path: str

    @property
    def nominal_arrival_probability(self) -> float:
        weighted = sum(
            (segment.end_s - segment.start_s) * segment.probability
            for segment in self.arrival_segments
        )
        return float(weighted / self.arrival_period_s)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "description": self.description,
            "task_source": self.task_source,
            "training_weight": self.training_weight,
            "slot_duration_s": self.slot_duration_s,
            "nominal_arrival_probability": self.nominal_arrival_probability,
            "arrival_period_s": self.arrival_period_s,
            "arrival_segments": [
                {
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                    "probability": segment.probability,
                }
                for segment in self.arrival_segments
            ],
            "resource": {
                "edge_initial": self.edge_initial_resource,
                "cloud_initial": self.cloud_initial_resource,
            },
            "network": {
                "initial_quality": self.network_initial_quality,
                "initial_state": self.network_initial_state,
                "good_quality": self.network_good_quality,
                "bad_quality": self.network_bad_quality,
                "good_to_bad_per_s": self.network_good_to_bad_per_s,
                "bad_to_good_per_s": self.network_bad_to_good_per_s,
                "min_bandwidth_mbps": self.min_bandwidth_mbps,
                "max_bandwidth_mbps": self.max_bandwidth_mbps,
                "noise_std_per_s": self.network_noise_std_per_s,
            },
            "service_time": {
                "edge_scale": self.edge_service_scale,
                "cloud_scale": self.cloud_service_scale,
            },
            "source_path": self.source_path,
        }


def load_experiment_scenario(path: str | Path) -> ExperimentScenario:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("场景配置必须是JSON对象")
    if payload.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ValueError(
            f"不支持的场景schema：{payload.get('schema_version')!r}"
        )

    scenario_id = str(payload.get("scenario_id", "")).strip()
    if not scenario_id:
        raise ValueError("场景配置缺少scenario_id")
    description = str(payload.get("description", "")).strip()
    task_source = str(payload.get("task_source", "typed_simulation")).strip()
    if task_source != "typed_simulation":
        raise ValueError("当前实验场景只支持task_source=typed_simulation")
    training_weight = _number(payload, "training_weight")
    if training_weight <= 0.0:
        raise ValueError("training_weight必须大于0")

    arrival = payload.get("arrival")
    if not isinstance(arrival, dict):
        raise ValueError("场景配置缺少arrival对象")
    if arrival.get("kind") != "periodic_bernoulli":
        raise ValueError("arrival.kind必须是periodic_bernoulli")
    slot_duration_s = _number(arrival, "slot_duration_s")
    period_s = _number(arrival, "period_s")
    if slot_duration_s <= 0.0 or period_s <= 0.0:
        raise ValueError("arrival时间参数必须大于0")
    raw_segments = arrival.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("arrival.segments必须是非空数组")
    segments: list[ArrivalSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ValueError("arrival segment必须是JSON对象")
        start = _number(raw, "start_s")
        end = _number(raw, "end_s")
        probability = _unit(raw, "probability")
        if start < 0.0 or end <= start:
            raise ValueError("arrival segment时间区间非法")
        segments.append(ArrivalSegment(start, end, probability))
    segments.sort(key=lambda row: row.start_s)
    cursor = 0.0
    for segment in segments:
        if abs(segment.start_s - cursor) > 1e-9:
            raise ValueError("arrival segments必须从0开始连续覆盖period_s")
        cursor = segment.end_s
    if abs(cursor - period_s) > 1e-9:
        raise ValueError("arrival segments必须完整覆盖period_s")
    if not any(segment.probability > 0.0 for segment in segments):
        raise ValueError("arrival至少一个segment的概率必须大于0")

    resource = payload.get("resource")
    network = payload.get("network")
    service = payload.get("service_time")
    if not isinstance(resource, dict) or not isinstance(network, dict):
        raise ValueError("场景配置缺少resource或network对象")
    if not isinstance(service, dict):
        raise ValueError("场景配置缺少service_time对象")

    initial_state = str(network.get("initial_state", "GOOD")).upper()
    if initial_state not in {"GOOD", "BAD"}:
        raise ValueError("network.initial_state必须是GOOD或BAD")
    min_bandwidth = _number(network, "min_bandwidth_mbps")
    max_bandwidth = _number(network, "max_bandwidth_mbps")
    if min_bandwidth <= 0.0 or max_bandwidth <= min_bandwidth:
        raise ValueError("network带宽上下界非法")
    good_to_bad = _number(network, "good_to_bad_per_s")
    bad_to_good = _number(network, "bad_to_good_per_s")
    noise = _number(network, "noise_std_per_s")
    if not 0.0 <= good_to_bad < 1.0 or not 0.0 <= bad_to_good < 1.0:
        raise ValueError("network转移概率必须位于[0,1)")
    if noise < 0.0:
        raise ValueError("network.noise_std_per_s不能为负数")
    edge_scale = _number(service, "edge_scale")
    cloud_scale = _number(service, "cloud_scale")
    if edge_scale <= 0.0 or cloud_scale <= 0.0:
        raise ValueError("service_time倍率必须大于0")

    return ExperimentScenario(
        scenario_id=scenario_id,
        description=description,
        task_source=task_source,
        training_weight=training_weight,
        slot_duration_s=slot_duration_s,
        arrival_period_s=period_s,
        arrival_segments=tuple(segments),
        edge_initial_resource=_unit(resource, "edge_initial"),
        cloud_initial_resource=_unit(resource, "cloud_initial"),
        network_initial_quality=_unit(network, "initial_quality"),
        network_initial_state=initial_state,
        network_good_quality=_unit(network, "good_quality"),
        network_bad_quality=_unit(network, "bad_quality"),
        network_good_to_bad_per_s=good_to_bad,
        network_bad_to_good_per_s=bad_to_good,
        min_bandwidth_mbps=min_bandwidth,
        max_bandwidth_mbps=max_bandwidth,
        network_noise_std_per_s=noise,
        edge_service_scale=edge_scale,
        cloud_service_scale=cloud_scale,
        source_path=str(resolved),
    )


def load_experiment_scenarios(
    paths: Iterable[str | Path] = DEFAULT_SCENARIO_FILES,
) -> tuple[ExperimentScenario, ...]:
    scenarios = tuple(load_experiment_scenario(path) for path in paths)
    if not scenarios:
        raise ValueError("至少需要一个实验场景")
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("实验场景scenario_id不能重复")
    return scenarios


def select_training_scenario(
    scenarios: tuple[ExperimentScenario, ...],
    *,
    seed: int,
) -> ExperimentScenario:
    weights = np.asarray([scenario.training_weight for scenario in scenarios], dtype=np.float64)
    weights /= weights.sum()
    rng = np.random.default_rng(int(seed))
    return scenarios[int(rng.choice(len(scenarios), p=weights))]


class ScenarioArrivalModel(TaskArrivalModel):
    """Periodic Bernoulli arrival clock with time-varying segment rates."""

    def __init__(self, scenario: ExperimentScenario, *, seed: int) -> None:
        self.scenario = scenario
        self.seed = int(seed)
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 101]))
        self._next: float | None = None

    def _probability_at(self, time_s: float) -> float:
        position = float(time_s) % self.scenario.arrival_period_s
        for segment in self.scenario.arrival_segments:
            if segment.start_s <= position < segment.end_s:
                return segment.probability
        return self.scenario.arrival_segments[-1].probability

    def _sample(self, rng: np.random.Generator, now_s: float, *, initial: bool) -> float:
        candidate = float(now_s) if initial else float(now_s) + self.scenario.slot_duration_s
        while True:
            if rng.random() < self._probability_at(candidate):
                return candidate
            candidate += self.scenario.slot_duration_s

    def reset(self, *, rng: np.random.Generator, now_s: float = 0.0) -> None:
        del rng
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 101]))
        self._next = self._sample(self._rng, now_s, initial=True)

    def next_arrival_time_s(self) -> float | None:
        return self._next

    def consume_arrival(self, *, rng: np.random.Generator, now_s: float) -> None:
        del rng
        self._next = self._sample(self._rng, now_s, initial=False)


class ScenarioResourceDynamics(LegacyResourceDynamics):
    """Legacy consume/recover dynamics with scenario-specific initial state."""

    def __init__(self, scenario: ExperimentScenario, *, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 202]))
        self._scenario_edge_initial = scenario.edge_initial_resource
        self._scenario_cloud_initial = scenario.cloud_initial_resource
        super().__init__("consume_recover")

    def reset(self) -> None:
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 202]))
        self.edge_resource = float(self._scenario_edge_initial)
        self.cloud_resource = float(self._scenario_cloud_initial)

    def advance(
        self,
        *,
        delta_t_s: float,
        edge_busy: bool,
        cloud_busy: bool,
        rng: np.random.Generator,
    ) -> None:
        del rng
        super().advance(
            delta_t_s=delta_t_s,
            edge_busy=edge_busy,
            cloud_busy=cloud_busy,
            rng=self._rng,
        )


class ScenarioNetworkDynamics(LegacyNetworkDynamics):
    """Markov network using explicit, versioned scenario parameters."""

    def __init__(self, scenario: ExperimentScenario, *, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 303]))
        self._scenario = scenario
        super().__init__("markov")
        self.min_bandwidth_mbps = scenario.min_bandwidth_mbps
        self.max_bandwidth_mbps = scenario.max_bandwidth_mbps
        self.network_noise_std_per_s = scenario.network_noise_std_per_s
        self.p_good_to_bad_per_s = scenario.network_good_to_bad_per_s
        self.p_bad_to_good_per_s = scenario.network_bad_to_good_per_s
        self.good_network_quality = scenario.network_good_quality
        self.bad_network_quality = scenario.network_bad_quality
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, 303]))
        self.network_quality = float(self._scenario.network_initial_quality)
        self.network_state = self._scenario.network_initial_state

    def advance(self, *, delta_t_s: float, rng: np.random.Generator) -> None:
        del rng
        super().advance(delta_t_s=delta_t_s, rng=self._rng)


class ScenarioServiceTimeModel(ServiceTimeModel):
    """Scale configured profile service times without editing task profiles."""

    def __init__(self, scenario: ExperimentScenario) -> None:
        self.edge_scale = scenario.edge_service_scale
        self.cloud_scale = scenario.cloud_service_scale
        self._base = ProfileServiceTimeModel()

    def estimate(self, task: Task, state: SystemState) -> dict[str, float]:
        scaled = copy(task)
        scaled.edge_base_latency_s = float(task.edge_base_latency_s) * self.edge_scale
        scaled.cloud_base_latency_s = float(task.cloud_base_latency_s) * self.cloud_scale
        scaled.collaborative_cloud_latency_s = (
            float(task.collaborative_cloud_latency_s) * self.cloud_scale
        )
        return self._base.estimate(scaled, state)


def scenario_simulator_kwargs(
    scenario: ExperimentScenario,
    *,
    seed: int,
) -> dict[str, Any]:
    """Return fresh pluggable models for one simulator instance."""

    return {
        "arrival_probability": scenario.nominal_arrival_probability,
        "slot_duration": scenario.slot_duration_s,
        "task_source": scenario.task_source,
        "arrival_model": ScenarioArrivalModel(scenario, seed=seed),
        "resource_dynamics": ScenarioResourceDynamics(scenario, seed=seed),
        "network_dynamics": ScenarioNetworkDynamics(scenario, seed=seed),
        "service_time_model": ScenarioServiceTimeModel(scenario),
    }


class AlwaysAcceptEdgePolicy:
    """Experiment-only Post-edge ablation that never requests cloud review."""

    rule_version = "experiment-always-accept-edge/1.0"
    schema_version = "experiment-post-edge-ablation/1.0"

    def evaluate(
        self,
        *,
        task: Any,
        edge_result: Mapping[str, Any],
        now_s: float,
        cloud_review_estimated_s: float,
    ) -> PostEdgePolicyResult:
        del task, now_s, cloud_review_estimated_s
        try:
            confidence = float(edge_result.get("confidence"))
        except (TypeError, ValueError):
            confidence = None
        try:
            failure_probability = float(edge_result.get("failure_probability"))
        except (TypeError, ValueError):
            failure_probability = None
        return PostEdgePolicyResult(
            decision=PostEdgeDecision.ACCEPT_EDGE,
            reason="EXPERIMENT_POST_EDGE_BYPASS",
            rule_id="always_accept_edge",
            rule_version=self.rule_version,
            raw_confidence=confidence,
            confidence_threshold=0.0,
            cloud_review_feasible=False,
            review_recommended=False,
            failure_probability=failure_probability,
            edge_review_state=(
                None
                if edge_result.get("review_state") is None
                else str(edge_result.get("review_state"))
            ),
        )
