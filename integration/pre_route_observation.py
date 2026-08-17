"""Integration-side implementation of pre-route observation schema 2.0.

The bundle is the deployment contract and supplies the ordered field names.
This module implements the normalization semantics pinned by
``pre-route-observation/2.0`` so real state can be converted without importing
the training simulator at runtime.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from task_profiles import PRIORITY_SCORE, TASK_TYPE_ORDER


OBSERVATION_SCHEMA_VERSION = "pre-route-observation/2.0"
BASE_OBSERVATION_NAMES = (
    "task_present",
    "task_size",
    "task_deadline",
    "task_urgency",
    "edge_queue",
    "cloud_queue",
    "edge_workload",
    "cloud_workload",
    "queue_difference",
    "edge_resource",
    "cloud_resource",
    "resource_difference",
    "network_quality",
    "bandwidth",
)
OBSERVATION_NAMES = (
    *BASE_OBSERVATION_NAMES,
    *(f"task_type_{task_type}" for task_type in TASK_TYPE_ORDER),
    "priority_score",
    "task_complexity",
)

ACTION_SCHEMA_VERSION = "pre-route-action/2.0"
ACTION_MAPPING = {"0": "LOCAL_FIRST", "1": "DIRECT_CLOUD"}

MAX_QUEUE_LENGTH = 30.0
MAX_WORKLOAD_S = 5.0
MIN_TASK_SIZE_MB = 0.005
MAX_TASK_SIZE_MB = 1.50
MIN_DEADLINE_S = 0.10
MAX_DEADLINE_S = 0.80
MIN_BANDWIDTH_MBPS = 5.0
MAX_BANDWIDTH_MBPS = 50.0


def _normalize(value: float, minimum: float, maximum: float) -> float:
    return float(np.clip((float(value) - minimum) / (maximum - minimum), 0.0, 1.0))


def validate_observation_names(names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(name) for name in names)
    if normalized != OBSERVATION_NAMES:
        raise ValueError(
            "bundle的24D observation字段或顺序与Integration支持的"
            f"{OBSERVATION_SCHEMA_VERSION}不一致"
        )
    if "task_confidence" in normalized:
        raise ValueError("pre-route observation不得包含task_confidence")
    return normalized


def build_pre_route_observation(
    task: Any | None,
    state: dict[str, Any],
    *,
    observation_names: Sequence[str] = OBSERVATION_NAMES,
) -> np.ndarray:
    """Build the exact Scheduler 24D pre-route vector from real state."""

    validate_observation_names(observation_names)
    type_one_hot = np.zeros(len(TASK_TYPE_ORDER), dtype=np.float32)

    if task is None:
        task_present = task_size = task_deadline = task_urgency = 0.0
        priority_score = task_complexity = 0.0
    else:
        task_present = 1.0
        task_size = _normalize(task.size_mb, MIN_TASK_SIZE_MB, MAX_TASK_SIZE_MB)
        task_deadline = _normalize(task.deadline_s, MIN_DEADLINE_S, MAX_DEADLINE_S)
        priority_score = PRIORITY_SCORE.get(str(task.priority), 0.5)
        minimum_workload = min(
            float(state["edge_workload_s"]),
            float(state["cloud_workload_s"]),
        )
        task_urgency = float(
            np.clip(
                max(
                    1.0 - task_deadline,
                    minimum_workload / max(float(task.deadline_s), 1e-6),
                    0.85 * priority_score,
                ),
                0.0,
                1.0,
            )
        )
        task_complexity = float(np.clip(float(task.complexity), 0.0, 1.0))
        task_type = str(getattr(task, "task_type", "generic_anomaly"))
        if task_type not in TASK_TYPE_ORDER:
            task_type = "generic_anomaly"
        type_one_hot[TASK_TYPE_ORDER.index(task_type)] = 1.0

    edge_queue = float(state["edge_queue"])
    cloud_queue = float(state["cloud_queue"])
    edge_resource = float(state["edge_resource"])
    cloud_resource = float(state["cloud_resource"])
    base = [
        task_present,
        task_size,
        task_deadline,
        task_urgency,
        np.clip(edge_queue / MAX_QUEUE_LENGTH, 0.0, 1.0),
        np.clip(cloud_queue / MAX_QUEUE_LENGTH, 0.0, 1.0),
        np.clip(float(state["edge_workload_s"]) / MAX_WORKLOAD_S, 0.0, 1.0),
        np.clip(float(state["cloud_workload_s"]) / MAX_WORKLOAD_S, 0.0, 1.0),
        np.clip((edge_queue - cloud_queue) / MAX_QUEUE_LENGTH, -1.0, 1.0),
        np.clip(edge_resource, 0.0, 1.0),
        np.clip(cloud_resource, 0.0, 1.0),
        np.clip(edge_resource - cloud_resource, -1.0, 1.0),
        np.clip(float(state["network_quality"]), 0.0, 1.0),
        _normalize(float(state["bandwidth_mbps"]), MIN_BANDWIDTH_MBPS, MAX_BANDWIDTH_MBPS),
    ]
    observation = np.asarray(
        [*base, *type_one_hot.tolist(), priority_score, task_complexity],
        dtype=np.float32,
    )
    if observation.shape != (24,):
        raise RuntimeError(f"真实pre-route observation不是24D: {observation.shape}")
    return observation
