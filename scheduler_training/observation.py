"""Phase4.5 pre-route PPO observation schema v2.

The edge inference confidence is intentionally excluded because it does not
exist before LOCAL_FIRST executes.  The schema contains 24 pre-route features.
"""

from __future__ import annotations

import numpy as np

from models import Task
from task_profiles import PRIORITY_SCORE, TASK_TYPE_ORDER


BASE_OBSERVATION_NAMES = [
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
]

TASK_TYPE_OBSERVATION_NAMES = [
    f"task_type_{task_type}" for task_type in TASK_TYPE_ORDER
]

OBSERVATION_NAMES = [
    *BASE_OBSERVATION_NAMES,
    *TASK_TYPE_OBSERVATION_NAMES,
    "priority_score",
    "task_complexity",
]

MAX_QUEUE_LENGTH = 30.0
MAX_WORKLOAD_S = 5.0
MIN_TASK_SIZE_MB = 0.005
MAX_TASK_SIZE_MB = 1.50
MIN_DEADLINE_S = 0.10
MAX_DEADLINE_S = 0.80
MIN_BANDWIDTH_MBPS = 5.0
MAX_BANDWIDTH_MBPS = 50.0


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        raise ValueError("maximum必须大于minimum")
    return float(
        np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0)
    )


def _calculate_urgency(
    task: Task,
    state,
    normalized_deadline: float,
) -> float:
    deadline_urgency = 1.0 - normalized_deadline
    minimum_workload = min(
        float(state["edge_workload_s"]),
        float(state["cloud_workload_s"]),
    )
    workload_urgency = minimum_workload / max(task.deadline_s, 1e-6)
    priority_urgency = PRIORITY_SCORE.get(task.priority, 0.5)
    return float(
        np.clip(
            max(
                deadline_urgency,
                workload_urgency,
                0.85 * priority_urgency,
            ),
            0.0,
            1.0,
        )
    )


def build_observation(task: Task | None, state) -> np.ndarray:
    type_one_hot = np.zeros(len(TASK_TYPE_ORDER), dtype=np.float32)

    if task is None:
        task_present = task_size = task_deadline = 0.0
        task_urgency = 0.0
        priority_score = task_complexity = 0.0
    else:
        task_present = 1.0
        task_size = _normalize(
            task.size_mb,
            MIN_TASK_SIZE_MB,
            MAX_TASK_SIZE_MB,
        )
        task_deadline = _normalize(
            task.deadline_s,
            MIN_DEADLINE_S,
            MAX_DEADLINE_S,
        )
        task_urgency = _calculate_urgency(task, state, task_deadline)
        priority_score = PRIORITY_SCORE.get(task.priority, 0.5)
        task_complexity = float(np.clip(task.complexity, 0.0, 1.0))
        try:
            type_one_hot[TASK_TYPE_ORDER.index(task.task_type)] = 1.0
        except ValueError:
            type_one_hot[TASK_TYPE_ORDER.index("generic_anomaly")] = 1.0

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
        np.clip(state["edge_workload_s"] / MAX_WORKLOAD_S, 0.0, 1.0),
        np.clip(state["cloud_workload_s"] / MAX_WORKLOAD_S, 0.0, 1.0),
        np.clip((edge_queue - cloud_queue) / MAX_QUEUE_LENGTH, -1.0, 1.0),
        np.clip(edge_resource, 0.0, 1.0),
        np.clip(cloud_resource, 0.0, 1.0),
        np.clip(edge_resource - cloud_resource, -1.0, 1.0),
        np.clip(state["network_quality"], 0.0, 1.0),
        _normalize(
            state["bandwidth_mbps"],
            MIN_BANDWIDTH_MBPS,
            MAX_BANDWIDTH_MBPS,
        ),
    ]

    return np.asarray(
        [*base, *type_one_hot.tolist(), priority_score, task_complexity],
        dtype=np.float32,
    )
