from __future__ import annotations

from collections import defaultdict
import threading
from typing import Any


class ServiceTimeEstimator:
    """EWMA service-time estimator seeded by legacy task-profile priors.

    Real measurements improve queue workload estimates while the initial values
    remain explicitly profile-based rather than measured.
    """

    def __init__(self, alpha: float = 0.25) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha必须位于(0, 1]")
        self.alpha = float(alpha)
        self._values: dict[tuple[str, str], float] = {}
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._lock = threading.Lock()

    @staticmethod
    def _prior(task: Any, stage: str) -> float:
        if stage == "edge":
            return max(0.001, float(getattr(task, "edge_base_latency_s", 0.15)))
        if stage == "cloud":
            return max(0.001, float(getattr(task, "cloud_base_latency_s", 0.15)))
        if stage == "collab_edge":
            return max(
                0.001,
                float(getattr(task, "collaborative_edge_latency_s", 0.08)),
            )
        if stage == "collab_cloud":
            return max(
                0.001,
                float(getattr(task, "collaborative_cloud_latency_s", 0.10)),
            )
        raise ValueError(f"未知service stage: {stage}")

    def estimate(self, task: Any, stage: str) -> float:
        key = (str(getattr(task, "task_type", "generic_anomaly")), stage)
        with self._lock:
            value = self._values.get(key)
        return self._prior(task, stage) if value is None else float(value)

    def has_measurement(self, task: Any, stage: str) -> bool:
        key = (str(getattr(task, "task_type", "generic_anomaly")), stage)
        with self._lock:
            return key in self._values

    def update(self, task: Any, stage: str, measured_s: float) -> float:
        measured = max(0.001, float(measured_s))
        key = (str(getattr(task, "task_type", "generic_anomaly")), stage)
        with self._lock:
            previous = self._values.get(key, self._prior(task, stage))
            updated = self.alpha * measured + (1.0 - self.alpha) * previous
            self._values[key] = updated
            self._counts[key] += 1
            return float(updated)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            rows = {}
            for (task_type, stage), value in sorted(self._values.items()):
                rows[f"{task_type}:{stage}"] = {
                    "ewma_s": round(float(value), 6),
                    "samples": int(self._counts[(task_type, stage)]),
                }
            return rows
