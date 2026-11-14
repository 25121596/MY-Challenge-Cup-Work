"""Per-device task queue manager (in-memory).

The cloud decision engine pushes tasks onto per-device queues.
Edges poll GET /api/v1/edge/tasks to drain their queue.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any


def _new_task_id() -> str:
    return f"task-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


class TaskQueueManager:
    """Thread-safe per-device task queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # device_id → list of TaskItem dicts
        self._queues: dict[str, list[dict[str, Any]]] = {}

    def enqueue(
        self,
        device_id: str,
        task_type: str,
        description: str = "",
        priority: int = 0,
        payload: dict[str, Any] | None = None,
        task_id: str | None = None,
        deadline_at_ms: int = 0,
        idempotency_key: str = "",
    ) -> str:
        """Push a task onto the device's queue. Returns the task_id."""
        if payload is None:
            payload = {}
        if task_id is None:
            task_id = _new_task_id()

        task = {
            "type": task_type,
            "task_id": task_id,
            "description": description,
            "priority": priority,
            "deadline_at_ms": deadline_at_ms,
            "enqueued_at_ms": int(time.time() * 1000),
            "idempotency_key": idempotency_key,
            "payload": payload,
        }
        with self._lock:
            self._queues.setdefault(device_id, []).append(task)
            # Priority first, then earliest deadline, then FIFO. A zero
            # deadline means best-effort and therefore sorts after deadlines.
            self._queues[device_id].sort(
                key=lambda t: (-t.get("priority", 0),
                               t.get("deadline_at_ms", 0) or float("inf"),
                               t.get("enqueued_at_ms", 0))
            )
        return task_id

    def drain(self, device_id: str) -> list[dict[str, Any]]:
        """Return (and remove) all pending tasks for *device_id*."""
        with self._lock:
            now_ms = int(time.time() * 1000)
            tasks = [task for task in self._queues.pop(device_id, [])
                     if not task.get("deadline_at_ms") or task["deadline_at_ms"] > now_ms]
            return list(tasks)

    def peek(self, device_id: str) -> list[dict[str, Any]]:
        """Return pending tasks without removing them."""
        with self._lock:
            return list(self._queues.get(device_id, []))

    def clear(self, device_id: str) -> None:
        with self._lock:
            self._queues.pop(device_id, None)

    @property
    def total_pending(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._queues.values())


# Singleton
task_queue_manager = TaskQueueManager()
