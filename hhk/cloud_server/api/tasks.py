"""`GET /api/v1/edge/tasks` — Edge polls for pending tasks."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from storage.task_queue import task_queue_manager
from storage.device_state import device_state
from storage.models import TaskItem

logger = logging.getLogger("cloud_server.api.tasks")

router = APIRouter(tags=["tasks"])


@router.get("/api/v1/edge/tasks", response_model=list[TaskItem])
async def get_tasks(device_id: str = Query(..., description="Edge device identifier")):
    """Return (and drain) all pending tasks for the given device.

    Edge devices poll this endpoint every ~2 seconds.  Tasks are generated
    by the :class:`CloudDecisionEngine` during heartbeat processing.

    Returns an empty list if no tasks are pending.
    """
    if not device_id:
        return []

    # Ensure the device has been seen recently
    if device_state.get(device_id) is None:
        logger.warning("Task poll from unknown device: %s", device_id)

    tasks = task_queue_manager.drain(device_id)

    logger.debug(
        "Dispatching %d tasks to device %s",
        len(tasks),
        device_id,
    )

    return [TaskItem(**t) for t in tasks]
