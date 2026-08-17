"""Task lifecycle helpers aligned with Integration v0.3 semantics."""

from __future__ import annotations

from models import Action, Task, TaskStatus


EDGE_WAITING = "EDGE_WAITING"
EDGE_RUNNING = "EDGE_RUNNING"
EDGE_RESULT_READY = "EDGE_RESULT_READY"
CLOUD_WAITING = "CLOUD_WAITING"
CLOUD_RUNNING = "CLOUD_RUNNING"
TRANSMITTING = "TRANSMITTING"
CLOUD_REVIEW_WAITING = "CLOUD_REVIEW_WAITING"
CLOUD_REVIEW_RUNNING = "CLOUD_REVIEW_RUNNING"
FINISHED = "FINISHED"
TIMEOUT = "TIMEOUT"


def mark_waiting(task: Task, route: Action, stage: str) -> None:
    task.route = route
    task.status = TaskStatus.WAITING
    task.execution_stage = stage


def mark_running(task: Task, stage: str, now_s: float) -> None:
    task.status = TaskStatus.RUNNING
    task.execution_stage = stage
    if task.start_time is None:
        task.start_time = now_s


def mark_transmitting(task: Task, now_s: float) -> None:
    task.status = TaskStatus.TRANSMITTING
    task.execution_stage = TRANSMITTING
    task.transmission_start_time = now_s


def mark_edge_result_ready(task: Task, now_s: float) -> None:
    task.status = TaskStatus.RUNNING
    task.execution_stage = EDGE_RESULT_READY
    task.edge_stage_done = True
    task.edge_stage_finish_time = now_s


def mark_finished(task: Task, now_s: float) -> None:
    task.status = TaskStatus.COMPLETED
    task.execution_stage = FINISHED
    task.finish_time = now_s


def mark_timeout(task: Task, now_s: float) -> None:
    task.status = TaskStatus.TIMEOUT
    task.execution_stage = TIMEOUT
    task.finish_time = now_s
