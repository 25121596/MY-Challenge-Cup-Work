from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import load_ground_truth, load_tasks, write_jsonl
from .schema import TeacherConfig
from .teacher_backends import (
    TeacherBackendError,
    build_teacher_backend,
    student_visible_view,
)


@dataclass(frozen=True)
class TeacherGenerationReport:
    input_count: int
    success_count: int
    failed_count: int
    valid_count: int
    output_path: str
    backend: str
    model: str


def generate_teacher_outputs(
    input_path: str | Path,
    output_path: str | Path,
    config: TeacherConfig,
    *,
    ground_truth_path: str | Path | None = None,
    limit: int | None = None,
    continue_on_error: bool = False,
) -> TeacherGenerationReport:
    tasks = load_tasks(input_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit必须大于0")
        tasks = tasks[:limit]
    truth_map = load_ground_truth(ground_truth_path)
    backend = build_teacher_backend(config)
    rows: list[dict[str, Any]] = []
    failed = 0
    for index, task in enumerate(tasks, start=1):
        task_id = str(task.get("task_id", "")).strip()
        if not task_id:
            raise ValueError(f"第{index}条任务缺少task_id")
        task_type = str(task.get("task_type", "generic_anomaly"))
        truth_record = truth_map.get(task_id, {})
        try:
            raw, validation = backend.generate(task)
            teacher_output = validation.parsed
            error = None
        except TeacherBackendError as exc:
            if not continue_on_error:
                raise
            failed += 1
            raw = ""
            teacher_output = None
            validation = None
            error = str(exc)
        rows.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "student_input": student_visible_view(task),
                "teacher_output": teacher_output,
                "teacher_raw_output": raw,
                "teacher_type": backend.name,
                "teacher_model": config.model if backend.name != "simulated_rule" else "simulated_rule",
                "teacher_valid": bool(validation and validation.valid),
                "teacher_validation_errors": [] if validation is None else list(validation.errors),
                "teacher_confidence": (
                    None if not teacher_output else teacher_output.get("confidence")
                ),
                "ground_truth": truth_record.get("ground_truth"),
                "label_source": truth_record.get(
                    "label_source", "teacher_pseudo_label"
                ),
                "generation_error": error,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        status = "ERROR" if error else "OK"
        print(f"[{index}/{len(tasks)}] {task_id} {task_type}: {status}")

    written = write_jsonl(output_path, rows)
    valid = sum(1 for row in rows if row["teacher_valid"])
    report = TeacherGenerationReport(
        input_count=len(tasks),
        success_count=written - failed,
        failed_count=failed,
        valid_count=valid,
        output_path=str(output_path),
        backend=backend.name,
        model=(config.model if backend.name != "simulated_rule" else "simulated_rule"),
    )
    return report
