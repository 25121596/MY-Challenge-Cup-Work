"""Read-only adapter for replaying HHK V2 AI4I edge results.

The adapter joins the standard input file and the edge result file by task_id.
It copies only matched records into Scheduler Task objects.  HHK files remain an
external, read-only data source; no HHK Python modules or internal paths are
imported here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from edge_input_adapter import EdgeInputFormatError, EdgeInputLoadReport, load_edge_inputs
from models import Task


AI4I_MODEL_ROUTE = "ai4i_tabular_moe_v1"


@dataclass(frozen=True)
class HhkV2ReplayReport:
    input_count: int
    parsed_input_count: int
    result_count: int
    matched_count: int
    unmatched_input_count: int
    orphan_result_count: int
    duplicate_result_id_count: int
    missing_review_margin_count: int
    producer_rule_inconsistency_count: int
    join_coverage: float
    arrival_timing_source: str
    replay_interval_s: float
    feature_schema_ids: tuple[str, ...]
    model_routes: tuple[str, ...]
    model_versions: tuple[str, ...]
    calibration_versions: tuple[str, ...]
    preprocessing_versions: tuple[str, ...]
    input_sha256: str
    result_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    result_path = Path(path)
    if not result_path.is_file():
        raise FileNotFoundError(f"找不到HHK V2边端结果文件：{result_path}")
    try:
        with result_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise EdgeInputFormatError(f"HHK V2边端结果不是合法JSON：{result_path}；{exc}") from exc
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise EdgeInputFormatError("HHK V2结果应为数组或包含results数组的对象")
    return [row for row in rows if isinstance(row, dict)]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probability(value: Any, *, task_id: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EdgeInputFormatError(f"任务{task_id}的failure_probability无法转换为数值") from exc
    if not 0.0 <= result <= 1.0:
        raise EdgeInputFormatError(f"任务{task_id}的failure_probability必须位于[0,1]")
    return result


def _evidence(row: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    task_id = str(row.get("task_id", "")).strip()
    if not task_id:
        raise EdgeInputFormatError("HHK V2结果缺少非空task_id")
    model_route = str(row.get("model_route", "")).strip()
    feature_schema = str(row.get("feature_schema_id", "")).strip()
    probability = _probability(row.get("failure_probability"), task_id=task_id)
    review_state = str(row.get("review_state", "")).strip().lower()
    if review_state not in {"required", "not_required"}:
        raise EdgeInputFormatError(f"任务{task_id}的review_state非法：{review_state!r}")
    if model_route == AI4I_MODEL_ROUTE and probability is None:
        raise EdgeInputFormatError(f"AI4I任务{task_id}缺少failure_probability")
    if model_route == AI4I_MODEL_ROUTE and feature_schema != "ai4i_8d_v1":
        raise EdgeInputFormatError(f"AI4I任务{task_id}的feature_schema_id不兼容：{feature_schema!r}")
    try:
        latency_s = max(0.0, float(row.get("latency_ms"))) / 1000.0
    except (TypeError, ValueError) as exc:
        raise EdgeInputFormatError(f"任务{task_id}缺少合法latency_ms") from exc
    threshold = _probability(row.get("threshold_used"), task_id=task_id)
    margin_missing = row.get("threshold_review_margin") is None
    margin = 0.0 if margin_missing else float(row["threshold_review_margin"])
    if margin < 0.0:
        raise EdgeInputFormatError(f"任务{task_id}的threshold_review_margin不能为负数")
    safety_trigger = bool(row.get("safety_trigger", False))
    predicted_review = (
        safety_trigger
        or probability is None
        or threshold is None
        or probability > threshold
        or abs(probability - threshold) <= margin
    )
    producer_consistent = predicted_review == (review_state == "required")
    return ({
        "source": "hhk_v2_edge_result_replay",
        "task_id": task_id,
        "output_valid": True,
        "model_route": model_route,
        "model_version": str(row.get("model_version", "")),
        "calibration_version": str(row.get("calibration_version", "")),
        "feature_schema_id": feature_schema,
        "failure_probability": probability,
        "local_prediction": row.get("local_prediction"),
        "threshold_used": threshold,
        "threshold_review_margin": margin,
        "review_state": review_state,
        "review_reason": row.get("review_reason"),
        "safety_priority": row.get("safety_priority"),
        "safety_trigger": safety_trigger,
        "producer_action": row.get("action"),
        "producer_route": row.get("route"),
        "producer_policy_version": row.get("policy_version"),
        "producer_rule_consistent": producer_consistent,
        "latency_s": latency_s,
    }, margin_missing, producer_consistent)


def load_hhk_v2_replay(
    input_path: str | Path,
    result_path: str | Path,
    *,
    strict: bool = True,
    replay_interval_s: float = 0.05,
    require_full_coverage: bool = False,
) -> tuple[list[Task], EdgeInputLoadReport, HhkV2ReplayReport]:
    """Join HHK V2 input/result files and return replayable matched tasks."""
    if replay_interval_s <= 0:
        raise ValueError("replay_interval_s必须大于0")
    tasks, input_report = load_edge_inputs(input_path, strict=strict)
    rows = _load_result_rows(result_path)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates = missing_margin = inconsistencies = 0
    for row in rows:
        evidence, margin_missing, consistent = _evidence(row)
        task_id = evidence["task_id"]
        if task_id in by_id:
            duplicates += 1
            if strict:
                raise EdgeInputFormatError(f"HHK V2结果包含重复task_id：{task_id}")
            continue
        by_id[task_id] = evidence
        missing_margin += int(margin_missing)
        inconsistencies += int(not consistent)

    input_ids = {str(task.external_task_id) for task in tasks}
    matched: list[Task] = []
    for task in tasks:
        evidence = by_id.get(str(task.external_task_id))
        if evidence is None:
            continue
        task.edge_result_evidence = dict(evidence)
        task.edge_result_source = "hhk_v2_edge_result_replay"
        task.edge_latency_source = "hhk_v2_measured_result_latency"
        task.edge_base_latency_s = max(float(evidence["latency_s"]), 1e-6)
        task.source = "hhk_v2_replay"
        matched.append(task)

    unmatched = len(tasks) - len(matched)
    orphans = len(set(by_id) - input_ids)
    if require_full_coverage and (unmatched or orphans):
        raise EdgeInputFormatError(
            f"HHK V2输入/结果未完全匹配：unmatched_inputs={unmatched}, orphan_results={orphans}"
        )
    if not matched:
        raise EdgeInputFormatError("HHK V2输入与结果没有可回放的task_id交集")

    offsets = [task.source_arrival_offset_s for task in matched]
    usable_timing = len(offsets) <= 1 or all(
        offsets[index] is not None
        and offsets[index - 1] is not None
        and float(offsets[index]) > float(offsets[index - 1])
        for index in range(1, len(offsets))
    )
    timing_source = "input_timestamps" if usable_timing else "configured_replay_interval"
    if not usable_timing:
        for index, task in enumerate(matched):
            task.source_arrival_offset_s = index * replay_interval_s

    def versions(key: str) -> tuple[str, ...]:
        return tuple(sorted({str(item.get(key, "")) for item in by_id.values() if item.get(key)}))

    report = HhkV2ReplayReport(
        input_count=input_report.input_count,
        parsed_input_count=input_report.loaded_count,
        result_count=len(rows),
        matched_count=len(matched),
        unmatched_input_count=unmatched,
        orphan_result_count=orphans,
        duplicate_result_id_count=duplicates,
        missing_review_margin_count=missing_margin,
        producer_rule_inconsistency_count=inconsistencies,
        join_coverage=len(matched) / max(input_report.loaded_count, 1),
        arrival_timing_source=timing_source,
        replay_interval_s=float(replay_interval_s),
        feature_schema_ids=versions("feature_schema_id"),
        model_routes=versions("model_route"),
        model_versions=versions("model_version"),
        calibration_versions=versions("calibration_version"),
        preprocessing_versions=tuple(sorted({
            str((task.edge_input.get("data") or {}).get("preprocessing_version"))
            for task in matched
            if (task.edge_input.get("data") or {}).get("preprocessing_version")
        })),
        input_sha256=_sha256(input_path),
        result_sha256=_sha256(result_path),
    )
    return matched, input_report, report
