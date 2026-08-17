"""hhk边端预处理input到Phase4.5 Task的适配器。

只读取PPO决策前已存在的标准input，不读取边端模型output。任务类型、优先级、
截止时间和载荷来自input；缺失的模型性能字段使用task_profiles中的仿真先验。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from models import Task
from task_profiles import (
    PRIORITY_SCORE,
    VALID_TASK_TYPES,
    get_task_profile,
)


class EdgeInputFormatError(ValueError):
    """边端input文件格式错误。"""


@dataclass(frozen=True)
class EdgeInputLoadReport:
    input_count: int
    loaded_count: int
    skipped_count: int
    duplicate_task_id_count: int
    unknown_task_type_count: int
    default_deadline_count: int
    explicit_size_count: int
    estimated_size_count: int
    explicit_confidence_count: int
    profile_confidence_count: int
    timestamp_parsed_count: int


VALID_PRIORITIES = set(PRIORITY_SCORE)


def _read_json(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到边端input文件：{file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise EdgeInputFormatError(
            f"边端input不是合法JSON：{file_path}；{exc}"
        ) from exc


def _extract_inputs(raw_data: Any) -> list[dict[str, Any]]:
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]
    if isinstance(raw_data, dict):
        wrapped = raw_data.get("inputs")
        if isinstance(wrapped, list):
            return [item for item in wrapped if isinstance(item, dict)]
        if "task_id" in raw_data and "data" in raw_data:
            return [raw_data]
    raise EdgeInputFormatError(
        "input应为标准任务数组、单条标准任务，或包含inputs数组的对象。"
    )


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EdgeInputFormatError(f"字段{name}无法转换为数值：{value!r}") from exc
    if result <= 0:
        raise EdgeInputFormatError(f"字段{name}必须大于0：{result}")
    return result


def _unit_float(value: Any, name: str) -> float:
    result = _positive_float(value, name)
    if result > 1.0:
        raise EdgeInputFormatError(f"字段{name}必须位于(0, 1]：{result}")
    return result


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _estimate_payload_size_mb(
    edge_input: dict[str, Any],
    task_type: str,
) -> tuple[float, str]:
    data = edge_input.get("data")
    if not isinstance(data, dict):
        raise EdgeInputFormatError("字段data必须是JSON对象。")

    explicit = edge_input.get(
        "payload_size_mb",
        data.get("payload_size_mb"),
    )
    if explicit is not None:
        return _positive_float(explicit, "payload_size_mb"), "explicit"

    profile = get_task_profile(task_type)
    json_size_mb = len(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ) / (1024.0 * 1024.0)

    # 路径本身不代表真实媒体大小，因此用任务类型典型载荷作为下界。
    media_hint = any(
        key in data
        for key in ("image_url", "image_path", "video_url", "video_path")
    )
    profile_floor = profile.payload_typical_mb if media_hint else profile.payload_min_mb
    return max(json_size_mb, profile_floor), "profile_estimate"


def _deadline_s(
    edge_input: dict[str, Any],
    task_type: str,
) -> tuple[float, str]:
    raw = edge_input.get("max_latency_ms")
    if raw is not None:
        return _positive_float(raw, "max_latency_ms") / 1000.0, "explicit"
    return get_task_profile(task_type).default_deadline_s, "task_type_default"


def _confidence_prior(
    edge_input: dict[str, Any],
    task_type: str,
) -> tuple[float, str]:
    scheduling = edge_input.get("scheduling")
    if not isinstance(scheduling, dict):
        scheduling = {}
    raw = edge_input.get(
        "estimated_edge_confidence",
        scheduling.get("estimated_edge_confidence"),
    )
    if raw is not None:
        return _unit_float(raw, "estimated_edge_confidence"), "input_estimate"
    return get_task_profile(task_type).edge_confidence_typical, "task_type_prior"


def _performance_prior(
    edge_input: dict[str, Any],
    task_type: str,
) -> tuple[float, float, float, float, str]:
    profile = get_task_profile(task_type)
    scheduling = edge_input.get("scheduling")
    if not isinstance(scheduling, dict):
        scheduling = {}

    edge_ms = edge_input.get(
        "estimated_edge_latency_ms",
        scheduling.get("estimated_edge_latency_ms"),
    )
    cloud_ms = edge_input.get(
        "estimated_cloud_latency_ms",
        scheduling.get("estimated_cloud_latency_ms"),
    )
    cloud_accuracy = edge_input.get(
        "estimated_cloud_accuracy",
        scheduling.get("estimated_cloud_accuracy"),
    )
    complexity = edge_input.get(
        "complexity",
        scheduling.get("complexity"),
    )

    explicit = any(value is not None for value in (edge_ms, cloud_ms, cloud_accuracy, complexity))
    edge_latency_s = (
        profile.edge_base_latency_s
        if edge_ms is None
        else _positive_float(edge_ms, "estimated_edge_latency_ms") / 1000.0
    )
    cloud_latency_s = (
        profile.cloud_base_latency_s
        if cloud_ms is None
        else _positive_float(cloud_ms, "estimated_cloud_latency_ms") / 1000.0
    )
    cloud_acc = (
        profile.cloud_accuracy
        if cloud_accuracy is None
        else _unit_float(cloud_accuracy, "estimated_cloud_accuracy")
    )
    if complexity is None:
        task_complexity = profile.complexity
    else:
        try:
            task_complexity = float(complexity)
        except (TypeError, ValueError) as exc:
            raise EdgeInputFormatError("complexity无法转换为数值") from exc
        if not 0.0 <= task_complexity <= 1.0:
            raise EdgeInputFormatError("complexity必须位于[0, 1]")

    return (
        edge_latency_s,
        cloud_latency_s,
        cloud_acc,
        task_complexity,
        "input_estimate" if explicit else "task_type_prior",
    )




def _collaborative_prior(
    edge_input: dict[str, Any],
    task_type: str,
) -> tuple[float, float, float, float]:
    profile = get_task_profile(task_type)
    scheduling = edge_input.get("scheduling")
    if not isinstance(scheduling, dict):
        scheduling = {}

    edge_s = edge_input.get(
        "collaborative_edge_latency_s",
        scheduling.get("collaborative_edge_latency_s"),
    )
    cloud_s = edge_input.get(
        "collaborative_cloud_latency_s",
        scheduling.get("collaborative_cloud_latency_s"),
    )
    accuracy = edge_input.get(
        "collaborative_accuracy",
        scheduling.get("collaborative_accuracy"),
    )
    payload_ratio = edge_input.get(
        "collaborative_payload_ratio",
        scheduling.get("collaborative_payload_ratio"),
    )
    return (
        profile.collaborative_edge_latency_s
        if edge_s is None
        else _positive_float(edge_s, "collaborative_edge_latency_s"),
        profile.collaborative_cloud_latency_s
        if cloud_s is None
        else _positive_float(cloud_s, "collaborative_cloud_latency_s"),
        profile.collaborative_accuracy
        if accuracy is None
        else _unit_float(accuracy, "collaborative_accuracy"),
        profile.collaborative_payload_ratio
        if payload_ratio is None
        else _unit_float(payload_ratio, "collaborative_payload_ratio"),
    )


def load_edge_inputs(
    input_path: str | Path,
    *,
    strict: bool = True,
) -> tuple[list[Task], EdgeInputLoadReport]:
    inputs = _extract_inputs(_read_json(input_path))

    parsed_rows: list[tuple[dict[str, Any], datetime | None]] = []
    for item in inputs:
        parsed_rows.append((item, _parse_timestamp(item.get("timestamp"))))
    parsed_timestamps = [stamp for _, stamp in parsed_rows if stamp is not None]
    origin = min(parsed_timestamps) if parsed_timestamps else None

    tasks: list[Task] = []
    seen: set[str] = set()
    skipped = duplicates = unknown_types = 0
    default_deadlines = explicit_sizes = estimated_sizes = 0
    explicit_confidences = profile_confidences = parsed_count = 0

    for index, (item, parsed_timestamp) in enumerate(parsed_rows):
        try:
            external_id = str(item.get("task_id", "")).strip()
            if not external_id:
                raise EdgeInputFormatError("input缺少非空task_id")
            if external_id in seen:
                duplicates += 1
                raise EdgeInputFormatError(f"重复task_id：{external_id}")
            seen.add(external_id)

            node_id = str(item.get("node_id", "")).strip()
            if not node_id:
                raise EdgeInputFormatError(f"任务{external_id}缺少node_id")

            task_type = str(item.get("task_type", "generic_anomaly")).strip()
            if task_type not in VALID_TASK_TYPES:
                unknown_types += 1
                task_type = "generic_anomaly"

            profile = get_task_profile(task_type)
            priority = str(item.get("priority", profile.default_priority)).lower()
            if priority not in VALID_PRIORITIES:
                raise EdgeInputFormatError(
                    f"任务{external_id}的priority非法：{priority}"
                )

            data = item.get("data")
            if not isinstance(data, dict):
                raise EdgeInputFormatError(
                    f"任务{external_id}的data必须是JSON对象"
                )

            size_mb, size_source = _estimate_payload_size_mb(item, task_type)
            deadline_s, deadline_source = _deadline_s(item, task_type)
            confidence, confidence_source = _confidence_prior(item, task_type)
            (
                edge_latency_s,
                cloud_latency_s,
                cloud_accuracy,
                complexity,
                performance_source,
            ) = _performance_prior(item, task_type)
            (
                collaborative_edge_latency_s,
                collaborative_cloud_latency_s,
                collaborative_accuracy,
                collaborative_payload_ratio,
            ) = _collaborative_prior(item, task_type)

            if size_source == "explicit":
                explicit_sizes += 1
            else:
                estimated_sizes += 1
            if deadline_source != "explicit":
                default_deadlines += 1
            if confidence_source == "input_estimate":
                explicit_confidences += 1
            else:
                profile_confidences += 1

            arrival_offset_s: float | None = None
            if parsed_timestamp is not None and origin is not None:
                try:
                    arrival_offset_s = max(
                        0.0,
                        (parsed_timestamp - origin).total_seconds(),
                    )
                    parsed_count += 1
                except TypeError:
                    # 混合时区信息时退回文件顺序。
                    arrival_offset_s = None

            tasks.append(
                Task(
                    task_id=len(tasks) + 1,
                    arrival_time=0,
                    size_mb=size_mb,
                    deadline_s=deadline_s,
                    confidence=confidence,
                    external_task_id=external_id,
                    node_id=node_id,
                    source_timestamp=str(item.get("timestamp", "")) or None,
                    source_arrival_offset_s=arrival_offset_s,
                    task_type=task_type,
                    priority=priority,
                    device_id=(
                        None if data.get("device_id") is None else str(data.get("device_id"))
                    ),
                    device_type=(
                        None if data.get("device_type") is None else str(data.get("device_type"))
                    ),
                    complexity=complexity,
                    edge_base_latency_s=edge_latency_s,
                    cloud_base_latency_s=cloud_latency_s,
                    estimated_cloud_accuracy=cloud_accuracy,
                    collaborative_edge_latency_s=collaborative_edge_latency_s,
                    collaborative_cloud_latency_s=collaborative_cloud_latency_s,
                    collaborative_accuracy=collaborative_accuracy,
                    collaborative_payload_ratio=collaborative_payload_ratio,
                    edge_input=dict(item),
                    confidence_source=confidence_source,
                    size_source=size_source,
                    deadline_source=deadline_source,
                    performance_source=performance_source,
                    source="edge_input",
                )
            )
        except EdgeInputFormatError:
            skipped += 1
            if strict:
                raise

    if not tasks:
        raise EdgeInputFormatError("没有成功读取任何边端input任务")

    return tasks, EdgeInputLoadReport(
        input_count=len(inputs),
        loaded_count=len(tasks),
        skipped_count=skipped,
        duplicate_task_id_count=duplicates,
        unknown_task_type_count=unknown_types,
        default_deadline_count=default_deadlines,
        explicit_size_count=explicit_sizes,
        estimated_size_count=estimated_sizes,
        explicit_confidence_count=explicit_confidences,
        profile_confidence_count=profile_confidences,
        timestamp_parsed_count=parsed_count,
    )
