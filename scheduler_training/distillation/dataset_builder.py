from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

from .io_utils import read_jsonl, write_json, write_jsonl
from .output_validation import validate_output
from .schema import TASK_INSTRUCTIONS


@dataclass(frozen=True)
class DatasetBuildReport:
    input_count: int
    accepted_count: int
    rejected_count: int
    ground_truth_target_count: int
    teacher_target_count: int
    train_count: int
    test_count: int
    train_path: str
    test_path: str
    manifest_path: str


def _sample_from_record(
    record: dict[str, Any],
    *,
    target_policy: str,
    min_teacher_confidence: float,
) -> tuple[dict[str, Any] | None, str | None]:
    task_id = str(record.get("task_id", "")).strip()
    task_type = str(record.get("task_type", "generic_anomaly"))
    student_input = record.get("student_input")
    teacher_output = record.get("teacher_output")
    ground_truth = record.get("ground_truth")
    if not task_id:
        return None, "缺少task_id"
    if not isinstance(student_input, dict):
        return None, "缺少student_input对象"
    if not record.get("teacher_valid") or not validate_output(teacher_output or {}).valid:
        return None, "教师输出无效"
    confidence = record.get("teacher_confidence")
    if confidence is not None:
        try:
            if float(confidence) < min_teacher_confidence:
                return None, "教师置信度低于阈值"
        except (TypeError, ValueError):
            return None, "teacher_confidence不是数值"

    policy = target_policy.lower()
    if policy == "ground-truth-only":
        if not isinstance(ground_truth, dict):
            return None, "缺少ground_truth"
        target = ground_truth
        target_source = "ground_truth"
    elif policy == "teacher-only":
        target = teacher_output
        target_source = "teacher_pseudo_label"
    elif policy == "prefer-ground-truth":
        if isinstance(ground_truth, dict):
            target = ground_truth
            target_source = "ground_truth"
        else:
            target = teacher_output
            target_source = "teacher_pseudo_label"
    else:
        raise ValueError(
            "target_policy必须是prefer-ground-truth、ground-truth-only或teacher-only"
        )
    target_validation = validate_output(target)
    if not target_validation.valid:
        return None, "训练目标不符合统一输出协议"

    return {
        "task_id": task_id,
        "task_type": task_type,
        "instruction": TASK_INSTRUCTIONS.get(
            task_type, TASK_INSTRUCTIONS["generic_anomaly"]
        ) + "只输出符合协议的JSON对象。",
        "input": student_input,
        "output": target_validation.parsed,
        "target_source": target_source,
        "label_source": record.get("label_source", target_source),
        "teacher_model": record.get("teacher_model"),
        "teacher_output": teacher_output,
        "ground_truth": ground_truth,
    }, None


def _stratified_split(
    samples: list[dict[str, Any]], test_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio必须位于(0,1)")
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["task_type"])].append(sample)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for group in groups.values():
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        test_count = max(1, round(len(group) * test_ratio))
        test_count = min(test_count, len(group) - 1)
        test.extend(group[:test_count])
        train.extend(group[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_dataset(
    teacher_outputs_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    manifest_path: str | Path,
    *,
    target_policy: str = "prefer-ground-truth",
    test_ratio: float = 0.2,
    seed: int = 42,
    min_teacher_confidence: float = 0.0,
    require_ground_truth_for_test: bool = False,
) -> DatasetBuildReport:
    records = read_jsonl(teacher_outputs_path)
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        task_id = str(record.get("task_id", "")).strip()
        if task_id in seen:
            rejections.append({"task_id": task_id, "reason": "重复task_id"})
            continue
        seen.add(task_id)
        sample, error = _sample_from_record(
            record,
            target_policy=target_policy,
            min_teacher_confidence=min_teacher_confidence,
        )
        if sample is None:
            rejections.append({"task_id": task_id, "reason": error or "未知错误"})
        else:
            accepted.append(sample)
    if len(accepted) < 2:
        raise ValueError("有效蒸馏样本少于2条，无法划分训练集和测试集。")

    train, test = _stratified_split(accepted, test_ratio, seed)
    if require_ground_truth_for_test:
        ground_truth_test = [s for s in accepted if isinstance(s.get("ground_truth"), dict)]
        if not ground_truth_test:
            raise ValueError("要求真实标签测试集，但当前没有ground_truth样本。")
        test_ids = {s["task_id"] for s in ground_truth_test}
        test = ground_truth_test
        train = [s for s in accepted if s["task_id"] not in test_ids]
        if not train:
            raise ValueError("全部样本都进入ground_truth测试集，训练集为空。")

    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    manifest = {
        "source": str(teacher_outputs_path),
        "target_policy": target_policy,
        "test_ratio": test_ratio,
        "seed": seed,
        "min_teacher_confidence": min_teacher_confidence,
        "require_ground_truth_for_test": require_ground_truth_for_test,
        "input_count": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(rejections),
        "ground_truth_target_count": sum(
            1 for s in accepted if s["target_source"] == "ground_truth"
        ),
        "teacher_target_count": sum(
            1 for s in accepted if s["target_source"] == "teacher_pseudo_label"
        ),
        "train_count": len(train),
        "test_count": len(test),
        "train_task_ids": [s["task_id"] for s in train],
        "test_task_ids": [s["task_id"] for s in test],
        "rejections": rejections,
    }
    write_json(manifest_path, manifest)
    return DatasetBuildReport(
        input_count=len(records),
        accepted_count=len(accepted),
        rejected_count=len(rejections),
        ground_truth_target_count=manifest["ground_truth_target_count"],
        teacher_target_count=manifest["teacher_target_count"],
        train_count=len(train),
        test_count=len(test),
        train_path=str(train_path),
        test_path=str(test_path),
        manifest_path=str(manifest_path),
    )
