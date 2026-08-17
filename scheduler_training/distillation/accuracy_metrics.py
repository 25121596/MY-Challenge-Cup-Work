from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .output_validation import validate_output


@dataclass(frozen=True)
class AccuracyReport:
    sample_count: int
    valid_json_rate: float
    category_accuracy: float | None
    category_macro_precision: float | None
    category_macro_recall: float | None
    category_macro_f1: float | None
    level_accuracy: float | None
    action_accuracy: float | None
    critical_recall: float | None
    complete_record_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _safe_ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _macro_scores(pairs: list[tuple[str, str]]) -> tuple[float | None, float | None, float | None]:
    labels = sorted({value for pair in pairs for value in pair if value})
    if not labels:
        return None, None, None
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for pred, truth in pairs if pred == label and truth == label)
        fp = sum(1 for pred, truth in pairs if pred == label and truth != label)
        fn = sum(1 for pred, truth in pairs if pred != label and truth == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return (
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1s) / len(f1s),
    )


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> AccuracyReport:
    """rows每条需含prediction与ground_truth对象。没有真实标签时只统计JSON有效率。"""
    valid = 0
    truth_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        prediction = row.get("prediction")
        validation = validate_output(prediction if prediction is not None else "")
        if validation.valid and validation.parsed:
            valid += 1
        truth = row.get("ground_truth")
        if isinstance(truth, dict) and validation.valid and validation.parsed:
            truth_rows.append((validation.parsed, truth))
    n = len(rows)
    if not truth_rows:
        return AccuracyReport(
            sample_count=n,
            valid_json_rate=0.0 if n == 0 else valid / n,
            category_accuracy=None,
            category_macro_precision=None,
            category_macro_recall=None,
            category_macro_f1=None,
            level_accuracy=None,
            action_accuracy=None,
            critical_recall=None,
            complete_record_accuracy=None,
        )

    category_pairs = [
        (str(pred.get("category", "")), str(truth.get("category", "")))
        for pred, truth in truth_rows
    ]
    macro_p, macro_r, macro_f1 = _macro_scores(category_pairs)
    category_correct = sum(pred == truth for pred, truth in category_pairs)
    level_correct = sum(
        str(pred.get("level", "")) == str(truth.get("level", ""))
        for pred, truth in truth_rows
    )
    action_correct = sum(
        str(pred.get("action", "")) == str(truth.get("action", ""))
        for pred, truth in truth_rows
    )
    complete_correct = sum(
        all(str(pred.get(field, "")) == str(truth.get(field, ""))
            for field in ("category", "level", "action"))
        for pred, truth in truth_rows
    )
    critical_truth = sum(1 for _, truth in truth_rows if truth.get("level") == "critical")
    critical_tp = sum(
        1 for pred, truth in truth_rows
        if truth.get("level") == "critical" and pred.get("level") == "critical"
    )
    count = len(truth_rows)
    return AccuracyReport(
        sample_count=n,
        valid_json_rate=0.0 if n == 0 else valid / n,
        category_accuracy=category_correct / count,
        category_macro_precision=macro_p,
        category_macro_recall=macro_r,
        category_macro_f1=macro_f1,
        level_accuracy=level_correct / count,
        action_accuracy=action_correct / count,
        critical_recall=_safe_ratio(critical_tp, critical_truth),
        complete_record_accuracy=complete_correct / count,
    )
