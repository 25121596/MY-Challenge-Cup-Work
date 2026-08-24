"""Auditable edge/cloud observation records for shadow review.

This module intentionally treats a cloud model as a reviewer/teacher signal.
It never promotes a cloud answer to ground truth and never creates a business
priority score.  Records are append-only JSONL so failed and unlabeled cases
remain in the denominator of later analyses.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from routing_shadow import validate_routing_selection


FEEDBACK_SCHEMA_VERSION = "edge-cloud-feedback/v1"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _prediction(value: Any) -> str | None:
    if isinstance(value, str) and value in {"normal", "fault"}:
        return value
    if isinstance(value, dict):
        for key in ("local_prediction", "cloud_prediction", "prediction", "decision"):
            found = _prediction(value.get(key))
            if found:
                return found
        provisional = value.get("provisional_category")
        if isinstance(provisional, dict):
            name = provisional.get("name")
            if name in {"normal", "fault"}:
                return name
    return None


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compare_predictions(edge_output: dict[str, Any], cloud_output: dict[str, Any] | None) -> dict[str, Any]:
    """Return comparison labels without guessing when either side is absent."""
    edge_prediction = _prediction(edge_output)
    cloud_prediction = _prediction(cloud_output or {})
    comparable = edge_prediction is not None and cloud_prediction is not None
    disagreement = (edge_prediction != cloud_prediction) if comparable else None
    return {
        "edge_prediction": edge_prediction,
        "cloud_prediction": cloud_prediction,
        "disagreement": disagreement,
        "cloud_changed_decision": disagreement,
    }


def build_observation(task: dict[str, Any], edge_output: dict[str, Any], *,
                      raw_model_response: str = "", timing: dict[str, Any] | None = None,
                      cloud_output: dict[str, Any] | None = None,
                      cloud_ack: dict[str, Any] | None = None,
                      outcome: dict[str, Any] | None = None,
                      routing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one version-bound shadow observation record.

    ``review_useful`` is only accepted from an explicit outcome annotation or
    human confirmation.  A cloud disagreement alone is evidence, not truth.
    """
    comparison = compare_predictions(edge_output, cloud_output)
    outcome = dict(outcome or {})
    useful = outcome.get("review_useful")
    if useful is not None and not isinstance(useful, bool):
        raise ValueError("review_useful must be boolean or null")
    if useful is True and not (outcome.get("human_label") or outcome.get("confirmed_outcome")
                               or outcome.get("label_source")):
        raise ValueError("review_useful requires an explicit outcome source")

    def has_legacy_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(key in {"confidence", "priority", "device_label"} or has_legacy_key(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(has_legacy_key(item) for item in value)
        return False
    if has_legacy_key(task) or has_legacy_key(edge_output) or has_legacy_key(cloud_output or {}):
        raise ValueError("legacy confidence/priority fields are not accepted")
    task_id = str(task.get("task_id", edge_output.get("task_id", "unknown")))
    versions = {
        "schema_version": task.get("schema_version", "edge-input/v3"),
        "edge_model_version": edge_output.get("model_version"),
        "edge_prompt_version": edge_output.get("prompt_version"),
        "calibration_version": edge_output.get("calibration_version"),
        "cloud_model_version": (cloud_output or {}).get("model_version"),
        "cloud_prompt_version": (cloud_output or {}).get("prompt_version"),
        "routing_policy_version": edge_output.get("policy_version"),
    }
    record = {
        "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
        "sample_id": task_id,
        "trace_id": (edge_output.get("cloud_payload") or {}).get("trace_id"),
        "timestamp": _utc_now(),
        "observed_at": task.get("observed_at", task.get("timestamp")),
        "input": {
            "task_type": task.get("task_type"),
            "feature_schema_id": task.get("data", {}).get("feature_schema_id"),
            "feature_hash": _digest(task.get("data", {}).get("features")),
            "context": task.get("data", {}).get("context"),
        },
        "versions": versions,
        "edge": {
            "model_route": edge_output.get("model_route"),
            "model_version": edge_output.get("model_version"),
            "raw_logit": edge_output.get("raw_logit"),
            "failure_probability": edge_output.get("failure_probability"),
            "calibration_version": edge_output.get("calibration_version"),
            "local_prediction": edge_output.get("local_prediction"),
            "review_state": edge_output.get("review_state"),
            "review_reason": edge_output.get("review_reason", []),
            "raw_output": raw_model_response,
            "parsed_output": edge_output.get("decision"),
            "latency_ms": edge_output.get("latency_ms"),
            "timing": timing or {},
        },
        "cloud": {
            "model_version": (cloud_output or {}).get("model_version"),
            "prompt_version": (cloud_output or {}).get("prompt_version"),
            "raw_output": (cloud_output or {}).get("raw_output", (cloud_output or {}).get("content")),
            "parsed_output": cloud_output,
            "admission_ack": cloud_ack,
        },
        "comparison": comparison,
        "outcome": {
            "human_label": outcome.get("human_label"),
            "confirmed_outcome": outcome.get("confirmed_outcome"),
            "review_useful": useful,
            "label_source": outcome.get("label_source"),
            "available_at": outcome.get("available_at"),
        },
    }
    if routing is not None:
        validate_routing_selection(routing)
        record["routing"] = routing
    validate_observation(record)
    return record


def validate_observation(record: dict[str, Any]) -> None:
    """Reject protocol regressions before data reaches a training directory."""
    if record.get("feedback_schema_version") != FEEDBACK_SCHEMA_VERSION:
        raise ValueError("unsupported feedback schema version")
    def forbidden_key(value: Any) -> bool:
        if isinstance(value, dict):
            return any(key in {"confidence", "priority", "device_label"} or forbidden_key(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(forbidden_key(item) for item in value)
        return False
    if forbidden_key(record):
        raise ValueError("feedback records must not contain legacy confidence/priority fields")
    comparison = record.get("comparison", {})
    if comparison.get("disagreement") is not None and not isinstance(comparison["disagreement"], bool):
        raise ValueError("disagreement must be boolean or null")
    useful = record.get("outcome", {}).get("review_useful")
    if useful is not None and not isinstance(useful, bool):
        raise ValueError("review_useful must be boolean or null")
    routing = record.get("routing")
    if routing is not None:
        if not isinstance(routing, dict):
            raise ValueError("routing must be an object")
        validate_routing_selection(routing)


def append_observation(path: str | Path, record: dict[str, Any]) -> None:
    validate_observation(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    comparable = [row for row in rows if row.get("comparison", {}).get("disagreement") is not None]
    labeled = [row for row in rows if row.get("outcome", {}).get("review_useful") is not None]
    disagreements = [row for row in comparable if row["comparison"]["disagreement"]]
    useful = [row for row in labeled if row["outcome"]["review_useful"]]
    return {
        "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
        "records": len(rows),
        "comparable_predictions": len(comparable),
        "disagreement_count": len(disagreements),
        "disagreement_rate": len(disagreements) / len(comparable) if comparable else None,
        "review_useful_labeled": len(labeled),
        "review_useful_count": len(useful),
        "review_useful_rate": len(useful) / len(labeled) if labeled else None,
        "unlabeled_records": len(rows) - len(labeled),
    }
