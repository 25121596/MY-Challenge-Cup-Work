"""Normalize measured edge responses into Post-edge policy signals.

Missing values remain unavailable.  In particular this adapter never invents
confidence, OOD status, or output validity from model latency/profile data.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parse_answer(answer: Any) -> Mapping[str, Any]:
    if isinstance(answer, Mapping):
        return answer
    if not isinstance(answer, str):
        return {}
    text = answer.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return _mapping(parsed)


def _first(mapping_rows: list[Mapping[str, Any]], *keys: str) -> Any:
    for row in mapping_rows:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
    return None


def normalize_edge_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return measured HHK V2 signals plus legacy confidence signals."""

    root = _mapping(result)
    output = _mapping(root.get("output"))
    parsed = _parse_answer(root.get("answer"))
    parsed_output = _mapping(parsed.get("output"))
    decision = _mapping(output.get("decision"))
    parsed_decision = _mapping(parsed_output.get("decision") or parsed.get("decision"))
    rows = [root, output, parsed, parsed_output, decision, parsed_decision]

    confidence = _first(rows, "confidence", "score", "probability")
    level = _first([decision, parsed_decision, output, parsed, root], "level", "risk_level")
    category = _first(
        [decision, parsed_decision, output, parsed, root],
        "category",
        "class",
        "label",
        "action",
    )
    explicit_valid = _first(rows, "output_valid", "valid")
    parse_error = _first(rows, "parse_error")
    if explicit_valid is None:
        output_valid = bool(output or parsed) and not bool(parse_error)
    else:
        output_valid = bool(explicit_valid) and not bool(parse_error)
    ood_value = _first(rows, "ood", "is_ood", "out_of_distribution")
    escalation_hint = _first(rows, "escalate_to_cloud", "request_cloud")
    failure_probability = _first(rows, "failure_probability")
    threshold_used = _first(rows, "threshold_used")
    threshold_review_margin = _first(rows, "threshold_review_margin")
    model_route = _first(rows, "model_route")
    calibration_version = _first(rows, "calibration_version")
    feature_schema_id = _first(rows, "feature_schema_id")
    review_state = _first(rows, "review_state")
    review_reason = _first(rows, "review_reason")
    safety_priority = _first(rows, "safety_priority")
    safety_trigger = _first(rows, "safety_trigger")
    local_prediction = _first(rows, "local_prediction")
    source = _first(rows, "source")
    producer_rule_consistent = _first(rows, "producer_rule_consistent")

    normalized = {
        "confidence": confidence,
        "level": "" if level is None else str(level),
        "category": "" if category is None else str(category),
        "output_valid": output_valid,
        "ood": bool(ood_value) if ood_value is not None else False,
        # This is observability only. Integration PostEdgePolicy is authoritative.
        "edge_escalation_hint": (
            bool(escalation_hint) if escalation_hint is not None else None
        ),
    }
    optional_v2 = {
        "failure_probability": failure_probability,
        "threshold_used": threshold_used,
        "threshold_review_margin": threshold_review_margin,
        "model_route": None if model_route is None else str(model_route),
        "calibration_version": None if calibration_version is None else str(calibration_version),
        "feature_schema_id": None if feature_schema_id is None else str(feature_schema_id),
        "review_state": None if review_state is None else str(review_state),
        "review_reason": review_reason if isinstance(review_reason, list) else None,
        "safety_priority": None if safety_priority is None else str(safety_priority),
        "safety_trigger": safety_trigger,
        "local_prediction": None if local_prediction is None else str(local_prediction),
        "source": None if source is None else str(source),
        "producer_rule_consistent": (
            producer_rule_consistent if isinstance(producer_rule_consistent, bool) else None
        ),
    }
    normalized.update({key: value for key, value in optional_v2.items() if value is not None})
    return normalized
