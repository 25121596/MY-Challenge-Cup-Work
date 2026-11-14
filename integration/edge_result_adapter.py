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


def _probability_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if 0.0 <= numeric <= 1.0 else None


def normalize_edge_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return v3 edge semantics plus legacy signals consumed by PostEdgePolicy."""

    root = _mapping(result)
    output = _mapping(root.get("output"))
    parsed = _parse_answer(root.get("answer"))
    parsed_output = _mapping(parsed.get("output"))
    decision = _mapping(output.get("decision"))
    parsed_decision = _mapping(parsed_output.get("decision") or parsed.get("decision"))
    rows = [root, output, parsed, parsed_output, decision, parsed_decision]

    # ``failure_probability`` is a calibrated risk probability, not a generic
    # confidence score. Keep it separate so the scheduler never reverses its
    # meaning by applying a low-confidence threshold to it.
    failure_probability = _probability_or_none(_first(rows, "failure_probability"))
    local_prediction = _first(rows, "local_prediction")
    review_state = _first(rows, "review_state")
    safety_priority = _first(rows, "safety_priority")
    safety_trigger = _first(rows, "safety_trigger")
    model_route = _first(rows, "model_route")
    edge_action = _first(rows, "action")

    # Legacy generic models may still provide this field. Current edge-output/v3
    # does not, and failure_probability must never be copied here.
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

    return {
        "confidence": confidence,
        "failure_probability": failure_probability,
        "local_prediction": (
            str(local_prediction) if local_prediction in {"normal", "fault"} else None
        ),
        "review_state": (
            str(review_state) if review_state in {"required", "not_required"} else None
        ),
        "safety_priority": (
            str(safety_priority) if safety_priority in {"P0", "P1", "P2"} else None
        ),
        "safety_trigger": None if safety_trigger is None else str(safety_trigger),
        "model_route": None if model_route is None else str(model_route),
        "edge_action": None if edge_action is None else str(edge_action),
        "level": "" if level is None else str(level),
        "category": "" if category is None else str(category),
        "output_valid": output_valid,
        "ood": bool(ood_value) if ood_value is not None else None,
        # This is observability only. Integration PostEdgePolicy is authoritative.
        "edge_escalation_hint": (
            bool(escalation_hint) if escalation_hint is not None else None
        ),
    }
