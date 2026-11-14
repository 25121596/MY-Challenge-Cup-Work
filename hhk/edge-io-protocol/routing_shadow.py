"""Deterministic Phase-A exploration records for edge/cloud routing.

This module deliberately does not train a router or choose a production
threshold.  It defines the population eligible for a future cloud dual-run,
then performs reproducible randomized shadow selection so that later quality
estimates can account for the selection mechanism.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


ROUTING_OBSERVATION_SCHEMA_VERSION = "edge-cloud-routing-observation/v1"
ROUTING_POLICY_VERSION = "routing-shadow-exploration/v1"


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _network(task: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if override is not None:
        return dict(override)
    data = task.get("data", {})
    if isinstance(data.get("network"), dict):
        return dict(data["network"])
    if isinstance(task.get("network"), dict):
        return dict(task["network"])
    # Existing inputs predate network instrumentation.  Missing transport
    # state is not silently assumed healthy: it is outside exploration until
    # a collector supplies a decision-time network observation.
    return {"cloud_reachable": None, "network_state_source": "not_instrumented"}


def structural_eligibility(task: dict[str, Any], edge_output: dict[str, Any], *,
                           network: dict[str, Any] | None = None,
                           now: datetime | None = None) -> tuple[bool, str | None, dict[str, Any]]:
    """Return whether a sample may enter the learned-router population.

    P0 is intentionally excluded: it has a deterministic local protection
    action and any cloud review is parallel safety review, not exploration.
    """
    network_state = _network(task, network)
    if edge_output.get("safety_priority") == "P0":
        return False, "hard_safety_p0", network_state
    if task.get("data", {}).get("cloud_review_supported", task.get("cloud_review_supported", True)) is not True:
        return False, "cloud_review_unsupported", network_state
    if network_state.get("cloud_reachable") is not True:
        reason = "cloud_unreachable" if network_state.get("cloud_reachable") is False else "network_state_unavailable"
        return False, reason, network_state

    deadline = _parse_time(task.get("deadline_at"))
    if deadline is not None:
        current = now or datetime.now(timezone.utc)
        if deadline <= current:
            return False, "deadline_expired", network_state
        minimum_completion_ms = network_state.get("minimum_cloud_completion_ms")
        if minimum_completion_ms is not None:
            if isinstance(minimum_completion_ms, bool) or not isinstance(minimum_completion_ms, (int, float)):
                return False, "invalid_cloud_completion_budget", network_state
            remaining_ms = (deadline - current).total_seconds() * 1000.0
            if remaining_ms < minimum_completion_ms:
                return False, "deadline_budget_insufficient", network_state
    return True, None, network_state


def _draw(sample_id: str, seed: str, policy_version: str) -> float:
    material = f"{seed}|{policy_version}|{sample_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value / float(1 << 64)


def _feature_availability(task: dict[str, Any], edge_output: dict[str, Any],
                          decision_time: str) -> dict[str, dict[str, Any]]:
    """Declare the values available before the routing decision.

    The declaration is intentionally explicit.  A later training loader must
    reject any feature without this contract instead of inferring safety from
    a convenient column name.
    """
    task_time = task.get("observed_at", task.get("timestamp"))
    task_available = task_time if _parse_time(task_time) is not None else decision_time
    values = {
        "task_type": task.get("task_type"),
        "failure_probability": edge_output.get("failure_probability"),
        "raw_logit": edge_output.get("raw_logit"),
        "local_prediction": edge_output.get("local_prediction"),
        "review_state": edge_output.get("review_state"),
        "review_reason": edge_output.get("review_reason", []),
        "moe_gate_weights": edge_output.get("decision", {}).get("detail", {}).get("gate_weights"),
    }
    return {
        name: {
            "available_at": task_available if name == "task_type" else decision_time,
            "available_at_decision_time": True,
            "value_present": value is not None,
        }
        for name, value in values.items()
    }


def select_shadow(task: dict[str, Any], edge_output: dict[str, Any], *,
                  upload_probability: float, seed: str,
                  policy_version: str = ROUTING_POLICY_VERSION,
                  network: dict[str, Any] | None = None,
                  now: datetime | None = None) -> dict[str, Any]:
    """Build one auditable routing-selection record.

    A baseline-required review remains a policy upload.  Randomized shadow
    applies only to structurally eligible records that the baseline would keep
    local.  The caller must not submit those shadow records to the current
    admission mock, which accepts only ``review_state=required`` requests.
    """
    if not 0.0 <= float(upload_probability) <= 1.0:
        raise ValueError("upload_probability must be in [0, 1]")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a non-empty string")
    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError("policy_version must be a non-empty string")

    sample_id = str(task.get("task_id", edge_output.get("task_id", "unknown")))
    decision_time = str(edge_output.get("decided_at") or task.get("observed_at") or task.get("timestamp") or "")
    if _parse_time(decision_time) is None:
        decision_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    eligible, ineligible_reason, network_state = structural_eligibility(
        task, edge_output, network=network, now=now
    )
    baseline_upload = edge_output.get("review_state") == "required"
    draw = _draw(sample_id, seed, policy_version) if eligible and not baseline_upload else None
    randomized_shadow = draw is not None and draw < float(upload_probability)

    if not eligible:
        population_state = "structurally_ineligible"
        selection_reason = ineligible_reason
    elif baseline_upload:
        population_state = "policy_uploaded"
        selection_reason = "baseline_review_required"
    elif randomized_shadow:
        population_state = "randomized_shadow"
        selection_reason = "randomized_shadow_exploration"
    else:
        population_state = "eligible_not_uploaded"
        selection_reason = "randomized_shadow_not_selected"

    return {
        "routing_observation_schema_version": ROUTING_OBSERVATION_SCHEMA_VERSION,
        "policy_version": policy_version,
        "decision_time": decision_time,
        "shadow_eligible": eligible,
        # This is the probability of the observed upload assignment under the
        # active design: structural exclusions=0, deterministic baseline
        # review=1, randomized exploration=the configured propensity.
        "upload_probability": (
            0.0 if not eligible else (1.0 if baseline_upload else float(upload_probability))
        ),
        "selection_reason": selection_reason,
        "population_state": population_state,
        "baseline_upload": baseline_upload,
        "randomized_shadow": randomized_shadow,
        "random_draw": draw,
        "network": network_state,
        "feature_availability": _feature_availability(task, edge_output, decision_time),
    }


def validate_routing_selection(selection: dict[str, Any]) -> None:
    if selection.get("routing_observation_schema_version") != ROUTING_OBSERVATION_SCHEMA_VERSION:
        raise ValueError("unsupported routing observation schema version")
    if not isinstance(selection.get("shadow_eligible"), bool):
        raise ValueError("shadow_eligible must be boolean")
    probability = selection.get("upload_probability")
    if not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
        raise ValueError("upload_probability must be in [0, 1]")
    expected_states = {
        "structurally_ineligible", "eligible_not_uploaded", "randomized_shadow", "policy_uploaded",
    }
    if selection.get("population_state") not in expected_states:
        raise ValueError("unsupported routing population_state")
    if selection["population_state"] == "randomized_shadow":
        if not selection["shadow_eligible"] or not selection["randomized_shadow"]:
            raise ValueError("randomized_shadow state requires an eligible random selection")
    if selection["population_state"] == "structurally_ineligible" and selection["shadow_eligible"]:
        raise ValueError("structurally_ineligible must have shadow_eligible=false")
    availability = selection.get("feature_availability")
    if not isinstance(availability, dict) or not availability:
        raise ValueError("feature_availability is required")
    for name, item in availability.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise ValueError("invalid feature availability entry")
        if item.get("available_at_decision_time") is not True or _parse_time(item.get("available_at")) is None:
            raise ValueError("routing features require decision-time availability timestamps")
