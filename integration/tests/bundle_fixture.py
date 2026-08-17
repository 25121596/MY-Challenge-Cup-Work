from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pre_route_observation import ACTION_MAPPING, OBSERVATION_NAMES


TEST_RULES = {
    "schema_version": "post-edge-rules/1.0",
    "rule_version": "post-edge-test-v1",
    "default_confidence_threshold": 0.78,
    "task_confidence_thresholds": {},
    "force_review_on_parse_failure": True,
    "force_review_on_ood": True,
    "force_review_levels": ["critical"],
    "force_review_categories": ["unknown_fault", "unknown"],
    "require_deadline_feasibility": True,
    "cloud_review_deadline_margin_s": 0.0,
    "deadline_infeasible_action": "ACCEPT_EDGE",
}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_test_bundle(root: str | Path) -> Path:
    release = Path(root).resolve()
    release.mkdir(parents=True, exist_ok=True)
    model = release / "pre_route_ppo.zip"
    observation = release / "observation_schema.json"
    action = release / "action_schema.json"
    rules = release / "post_edge_rules.json"
    model.write_bytes(b"test-only-placeholder-not-a-real-ppo")
    _write_json(observation, {
        "schema_version": "pre-route-observation/2.0",
        "dimension": 24,
        "names": list(OBSERVATION_NAMES),
        "removed_from_v1": ["task_confidence"],
        "decision_time": "pre_route",
    })
    _write_json(action, {
        "schema_version": "pre-route-action/2.0",
        "space": "Discrete(2)",
        "mapping": ACTION_MAPPING,
        "note": "CLOUD_REVIEW is created only by PostEdgePolicy.",
    })
    _write_json(rules, TEST_RULES)
    artifacts = {}
    for name, path in (
        ("ppo_model", model),
        ("observation_schema", observation),
        ("action_schema", action),
        ("post_edge_rules", rules),
    ):
        artifacts[name] = {"path": path.name, "sha256": _sha256(path)}
    artifacts["post_edge_rules"].update({
        "rule_schema_version": TEST_RULES["schema_version"],
        "rule_version": TEST_RULES["rule_version"],
    })
    manifest = release / "policy_bundle.json"
    _write_json(manifest, {
        "bundle_schema_version": "cloud-edge-policy-bundle/1.0",
        "policy_version": "pre-route-test-v1",
        "calibration_status": "test_fixture_not_real_calibrated",
        "compatibility": {
            "legacy_25d_discrete3_ppo": False,
            "reason": "Observation is 24D and action space is Discrete(2).",
        },
        "artifacts": artifacts,
        "training": {"fixture": True},
    })
    return manifest
