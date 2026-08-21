"""Publish and validate deployable Phase4.5 pre-route policy bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from models import PPO_ACTION_TO_ROUTE
from observation import OBSERVATION_NAMES
from post_edge_policy import PostEdgePolicy


BUNDLE_SCHEMA_VERSION = "cloud-edge-policy-bundle/1.0"
OBSERVATION_SCHEMA_VERSION = "pre-route-observation/2.0"
ACTION_SCHEMA_VERSION = "pre-route-action/2.0"
ACTION_MAPPING = {
    str(index): route.name for index, route in PPO_ACTION_TO_ROUTE.items()
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _zip_path(path: str | Path) -> Path:
    result = Path(path).resolve()
    if result.suffix == "":
        result = result.with_suffix(".zip")
    if result.suffix.lower() != ".zip":
        raise ValueError("PPO模型必须是Stable-Baselines3 .zip文件")
    return result


def publish_policy_bundle(
    *,
    model_path: str | Path,
    release_dir: str | Path,
    rules_path: str | Path,
    policy_version: str,
    training_metadata: dict[str, Any] | None = None,
    calibration_metadata: dict[str, Any] | None = None,
) -> Path:
    """Publish model, schemas, rules and hashes into one release directory."""
    source_model = _zip_path(model_path)
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    policy_version = str(policy_version).strip()
    if not policy_version:
        raise ValueError("policy_version不能为空")

    release = Path(release_dir).resolve()
    release.mkdir(parents=True, exist_ok=True)
    model_target = release / "pre_route_ppo.zip"
    if source_model != model_target:
        shutil.copy2(source_model, model_target)

    rules_source = Path(rules_path).resolve()
    rules_policy = PostEdgePolicy.from_json(rules_source)
    rules_target = release / "post_edge_rules.json"
    if rules_source != rules_target:
        shutil.copy2(rules_source, rules_target)

    observation_schema = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "dimension": len(OBSERVATION_NAMES),
        "names": list(OBSERVATION_NAMES),
        "removed_from_v1": ["task_confidence"],
        "decision_time": "pre_route",
    }
    action_schema = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "space": "Discrete(2)",
        "mapping": ACTION_MAPPING,
        "note": "CLOUD_REVIEW is created only by PostEdgePolicy and is not a PPO action.",
    }
    observation_path = release / "observation_schema.json"
    action_path = release / "action_schema.json"
    _write_json(observation_path, observation_schema)
    _write_json(action_path, action_schema)
    calibration_path = release / "calibration_manifest.json"
    calibration_payload = dict(calibration_metadata or {
        "calibration_status": "legacy_parameters_not_real_calibrated",
        "scope": "legacy simulator parameters",
    })
    _write_json(calibration_path, calibration_payload)

    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "policy_version": policy_version,
        "calibration_status": calibration_payload.get(
            "calibration_status", "legacy_parameters_not_real_calibrated"
        ),
        "compatibility": {
            "legacy_25d_discrete3_ppo": False,
            "reason": "Observation is 24D and action space is Discrete(2).",
        },
        "artifacts": {
            "ppo_model": {
                "path": model_target.name,
                "sha256": sha256_file(model_target),
            },
            "observation_schema": {
                "path": observation_path.name,
                "sha256": sha256_file(observation_path),
            },
            "action_schema": {
                "path": action_path.name,
                "sha256": sha256_file(action_path),
            },
            "post_edge_rules": {
                "path": rules_target.name,
                "sha256": sha256_file(rules_target),
                "rule_schema_version": rules_policy.schema_version,
                "rule_version": rules_policy.rule_version,
            },
            "calibration_manifest": {
                "path": calibration_path.name,
                "sha256": sha256_file(calibration_path),
            },
        },
        "training": dict(training_metadata or {}),
    }
    manifest_path = release / "policy_bundle.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_policy_bundle(
    manifest_path: str | Path,
    *,
    load_ppo: bool = False,
) -> dict[str, Any]:
    """Validate bundle schemas, paths, hashes and optionally SB3 spaces."""
    manifest_file = Path(manifest_path).resolve()
    manifest = _load_json(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError("policy_bundle.json必须包含JSON对象")
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("不支持的policy bundle schema")
    if manifest.get("compatibility", {}).get("legacy_25d_discrete3_ppo") is not False:
        raise ValueError("bundle必须明确声明旧25D/Discrete(3) PPO不兼容")

    bundle_dir = manifest_file.parent
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("bundle缺少artifacts")
    required = ("ppo_model", "observation_schema", "action_schema", "post_edge_rules")
    resolved: dict[str, Path] = {}
    for name in required:
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"bundle缺少artifact：{name}")
        path = (bundle_dir / str(entry.get("path", ""))).resolve()
        if path.parent != bundle_dir:
            raise ValueError(f"artifact必须直接位于bundle目录：{name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"artifact哈希不匹配：{name}")
        resolved[name] = path
    calibration_entry = artifacts.get("calibration_manifest")
    if isinstance(calibration_entry, dict):
        calibration_path = (bundle_dir / str(calibration_entry.get("path", ""))).resolve()
        if calibration_path.parent != bundle_dir or not calibration_path.is_file():
            raise ValueError("calibration_manifest路径非法或文件不存在")
        if sha256_file(calibration_path) != calibration_entry.get("sha256"):
            raise ValueError("artifact哈希不匹配：calibration_manifest")
        resolved["calibration_manifest"] = calibration_path

    observation = _load_json(resolved["observation_schema"])
    if observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("observation schema版本不匹配")
    if observation.get("dimension") != 24 or observation.get("names") != OBSERVATION_NAMES:
        raise ValueError("bundle observation不是当前24D pre-route契约")
    if "task_confidence" in observation.get("names", []):
        raise ValueError("24D pre-route observation不得包含task_confidence")

    action = _load_json(resolved["action_schema"])
    if action.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise ValueError("action schema版本不匹配")
    if action.get("space") != "Discrete(2)" or action.get("mapping") != ACTION_MAPPING:
        raise ValueError("bundle action不是LOCAL_FIRST/DIRECT_CLOUD Discrete(2)契约")

    rules = PostEdgePolicy.from_json(resolved["post_edge_rules"])
    rule_entry = artifacts["post_edge_rules"]
    if rules.rule_version != rule_entry.get("rule_version"):
        raise ValueError("manifest与Post-edge规则版本不一致")
    calibration = (
        _load_json(resolved["calibration_manifest"])
        if "calibration_manifest" in resolved
        else {"calibration_status": manifest.get("calibration_status", "legacy_parameters_not_real_calibrated")}
    )
    if not isinstance(calibration, dict) or not calibration.get("calibration_status"):
        raise ValueError("calibration_manifest缺少calibration_status")

    if load_ppo:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError("加载PPO校验需要stable-baselines3") from exc
        model = PPO.load(str(resolved["ppo_model"]))
        if getattr(getattr(model, "action_space", None), "n", None) != 2:
            raise ValueError("bundle PPO不是Discrete(2)")
        if tuple(getattr(getattr(model, "observation_space", None), "shape", ())) != (24,):
            raise ValueError("bundle PPO不是24D observation")

    return {
        "valid": True,
        "manifest": str(manifest_file),
        "policy_version": manifest.get("policy_version"),
        "observation_dimension": 24,
        "action_mapping": dict(ACTION_MAPPING),
        "post_edge_rule_version": rules.rule_version,
        "calibration_status": calibration["calibration_status"],
        "ppo_loaded": bool(load_ppo),
    }
