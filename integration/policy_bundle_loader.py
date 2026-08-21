"""Load and validate Scheduler policy bundles for the real runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pre_route_observation import (
    ACTION_MAPPING,
    ACTION_SCHEMA_VERSION,
    OBSERVATION_NAMES,
    OBSERVATION_SCHEMA_VERSION,
    validate_observation_names,
)


BUNDLE_SCHEMA_VERSION = "cloud-edge-policy-bundle/1.0"
RULE_SCHEMA_VERSION = "post-edge-rules/2.0"
SUPPORTED_RULE_SCHEMA_VERSIONS = {"post-edge-rules/1.0", RULE_SCHEMA_VERSION}


class PolicyBundleError(ValueError):
    pass


class PostEdgeDecision(str, Enum):
    ACCEPT_EDGE = "ACCEPT_EDGE"
    CLOUD_REVIEW = "CLOUD_REVIEW"


@dataclass(frozen=True)
class PostEdgePolicyResult:
    decision: PostEdgeDecision
    reason: str
    rule_id: str
    rule_version: str
    raw_confidence: float | None
    confidence_threshold: float
    cloud_review_feasible: bool
    review_recommended: bool
    raw_failure_probability: float | None = None
    failure_probability_threshold: float | None = None
    evidence_source: str | None = None
    producer_review_state: str | None = None
    producer_rule_consistent: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "raw_confidence": self.raw_confidence,
            "confidence_threshold": self.confidence_threshold,
            "cloud_review_feasible": self.cloud_review_feasible,
            "review_recommended": self.review_recommended,
            "raw_failure_probability": self.raw_failure_probability,
            "failure_probability_threshold": self.failure_probability_threshold,
            "evidence_source": self.evidence_source,
            "producer_review_state": self.producer_review_state,
            "producer_rule_consistent": self.producer_rule_consistent,
        }


class PostEdgePolicy:
    """Integration evaluator for Scheduler post-edge-rules/1.0."""

    def __init__(self, rules: Mapping[str, Any], *, source_path: Path) -> None:
        self.rules = dict(rules)
        self.source_path = source_path.resolve()
        self._validate_rules()

    @property
    def rule_version(self) -> str:
        return str(self.rules["rule_version"])

    @property
    def schema_version(self) -> str:
        return str(self.rules["schema_version"])

    def _validate_rules(self) -> None:
        if self.rules.get("schema_version") not in SUPPORTED_RULE_SCHEMA_VERSIONS:
            raise PolicyBundleError(
                f"不支持的Post-edge规则schema: {self.rules.get('schema_version')}，"
                f"支持{sorted(SUPPORTED_RULE_SCHEMA_VERSIONS)}"
            )
        if not str(self.rules.get("rule_version", "")).strip():
            raise PolicyBundleError("Post-edge规则缺少rule_version")
        threshold_name = (
            "default_failure_probability_threshold"
            if self.schema_version == RULE_SCHEMA_VERSION
            else "default_confidence_threshold"
        )
        threshold = float(self.rules.get(threshold_name, -1.0))
        if not 0.0 <= threshold <= 1.0:
            raise PolicyBundleError(f"{threshold_name}必须位于[0,1]")
        if self.schema_version == RULE_SCHEMA_VERSION:
            margin = float(self.rules.get("threshold_review_margin", 0.0))
            if not 0.0 <= margin <= 1.0:
                raise PolicyBundleError("threshold_review_margin必须位于[0,1]")
            routes = self.rules.get("always_review_model_routes", [])
            if not isinstance(routes, list):
                raise PolicyBundleError("always_review_model_routes必须是数组")
        task_thresholds = self.rules.get("task_confidence_thresholds", {})
        if not isinstance(task_thresholds, dict):
            raise PolicyBundleError("task_confidence_thresholds必须是JSON对象")
        for task_type, value in task_thresholds.items():
            if not 0.0 <= float(value) <= 1.0:
                raise PolicyBundleError(f"任务{task_type}的置信度阈值必须位于[0,1]")
        if self.rules.get("deadline_infeasible_action", "ACCEPT_EDGE") != "ACCEPT_EDGE":
            raise PolicyBundleError("当前schema只支持deadline_infeasible_action=ACCEPT_EDGE")

    def confidence_threshold(self, task_type: str) -> float:
        if self.schema_version == RULE_SCHEMA_VERSION:
            return float(self.rules.get("legacy_confidence_fallback_threshold", 0.78))
        thresholds = self.rules.get("task_confidence_thresholds", {})
        return float(thresholds.get(task_type, self.rules["default_confidence_threshold"]))

    def failure_probability_threshold(self, edge_result: Mapping[str, Any]) -> float:
        configured = float(self.rules["default_failure_probability_threshold"])
        if bool(self.rules.get("honor_producer_threshold", True)):
            producer = self._float_or_none(edge_result.get("threshold_used"))
            if producer is not None:
                return producer
        return configured

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if 0.0 <= numeric <= 1.0 else None

    def _review_feasible(
        self,
        *,
        task: Any,
        now_s: float,
        cloud_review_estimated_s: float,
    ) -> bool:
        if not bool(self.rules.get("require_deadline_feasibility", True)):
            return True
        absolute_deadline = getattr(task, "absolute_deadline_s", None)
        if absolute_deadline is None:
            return True
        margin = max(0.0, float(self.rules.get("cloud_review_deadline_margin_s", 0.0)))
        remaining = float(absolute_deadline) - float(now_s)
        return float(cloud_review_estimated_s) + margin <= remaining

    def evaluate(
        self,
        *,
        task: Any,
        edge_result: Mapping[str, Any],
        now_s: float,
        cloud_review_estimated_s: float,
    ) -> PostEdgePolicyResult:
        if self.schema_version == RULE_SCHEMA_VERSION and (
            "failure_probability" in edge_result or edge_result.get("model_route")
        ):
            return self._evaluate_failure_probability(
                task=task,
                edge_result=edge_result,
                now_s=now_s,
                cloud_review_estimated_s=cloud_review_estimated_s,
            )

        task_type = str(getattr(task, "task_type", "generic_anomaly"))
        threshold = self.confidence_threshold(task_type)
        confidence = self._float_or_none(edge_result.get("confidence"))
        level = str(edge_result.get("level", "")).strip().lower()
        category = str(edge_result.get("category", "")).strip().lower()
        output_valid = bool(edge_result.get("output_valid", True))
        is_ood = bool(edge_result.get("ood", False))

        recommendation: tuple[str, str] | None = None
        if bool(self.rules.get("force_review_on_parse_failure", True)) and not output_valid:
            recommendation = ("output_validation", "EDGE_OUTPUT_INVALID")
        elif bool(self.rules.get("force_review_on_ood", True)) and is_ood:
            recommendation = ("ood_detection", "EDGE_OUTPUT_OOD")
        elif level in {str(v).lower() for v in self.rules.get("force_review_levels", [])}:
            recommendation = ("risk_level", f"FORCE_REVIEW_LEVEL_{level.upper()}")
        elif category in {str(v).lower() for v in self.rules.get("force_review_categories", [])}:
            recommendation = ("category", f"FORCE_REVIEW_CATEGORY_{category.upper()}")
        elif confidence is None:
            recommendation = ("confidence_missing", "EDGE_CONFIDENCE_UNAVAILABLE")
        elif confidence < threshold:
            recommendation = ("confidence_threshold", "LOW_EDGE_CONFIDENCE")

        if recommendation is None:
            return PostEdgePolicyResult(
                PostEdgeDecision.ACCEPT_EDGE,
                "EDGE_RESULT_ACCEPTED",
                "accept_default",
                self.rule_version,
                confidence,
                threshold,
                True,
                False,
            )
        feasible = self._review_feasible(
            task=task,
            now_s=now_s,
            cloud_review_estimated_s=cloud_review_estimated_s,
        )
        rule_id, reason = recommendation
        if not feasible:
            return PostEdgePolicyResult(
                PostEdgeDecision.ACCEPT_EDGE,
                f"{reason}_BUT_DEADLINE_INFEASIBLE",
                "deadline_feasibility",
                self.rule_version,
                confidence,
                threshold,
                False,
                True,
            )
        return PostEdgePolicyResult(
            PostEdgeDecision.CLOUD_REVIEW,
            reason,
            rule_id,
            self.rule_version,
            confidence,
            threshold,
            True,
            True,
        )

    def _evaluate_failure_probability(
        self,
        *,
        task: Any,
        edge_result: Mapping[str, Any],
        now_s: float,
        cloud_review_estimated_s: float,
    ) -> PostEdgePolicyResult:
        probability = self._float_or_none(edge_result.get("failure_probability"))
        threshold = self.failure_probability_threshold(edge_result)
        source = str(edge_result.get("source", "hhk_v2_edge_result"))
        model_route = str(edge_result.get("model_route", "")).strip()
        producer_state = str(edge_result.get("review_state", "")).strip().lower() or None
        consistent_raw = edge_result.get("producer_rule_consistent")
        producer_consistent = consistent_raw if isinstance(consistent_raw, bool) else None
        output_valid = bool(edge_result.get("output_valid", True))
        safety_trigger = bool(edge_result.get("safety_trigger", False))
        safety_priority = str(edge_result.get("safety_priority", "")).strip().upper()
        calibration = str(edge_result.get("calibration_version", "")).strip()
        feature_schema = str(edge_result.get("feature_schema_id", "")).strip()
        configured_margin = float(self.rules.get("threshold_review_margin", 0.0))
        producer_margin = max(0.0, float(edge_result.get("threshold_review_margin", 0.0) or 0.0))
        margin = max(configured_margin, producer_margin)

        recommendation: tuple[str, str] | None = None
        if not output_valid:
            recommendation = ("output_validation", "EDGE_OUTPUT_INVALID")
        elif safety_trigger or safety_priority in {"P0", "CRITICAL"}:
            recommendation = ("safety", "EDGE_SAFETY_TRIGGER")
        elif model_route in {str(v) for v in self.rules.get("always_review_model_routes", [])}:
            recommendation = ("uncalibrated_model_route", "UNCALIBRATED_MODEL_REQUIRES_REVIEW")
        elif bool(self.rules.get("require_calibration_version", True)) and not calibration:
            recommendation = ("calibration_missing", "EDGE_CALIBRATION_UNAVAILABLE")
        elif bool(self.rules.get("require_feature_schema_id", True)) and not feature_schema:
            recommendation = ("feature_schema_missing", "EDGE_FEATURE_SCHEMA_UNAVAILABLE")
        elif probability is None:
            recommendation = ("failure_probability_missing", "EDGE_FAILURE_PROBABILITY_UNAVAILABLE")
        elif probability > threshold:
            recommendation = ("failure_probability_threshold", "HIGH_FAILURE_PROBABILITY")
        elif abs(probability - threshold) <= margin:
            recommendation = ("failure_probability_margin", "FAILURE_PROBABILITY_NEAR_THRESHOLD")

        review_recommended = recommendation is not None
        feasible = True
        if review_recommended:
            feasible = self._review_feasible(
                task=task,
                now_s=now_s,
                cloud_review_estimated_s=cloud_review_estimated_s,
            )
        if recommendation is None:
            decision = PostEdgeDecision.ACCEPT_EDGE
            rule_id, reason = "accept_calibrated_ai4i", "EDGE_RESULT_ACCEPTED"
        elif not feasible:
            decision = PostEdgeDecision.ACCEPT_EDGE
            rule_id, reason = "deadline_feasibility", f"{recommendation[1]}_BUT_DEADLINE_INFEASIBLE"
        else:
            decision = PostEdgeDecision.CLOUD_REVIEW
            rule_id, reason = recommendation
        return PostEdgePolicyResult(
            decision=decision,
            reason=reason,
            rule_id=rule_id,
            rule_version=self.rule_version,
            raw_confidence=None,
            confidence_threshold=self.confidence_threshold(
                str(getattr(task, "task_type", "generic_anomaly"))
            ),
            cloud_review_feasible=feasible,
            review_recommended=review_recommended,
            raw_failure_probability=probability,
            failure_probability_threshold=threshold,
            evidence_source=source,
            producer_review_state=producer_state,
            producer_rule_consistent=producer_consistent,
        )


@dataclass(frozen=True)
class PolicyBundle:
    manifest_path: Path
    manifest: dict[str, Any]
    model_path: Path
    observation_schema_path: Path
    action_schema_path: Path
    post_edge_rules_path: Path
    observation_names: tuple[str, ...]
    post_edge_policy: PostEdgePolicy

    @property
    def policy_version(self) -> str:
        return str(self.manifest.get("policy_version", ""))

    @property
    def rule_version(self) -> str:
        return self.post_edge_policy.rule_version


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyBundleError(f"无法读取JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PolicyBundleError(f"JSON必须包含对象: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_bundle(manifest_path: str | Path) -> PolicyBundle:
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"找不到Scheduler policy bundle manifest: {manifest_file}\n"
            "请先在scheduler_training中训练并发布完整bundle，或使用--policy-bundle指定。"
        )
    manifest = _read_json(manifest_file)
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise PolicyBundleError("不支持的policy bundle schema")
    compatibility = manifest.get("compatibility", {})
    if not isinstance(compatibility, dict) or compatibility.get("legacy_25d_discrete3_ppo") is not False:
        raise PolicyBundleError("bundle必须明确声明旧25D/Discrete(3) PPO不兼容")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PolicyBundleError("bundle缺少artifacts")
    required = ("ppo_model", "observation_schema", "action_schema", "post_edge_rules")
    resolved: dict[str, Path] = {}
    for name in required:
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise PolicyBundleError(f"bundle缺少artifact: {name}")
        path = (manifest_file.parent / str(entry.get("path", ""))).resolve()
        if path.parent != manifest_file.parent:
            raise PolicyBundleError(f"artifact必须直接位于bundle目录: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = str(entry.get("sha256", ""))
        if not expected_hash or _sha256(path) != expected_hash:
            raise PolicyBundleError(f"artifact哈希不匹配: {name}")
        resolved[name] = path

    observation = _read_json(resolved["observation_schema"])
    if observation.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise PolicyBundleError("observation schema版本不匹配")
    if observation.get("dimension") != 24:
        raise PolicyBundleError("Integration只支持24D pre-route observation")
    observation_names = validate_observation_names(observation.get("names", []))

    action = _read_json(resolved["action_schema"])
    if action.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise PolicyBundleError("action schema版本不匹配")
    if action.get("space") != "Discrete(2)" or action.get("mapping") != ACTION_MAPPING:
        raise PolicyBundleError("Integration只支持LOCAL_FIRST/DIRECT_CLOUD Discrete(2)")

    rules = _read_json(resolved["post_edge_rules"])
    post_edge_policy = PostEdgePolicy(rules, source_path=resolved["post_edge_rules"])
    rule_entry = artifacts["post_edge_rules"]
    if post_edge_policy.schema_version != rule_entry.get("rule_schema_version"):
        raise PolicyBundleError("manifest与Post-edge规则schema不一致")
    if post_edge_policy.rule_version != rule_entry.get("rule_version"):
        raise PolicyBundleError("manifest与Post-edge规则版本不一致")

    return PolicyBundle(
        manifest_path=manifest_file,
        manifest=manifest,
        model_path=resolved["ppo_model"],
        observation_schema_path=resolved["observation_schema"],
        action_schema_path=resolved["action_schema"],
        post_edge_rules_path=resolved["post_edge_rules"],
        observation_names=observation_names,
        post_edge_policy=post_edge_policy,
    )
