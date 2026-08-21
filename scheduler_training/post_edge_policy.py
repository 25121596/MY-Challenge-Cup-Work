"""Versioned post-edge review policy shared by simulation and real runtime.

The pre-route PPO never sees edge inference confidence.  After LOCAL_FIRST
finishes its edge stage, this policy decides whether the edge result can be
accepted or must be sent to the cloud review queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


RULE_SCHEMA_VERSION = "post-edge-rules/2.0"
SUPPORTED_RULE_SCHEMA_VERSIONS = {"post-edge-rules/1.0", RULE_SCHEMA_VERSION}
DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "config" / "post_edge_rules_v2.json"


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
    """Deterministic, configuration-driven post-edge rule evaluator."""

    def __init__(self, rules: Mapping[str, Any], *, source_path: str | Path | None = None) -> None:
        self.rules = dict(rules)
        self.source_path = None if source_path is None else str(Path(source_path).resolve())
        self._validate_rules()

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_RULES_PATH) -> "PostEdgePolicy":
        resolved = Path(path).resolve()
        with resolved.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("Post-edge规则文件必须包含JSON对象")
        return cls(payload, source_path=resolved)

    @property
    def rule_version(self) -> str:
        return str(self.rules["rule_version"])

    @property
    def schema_version(self) -> str:
        return str(self.rules["schema_version"])

    def _validate_rules(self) -> None:
        if self.rules.get("schema_version") not in SUPPORTED_RULE_SCHEMA_VERSIONS:
            raise ValueError(
                f"不支持的Post-edge规则schema：{self.rules.get('schema_version')}，"
                f"支持{sorted(SUPPORTED_RULE_SCHEMA_VERSIONS)}"
            )
        if not str(self.rules.get("rule_version", "")).strip():
            raise ValueError("Post-edge规则缺少rule_version")
        threshold_name = (
            "default_failure_probability_threshold"
            if self.schema_version == RULE_SCHEMA_VERSION
            else "default_confidence_threshold"
        )
        threshold = float(self.rules.get(threshold_name, -1.0))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{threshold_name}必须位于[0,1]")
        if self.schema_version == RULE_SCHEMA_VERSION:
            margin = float(self.rules.get("threshold_review_margin", 0.0))
            if not 0.0 <= margin <= 1.0:
                raise ValueError("threshold_review_margin必须位于[0,1]")
            routes = self.rules.get("always_review_model_routes", [])
            if not isinstance(routes, list):
                raise ValueError("always_review_model_routes必须是数组")
        task_thresholds = self.rules.get("task_confidence_thresholds", {})
        if not isinstance(task_thresholds, dict):
            raise ValueError("task_confidence_thresholds必须是JSON对象")
        for task_type, value in task_thresholds.items():
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"任务{task_type}的置信度阈值必须位于[0,1]")
        fallback = self.rules.get("deadline_infeasible_action", "ACCEPT_EDGE")
        if fallback != PostEdgeDecision.ACCEPT_EDGE.value:
            raise ValueError("当前schema只支持deadline_infeasible_action=ACCEPT_EDGE")

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
                decision=PostEdgeDecision.ACCEPT_EDGE,
                reason="EDGE_RESULT_ACCEPTED",
                rule_id="accept_default",
                rule_version=self.rule_version,
                raw_confidence=confidence,
                confidence_threshold=threshold,
                cloud_review_feasible=True,
                review_recommended=False,
            )

        feasible = self._review_feasible(
            task=task,
            now_s=now_s,
            cloud_review_estimated_s=cloud_review_estimated_s,
        )
        rule_id, reason = recommendation
        if not feasible:
            return PostEdgePolicyResult(
                decision=PostEdgeDecision.ACCEPT_EDGE,
                reason=f"{reason}_BUT_DEADLINE_INFEASIBLE",
                rule_id="deadline_feasibility",
                rule_version=self.rule_version,
                raw_confidence=confidence,
                confidence_threshold=threshold,
                cloud_review_feasible=False,
                review_recommended=True,
            )
        return PostEdgePolicyResult(
            decision=PostEdgeDecision.CLOUD_REVIEW,
            reason=reason,
            rule_id=rule_id,
            rule_version=self.rule_version,
            raw_confidence=confidence,
            confidence_threshold=threshold,
            cloud_review_feasible=True,
            review_recommended=True,
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
            confidence_threshold=self.confidence_threshold(str(getattr(task, "task_type", "generic_anomaly"))),
            cloud_review_feasible=feasible,
            review_recommended=review_recommended,
            raw_failure_probability=probability,
            failure_probability_threshold=threshold,
            evidence_source=source,
            producer_review_state=producer_state,
            producer_rule_consistent=producer_consistent,
        )
