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


RULE_SCHEMA_VERSION = "post-edge-rules/1.0"
DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "config" / "post_edge_rules_v1.json"


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
        if self.rules.get("schema_version") != RULE_SCHEMA_VERSION:
            raise ValueError(
                f"不支持的Post-edge规则schema：{self.rules.get('schema_version')}，"
                f"需要{RULE_SCHEMA_VERSION}"
            )
        if not str(self.rules.get("rule_version", "")).strip():
            raise ValueError("Post-edge规则缺少rule_version")
        threshold = float(self.rules.get("default_confidence_threshold", -1.0))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("default_confidence_threshold必须位于[0,1]")
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
        thresholds = self.rules.get("task_confidence_thresholds", {})
        return float(thresholds.get(task_type, self.rules["default_confidence_threshold"]))

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
