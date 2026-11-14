from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .schema import REQUIRED_OUTPUT_FIELDS, VALID_LEVELS


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    parsed: dict[str, Any] | None
    errors: tuple[str, ...]


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    depth = 0
    start = None
    candidates: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(stripped[start:index + 1])
                start = None
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def normalize_output(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for field in REQUIRED_OUTPUT_FIELDS:
        if field in result and result[field] is not None:
            result[field] = str(result[field]).strip()
    if "level" in result:
        result["level"] = str(result["level"]).strip().lower()
    if "confidence" in result:
        try:
            result["confidence"] = float(result["confidence"])
        except (TypeError, ValueError):
            pass
    return result


def validate_output(value: str | dict[str, Any]) -> ValidationResult:
    parsed = extract_json_object(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        return ValidationResult(False, None, ("无法解析为JSON对象",))
    parsed = normalize_output(parsed)
    errors: list[str] = []
    for field in REQUIRED_OUTPUT_FIELDS:
        if not isinstance(parsed.get(field), str) or not parsed[field]:
            errors.append(f"缺少非空字段：{field}")
    if parsed.get("level") not in VALID_LEVELS:
        errors.append(
            f"level必须是{list(VALID_LEVELS)}之一，当前为{parsed.get('level')!r}"
        )
    confidence = parsed.get("confidence")
    if confidence is not None and (
        not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0
    ):
        errors.append("confidence存在时必须位于[0,1]")
    return ValidationResult(not errors, parsed, tuple(errors))
