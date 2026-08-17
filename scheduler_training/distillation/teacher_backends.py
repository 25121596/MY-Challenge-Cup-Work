from __future__ import annotations

from abc import ABC, abstractmethod
import json
import math
import time
from typing import Any
import urllib.error
import urllib.request

from .output_validation import ValidationResult, validate_output
from .schema import CATEGORY_HINTS, TASK_INSTRUCTIONS, TeacherConfig


class TeacherBackendError(RuntimeError):
    """教师服务调用或输出失败。"""


def student_visible_view(task: dict[str, Any]) -> dict[str, Any]:
    """提取未来学生部署时可获得的信息，避免把标签或系统后验泄漏给学生。"""
    data = task.get("data") if isinstance(task.get("data"), dict) else {}
    view: dict[str, Any] = {
        "task_type": str(task.get("task_type", "generic_anomaly")),
        "priority": str(task.get("priority", "medium")),
        "data": data,
    }
    if task.get("max_latency_ms") is not None:
        view["max_latency_ms"] = task["max_latency_ms"]
    return view


def build_prompt(task: dict[str, Any], *, no_think: bool = True) -> tuple[str, str]:
    task_type = str(task.get("task_type", "generic_anomaly"))
    instruction = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS["generic_anomaly"])
    categories = CATEGORY_HINTS.get(task_type, CATEGORY_HINTS["generic_anomaly"])
    system = (
        "你是工业任务教师模型。你的答案将用于训练边端轻量学生模型。"
        "只依据学生可见输入判断，不得假设未提供的图像、视频、日志或传感器信息。"
        "只输出一个JSON对象，不要输出Markdown、思考过程或额外文字。"
        "JSON必须包含action、level、category、summary、reason；"
        "level只能为low、medium、high、critical。"
        "可选增加confidence（0到1）。"
    )
    user = (
        f"任务要求：{instruction}\n"
        f"建议类别集合：{json.dumps(categories, ensure_ascii=False)}\n"
        "学生可见输入：\n"
        f"{json.dumps(student_visible_view(task), ensure_ascii=False, indent=2)}"
    )
    if no_think:
        user += "\n/no_think"
    return system, user


class TeacherBackend(ABC):
    name: str

    @abstractmethod
    def generate(self, task: dict[str, Any]) -> tuple[str, ValidationResult]:
        raise NotImplementedError

    def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": self.name}


class OpenAICompatibleTeacher(TeacherBackend):
    name = "openai_compatible_api"

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        base = config.api_base.rstrip("/")
        self.endpoint = (
            base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        )

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_s
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TeacherBackendError(
                f"教师API返回HTTP {exc.code}：{detail[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise TeacherBackendError(f"无法连接教师API：{exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise TeacherBackendError("教师API响应不是合法JSON") from exc

    def generate(self, task: dict[str, Any]) -> tuple[str, ValidationResult]:
        system, user = build_prompt(task, no_think=self.config.no_think)
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = self._request(body)
                raw = str(response["choices"][0]["message"]["content"] or "")
                result = validate_output(raw)
                if result.valid:
                    return raw, result
                last_error = TeacherBackendError("；".join(result.errors))
            except (KeyError, IndexError, TypeError, TeacherBackendError) as exc:
                last_error = exc
            if attempt < self.config.retries:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
        raise TeacherBackendError(f"教师生成失败：{last_error}")

    def health(self) -> dict[str, Any]:
        base = self.config.api_base.rstrip("/")
        root = base[:-3] if base.endswith("/v1") else base
        request = urllib.request.Request(f"{root}/health", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                body = response.read().decode("utf-8", errors="replace")
                return {
                    "ok": response.status == 200,
                    "status": response.status,
                    "body": body[:500],
                    "backend": self.name,
                }
        except Exception as exc:  # 健康检查用于展示，保留简洁错误
            return {"ok": False, "error": str(exc), "backend": self.name}


class SimulatedTeacher(TeacherBackend):
    """可重复的规则教师，仅用于管线验证，不能作为真实准确率结论。"""

    name = "simulated_rule"

    def generate(self, task: dict[str, Any]) -> tuple[str, ValidationResult]:
        task_type = str(task.get("task_type", "generic_anomaly"))
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        readings = data.get("readings") if isinstance(data.get("readings"), dict) else {}
        numeric = []
        for value in readings.values():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                numeric.append(abs(number))
        score = 0.0 if not numeric else min(3.5, sum(numeric) / max(len(numeric), 1) / 10.0)
        if score >= 2.5:
            level = "critical"
        elif score >= 1.5:
            level = "high"
        elif score >= 0.7:
            level = "medium"
        else:
            level = "low"
        categories = CATEGORY_HINTS.get(task_type, CATEGORY_HINTS["generic_anomaly"])
        category = categories[0] if level == "low" else categories[min(1, len(categories) - 1)]
        action = {
            "low": "monitor",
            "medium": "inspect",
            "high": "alarm",
            "critical": "shutdown",
        }[level]
        output = {
            "action": action,
            "level": level,
            "category": category,
            "summary": f"规则教师判定为{category}，严重程度为{level}",
            "reason": "根据学生可见数值特征的规则分数生成，仅用于验证蒸馏流程。",
            "confidence": 0.5,
        }
        raw = json.dumps(output, ensure_ascii=False)
        return raw, validate_output(output)


def build_teacher_backend(config: TeacherConfig) -> TeacherBackend:
    backend = config.backend.strip().lower()
    if backend in {"api", "openai", "openai_compatible", "llama_server"}:
        return OpenAICompatibleTeacher(config)
    if backend in {"simulated", "rule", "mock"}:
        return SimulatedTeacher()
    raise ValueError(f"不支持的teacher.backend：{config.backend}")
