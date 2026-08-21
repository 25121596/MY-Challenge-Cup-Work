from __future__ import annotations

import hashlib
import json
import math
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from profiling import parse_llama_profile


CLOUD_RESULT_SCHEMA_VERSION = "ai4i-cloud-review-result/v1"
FEEDBACK_SCHEMA_VERSION = "edge-cloud-feedback/v1"
EVALUATION_SCHEMA_VERSION = "ai4i-edge-cloud-evaluation/v1"
PROMPT_ENCODING = "ai4i_qwen_8d_v1"
BINARY_GRAMMAR = "root ::= [01]"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number} 不是合法 JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number} 必须是 JSON object")
            rows.append(row)
    return rows


def index_unique(rows: Iterable[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{source}:{line_number} 缺少 task_id")
        if task_id in indexed:
            raise ValueError(f"{source} 包含重复 task_id: {task_id}")
        indexed[task_id] = row
    return indexed


def load_failure_cutoff(metrics_path: str | Path) -> float:
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        cutoff = float(payload["cloud_review_policy"]["normal_failure_probability_cutoff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metrics JSON 缺少 cloud_review_policy.normal_failure_probability_cutoff") from exc
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("normal_failure_probability_cutoff 必须位于 [0, 1]")
    return cutoff


def edge_prediction(row: dict[str, Any], cutoff: float) -> str:
    try:
        probability = float(row["failure_probability"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{row.get('task_id', 'unknown')} 缺少有效 failure_probability") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{row.get('task_id', 'unknown')} 的 failure_probability 非法")
    return "normal" if probability <= cutoff else "fault"


def classifier_prediction(row: dict[str, Any]) -> str:
    """Reference MoE class at the ordinary 0.5 probability threshold.

    This is intentionally distinct from HHK's conservative operational cutoff,
    which is used by ``edge_prediction`` and the Post-edge review route.
    """
    try:
        probability = float(row["failure_probability"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{row.get('task_id', 'unknown')} 缺少有效 failure_probability") from exc
    return "fault" if probability > 0.5 else "normal"


def label_prediction(row: dict[str, Any]) -> str:
    label = row.get("label")
    if label not in (0, 1):
        raise ValueError(f"{row.get('task_id', 'unknown')} 的 label 必须为 0 或 1")
    return "fault" if label == 1 else "normal"


def validate_sources(
    edge_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    cutoff: float,
) -> dict[str, dict[str, Any]]:
    prompts = index_unique(prompt_rows, source="Qwen input")
    seen_edge: set[str] = set()
    missing_prompts: list[str] = []
    inconsistent_routes: list[str] = []
    for row in edge_rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("edge predictions 包含缺少 task_id 的记录")
        if task_id in seen_edge:
            raise ValueError(f"edge predictions 包含重复 task_id: {task_id}")
        seen_edge.add(task_id)
        label_prediction(row)
        prediction = edge_prediction(row, cutoff)
        expected_review = prediction == "fault"
        if row.get("cloud_review") is not expected_review:
            inconsistent_routes.append(task_id)
        prompt_row = prompts.get(task_id)
        if prompt_row is None:
            missing_prompts.append(task_id)
            continue
        if prompt_row.get("qwen_input_encoding") != PROMPT_ENCODING:
            raise ValueError(f"{task_id} 的 qwen_input_encoding 不是 {PROMPT_ENCODING}")
        if not isinstance(prompt_row.get("prompt"), str) or not prompt_row["prompt"].strip():
            raise ValueError(f"{task_id} 缺少非空 prompt")
        forbidden = {"label", "ground_truth", "readings"}.intersection(prompt_row)
        if forbidden:
            raise ValueError(f"{task_id} 的云端输入泄漏字段: {sorted(forbidden)}")
    if missing_prompts:
        preview = ", ".join(missing_prompts[:5])
        raise ValueError(f"有 {len(missing_prompts)} 条边端记录找不到同 task_id 的 Qwen 输入: {preview}")
    if inconsistent_routes:
        preview = ", ".join(inconsistent_routes[:5])
        raise ValueError(
            f"有 {len(inconsistent_routes)} 条 cloud_review 与版本阈值不一致: {preview}"
        )
    return prompts


def select_edge_rows(
    edge_rows: list[dict[str, Any]],
    *,
    review_only: bool,
    pilot_per_label: int | None = None,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    selected = [row for row in edge_rows if not review_only or row.get("cloud_review") is True]
    if pilot_per_label is not None:
        if pilot_per_label <= 0:
            raise ValueError("pilot_per_label 必须大于 0")
        by_label = {0: [], 1: []}
        for row in selected:
            by_label[int(row["label"])].append(row)
        if not by_label[0] or not by_label[1]:
            raise ValueError("分层 pilot 需要同时包含正常和故障标签")
        selected = by_label[0][:pilot_per_label] + by_label[1][:pilot_per_label]
        selected.sort(key=lambda row: str(row["task_id"]))
    if max_tasks is not None:
        if max_tasks <= 0:
            raise ValueError("max_tasks 必须大于 0")
        selected = selected[:max_tasks]
    return selected


def parse_binary_cloud_output(raw_output: Any) -> tuple[str | None, str | None]:
    if not isinstance(raw_output, str):
        return None, "cloud response content is not a string"
    normalized = raw_output.strip()
    if normalized == "0":
        return "normal", None
    if normalized == "1":
        return "fault", None
    return None, f"expected exactly one digit 0 or 1, got {normalized!r}"


def build_request_body(model_id: str, prompt: str, *, seed: int) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "seed": int(seed),
        "stream": False,
        "cache_prompt": True,
        "grammar": BINARY_GRAMMAR,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@dataclass(frozen=True)
class CloudServiceInfo:
    endpoint: str
    model_id: str
    health: dict[str, Any]


class AI4ICloudReviewClient:
    """Small deterministic llama-server client for binary AI4I shadow review."""

    def __init__(self, endpoint: str, *, timeout_s: float = 180.0, seed: int = 42):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.seed = int(seed)
        self._service_info: CloudServiceInfo | None = None

    def _get_json(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.endpoint + path, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path} 未返回 JSON object")
        return payload

    def inspect_service(self) -> CloudServiceInfo:
        if self._service_info is None:
            health = self._get_json("/health")
            models = self._get_json("/v1/models")
            data = models.get("data")
            if not isinstance(data, list) or not data or not isinstance(data[0], dict):
                raise RuntimeError("/v1/models 未返回可用模型")
            model_id = data[0].get("id")
            if not isinstance(model_id, str) or not model_id:
                raise RuntimeError("/v1/models 的首个模型缺少 id")
            self._service_info = CloudServiceInfo(self.endpoint, model_id, health)
        return self._service_info

    def review(self, task_id: str, prompt: str) -> dict[str, Any]:
        service = self.inspect_service()
        url = self.endpoint + "/v1/chat/completions"
        body = build_request_body(service.model_id, prompt, seed=self.seed)
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            raw_payload = response.read().decode("utf-8")
        latency_ms = (time.perf_counter() - started) * 1000.0
        payload = json.loads(raw_payload)
        try:
            message = payload["choices"][0]["message"]
            raw_output = message.get("content")
            if raw_output in (None, ""):
                raw_output = message.get("reasoning_content", "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"chat/completions 返回结构无效: {payload}") from exc
        cloud_prediction, parser_error = parse_binary_cloud_output(raw_output)
        profile = parse_llama_profile(payload, client_latency_ms=latency_ms)
        profile["request_prompt_bytes"] = len(prompt.encode("utf-8"))
        return {
            "schema_version": CLOUD_RESULT_SCHEMA_VERSION,
            "task_id": task_id,
            "timestamp": utc_now(),
            "request": {
                "endpoint": service.endpoint,
                "model_id": service.model_id,
                "prompt_encoding": PROMPT_ENCODING,
                "prompt_sha256": sha256_text(prompt),
                "temperature": 0.0,
                "seed": self.seed,
                "max_tokens": 1,
                "grammar": BINARY_GRAMMAR,
                "thinking_enabled": False,
            },
            "raw_output": raw_output,
            "cloud_prediction": cloud_prediction,
            "output_valid": parser_error is None,
            "parser_error": parser_error,
            "latency_ms": round(latency_ms, 3),
            "usage": payload.get("usage", {}),
            "timings": payload.get("timings", {}),
            "profile": profile,
            "error": None,
        }


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return "http_error"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return "timeout"
        return "connection_error"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json_response"
    return "runtime_error"


def failed_result(
    task_id: str,
    prompt: str,
    *,
    endpoint: str,
    model_id: str | None,
    seed: int,
    exc: BaseException,
    attempt_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": CLOUD_RESULT_SCHEMA_VERSION,
        "task_id": task_id,
        "timestamp": utc_now(),
        "request": {
            "endpoint": endpoint.rstrip("/"),
            "model_id": model_id,
            "prompt_encoding": PROMPT_ENCODING,
            "prompt_sha256": sha256_text(prompt),
            "temperature": 0.0,
            "seed": int(seed),
            "max_tokens": 1,
            "grammar": BINARY_GRAMMAR,
            "thinking_enabled": False,
        },
        "raw_output": None,
        "cloud_prediction": None,
        "output_valid": False,
        "parser_error": None,
        "latency_ms": None,
        "usage": {},
        "timings": {},
        "profile": {},
        "error": {
            "kind": classify_exception(exc),
            "message": str(exc),
            "attempt_count": int(attempt_count),
        },
    }


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def latest_results(path: str | Path) -> dict[str, dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(target):
        if row.get("schema_version") != CLOUD_RESULT_SCHEMA_VERSION:
            raise ValueError(f"{target} 包含不兼容 schema_version")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{target} 包含缺少 task_id 的结果")
        latest[task_id] = row
    return latest


def binary_metrics(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("labels 与 predictions 长度不一致")
    tn = fp = fn = tp = 0
    for label, prediction in zip(labels, predictions):
        if label not in {"normal", "fault"} or prediction not in {"normal", "fault"}:
            raise ValueError("binary_metrics 只接受 normal/fault")
        if label == "normal" and prediction == "normal":
            tn += 1
        elif label == "normal" and prediction == "fault":
            fp += 1
        elif label == "fault" and prediction == "normal":
            fn += 1
        else:
            tp += 1
    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": total,
        "positives": sum(label == "fault" for label in labels),
        "accuracy": (tp + tn) / total if total else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_feedback_record(
    edge_row: dict[str, Any],
    cloud_result: dict[str, Any],
    *,
    cutoff: float,
    edge_model_version: str,
) -> dict[str, Any]:
    edge_value = edge_prediction(edge_row, cutoff)
    cloud_value = cloud_result.get("cloud_prediction") if cloud_result.get("output_valid") else None
    truth = label_prediction(edge_row)
    comparable = cloud_value in {"normal", "fault"}
    disagreement = edge_value != cloud_value if comparable else None
    edge_correct = edge_value == truth
    cloud_correct = cloud_value == truth if comparable else None
    review_useful = (not edge_correct and cloud_correct) if comparable else None
    cloud_regressed = (edge_correct and not cloud_correct) if comparable else None
    final_prediction = cloud_value if comparable else edge_value
    final_source = "cloud_review" if comparable else "edge_fallback"
    return {
        "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
        "sample_id": str(edge_row["task_id"]),
        "trace_id": None,
        "timestamp": cloud_result.get("timestamp") or utc_now(),
        "edge_prediction": edge_value,
        "cloud_prediction": cloud_value,
        "disagreement": disagreement,
        "cloud_changed_decision": disagreement if comparable else None,
        "review_useful": review_useful,
        "human_label": truth,
        "confirmed_outcome": truth,
        "label_source": "AI4I held-out test label",
        "available_at": utc_now(),
        "versions": {
            "edge_model_version": edge_model_version,
            "edge_policy": "failure_probability_above_cutoff_requests_cloud_review",
            "edge_failure_probability_cutoff": cutoff,
            "cloud_model_version": cloud_result.get("request", {}).get("model_id"),
            "cloud_prompt_version": PROMPT_ENCODING,
            "cloud_result_schema_version": CLOUD_RESULT_SCHEMA_VERSION,
        },
        "evidence": {
            "failure_probability": edge_row.get("failure_probability"),
            "cloud_review_requested": edge_row.get("cloud_review"),
            "cloud_output_valid": cloud_result.get("output_valid"),
            "cloud_raw_output": cloud_result.get("raw_output"),
            "cloud_parser_error": cloud_result.get("parser_error"),
            "cloud_error": cloud_result.get("error"),
            "cloud_latency_ms": cloud_result.get("latency_ms"),
            "edge_correct": edge_correct,
            "cloud_correct": cloud_correct,
            "cloud_regressed_decision": cloud_regressed,
            "final_prediction": final_prediction,
            "final_source": final_source,
            "final_correct": final_prediction == truth,
        },
    }


def evaluate_results(
    edge_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *,
    cutoff: float,
    edge_model_version: str,
    batch_scope: str,
    source_hashes: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_ids = {str(row["task_id"]) for row in target_rows}
    feedback: list[dict[str, Any]] = []
    for row in target_rows:
        result = results.get(str(row["task_id"]))
        if result is not None:
            feedback.append(
                build_feedback_record(
                    row,
                    result,
                    cutoff=cutoff,
                    edge_model_version=edge_model_version,
                )
            )

    attempted = len(feedback)
    valid = [row for row in feedback if row["cloud_prediction"] in {"normal", "fault"}]
    disagreements = [row for row in valid if row["disagreement"] is True]
    corrections = [row for row in valid if row["review_useful"] is True]
    regressions = [
        row for row in valid if row["evidence"].get("cloud_regressed_decision") is True
    ]
    errors = [results[task_id] for task_id in target_ids if task_id in results and results[task_id].get("error")]
    invalid_outputs = [
        results[task_id]
        for task_id in target_ids
        if task_id in results
        and not results[task_id].get("output_valid")
        and not results[task_id].get("error")
    ]
    latencies = [
        float(results[task_id]["latency_ms"])
        for task_id in target_ids
        if task_id in results and results[task_id].get("latency_ms") is not None
    ]
    timeout_count = sum(row.get("error", {}).get("kind") == "timeout" for row in errors)

    labels = [label_prediction(row) for row in edge_rows]
    edge_predictions = [edge_prediction(row, cutoff) for row in edge_rows]
    baseline = binary_metrics(labels, edge_predictions)
    classifier_reference = binary_metrics(
        labels,
        [classifier_prediction(row) for row in edge_rows],
    )

    all_review_rows = [row for row in edge_rows if row.get("cloud_review") is True]
    all_review_ids = {str(row["task_id"]) for row in all_review_rows}
    is_full_review_scope = target_ids == all_review_ids
    all_targets_attempted = target_ids.issubset(results)
    final_metrics = None
    if is_full_review_scope and all_targets_attempted:
        final_predictions: list[str] = []
        for row in edge_rows:
            task_id = str(row["task_id"])
            edge_value = edge_prediction(row, cutoff)
            result = results.get(task_id)
            if row.get("cloud_review") is True and result and result.get("output_valid"):
                final_predictions.append(str(result["cloud_prediction"]))
            else:
                final_predictions.append(edge_value)
        final_metrics = binary_metrics(labels, final_predictions)

    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "batch_scope": batch_scope,
        "source_hashes": source_hashes,
        "versions": {
            "edge_model_version": edge_model_version,
            "edge_failure_probability_cutoff": cutoff,
            "cloud_models": sorted(
                {
                    str(row.get("request", {}).get("model_id"))
                    for task_id, row in results.items()
                    if task_id in target_ids and row.get("request", {}).get("model_id")
                }
            ),
            "cloud_prompt_version": PROMPT_ENCODING,
            "feedback_schema_version": FEEDBACK_SCHEMA_VERSION,
        },
        "dataset": {
            "test_samples": len(edge_rows),
            "positive_samples": sum(row.get("label") == 1 for row in edge_rows),
            "post_edge_review_tasks": len(all_review_rows),
            "batch_target_tasks": len(target_rows),
        },
        "cloud_execution": {
            "attempted_tasks": attempted,
            "unattempted_tasks": len(target_rows) - attempted,
            "valid_outputs": len(valid),
            "valid_output_rate": _rate(len(valid), attempted),
            "invalid_outputs": len(invalid_outputs),
            "request_errors": len(errors),
            "timeout_count": timeout_count,
            "timeout_rate": _rate(timeout_count, attempted),
            "latency_ms": {
                "samples": len(latencies),
                "mean": sum(latencies) / len(latencies) if latencies else None,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "max": max(latencies) if latencies else None,
            },
        },
        "edge_cloud_comparison": {
            "comparable_predictions": len(valid),
            "agreement_count": len(valid) - len(disagreements),
            "agreement_rate": _rate(len(valid) - len(disagreements), len(valid)),
            "disagreement_count": len(disagreements),
            "disagreement_rate": _rate(len(disagreements), len(valid)),
            "cloud_changed_decision_count": len(disagreements),
            "cloud_changed_decision_rate": _rate(len(disagreements), len(valid)),
            "review_useful_count": len(corrections),
            "review_useful_rate": _rate(len(corrections), len(valid)),
            "cloud_regression_count": len(regressions),
            "cloud_regression_rate": _rate(len(regressions), len(valid)),
            "net_corrections": len(corrections) - len(regressions),
        },
        "edge_only_baseline": baseline,
        "edge_model_classifier_reference": {
            "decision_threshold": 0.5,
            **classifier_reference,
        },
        "final_system": final_metrics,
        "final_system_available": final_metrics is not None,
        "final_system_note": (
            "All Post-edge review tasks were attempted; invalid/error calls fall back to edge."
            if final_metrics is not None
            else "Final-system metrics require a complete attempt over every Post-edge review task."
        ),
    }
    if final_metrics is not None:
        metrics["improvement_vs_edge_only"] = {
            key: final_metrics[key] - baseline[key]
            for key in ("accuracy", "precision", "recall", "f1")
        }
    else:
        metrics["improvement_vs_edge_only"] = None
    return metrics, feedback


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown_report(metrics: dict[str, Any]) -> str:
    dataset = metrics["dataset"]
    cloud = metrics["cloud_execution"]
    comparison = metrics["edge_cloud_comparison"]
    edge = metrics["edge_only_baseline"]
    classifier = metrics["edge_model_classifier_reference"]
    final = metrics.get("final_system")
    latency = cloud["latency_ms"]
    lines = [
        "# AI4I Edge-Cloud Review Report",
        "",
        f"- Schema: `{metrics['schema_version']}`",
        f"- Generated at: `{metrics['generated_at']}`",
        f"- Batch scope: `{metrics['batch_scope']}`",
        f"- Held-out test samples: {dataset['test_samples']}",
        f"- Post-edge review tasks: {dataset['post_edge_review_tasks']}",
        f"- This batch targets: {dataset['batch_target_tasks']}",
        "",
        "## Cloud execution",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Attempted | {cloud['attempted_tasks']} |",
        f"| Valid outputs | {cloud['valid_outputs']} |",
        f"| Valid-output rate | {_format_metric(cloud['valid_output_rate'])} |",
        f"| Invalid outputs | {cloud['invalid_outputs']} |",
        f"| Request errors | {cloud['request_errors']} |",
        f"| Timeout rate | {_format_metric(cloud['timeout_rate'])} |",
        f"| Mean latency (ms) | {_format_metric(latency['mean'])} |",
        f"| P50 latency (ms) | {_format_metric(latency['p50'])} |",
        f"| P95 latency (ms) | {_format_metric(latency['p95'])} |",
        "",
        "## Edge-cloud comparison",
        "",
        "Cloud output is a reviewer signal, not ground truth. Agreement and correctness are reported separately.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Comparable predictions | {comparison['comparable_predictions']} |",
        f"| Agreement count | {comparison['agreement_count']} |",
        f"| Agreement rate | {_format_metric(comparison['agreement_rate'])} |",
        f"| Disagreement count | {comparison['disagreement_count']} |",
        f"| Disagreement rate | {_format_metric(comparison['disagreement_rate'])} |",
        f"| Useful corrections | {comparison['review_useful_count']} |",
        f"| Cloud regressions | {comparison['cloud_regression_count']} |",
        f"| Net corrections | {comparison['net_corrections']} |",
        "",
        "## Label-based metrics",
        "",
        "| System | Accuracy | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| MoE classifier reference (0.5) | {_format_metric(classifier['accuracy'])} | {_format_metric(classifier['precision'])} | {_format_metric(classifier['recall'])} | {_format_metric(classifier['f1'])} |",
        f"| Edge-only operational (HHK cutoff) | {_format_metric(edge['accuracy'])} | {_format_metric(edge['precision'])} | {_format_metric(edge['recall'])} | {_format_metric(edge['f1'])} |",
    ]
    if final is None:
        lines.append("| Edge + cloud review | N/A | N/A | N/A | N/A |")
        lines.extend(["", f"> {metrics['final_system_note']}"])
    else:
        lines.append(
            f"| Edge + cloud review | {_format_metric(final['accuracy'])} | "
            f"{_format_metric(final['precision'])} | {_format_metric(final['recall'])} | "
            f"{_format_metric(final['f1'])} |"
        )
        improvement = metrics["improvement_vs_edge_only"]
        lines.extend(
            [
                "",
                "## Improvement versus Edge-only",
                "",
                f"- Accuracy delta: {_format_metric(improvement['accuracy'])}",
                f"- Precision delta: {_format_metric(improvement['precision'])}",
                f"- Recall delta: {_format_metric(improvement['recall'])}",
                f"- F1 delta: {_format_metric(improvement['f1'])}",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `disagreement` only means the two models differ.",
            "- `review_useful` requires the AI4I held-out label: edge wrong and cloud correct.",
            "- The 0.5 classifier reference is not the deployed Edge-only decision; HHK uses its versioned conservative cutoff.",
            "- A failed or invalid cloud response falls back to the edge prediction.",
            "- Accuracy alone is insufficient for this imbalanced dataset; inspect recall and F1.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(target)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def write_report(path: str | Path, metrics: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(render_markdown_report(metrics), encoding="utf-8")
    temporary.replace(target)
