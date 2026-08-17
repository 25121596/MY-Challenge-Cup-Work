from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import statistics
import threading
import time
from typing import Any


TRACE_SCHEMA_VERSION = "cloud-edge-real-trace/2.0"


def _first_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_int(mapping: dict[str, Any], *keys: str) -> int | None:
    value = _first_number(mapping, *keys)
    return None if value is None else int(value)


def parse_llama_profile(payload: dict[str, Any], *, client_latency_ms: float) -> dict[str, Any]:
    """Normalize llama.cpp/OpenAI-compatible usage and timing fields.

    llama.cpp currently exposes OpenAI-style ``usage`` plus a ``timings`` object
    on non-streaming completions.  Different builds use slightly different
    names, so this parser is intentionally defensive.  Missing values remain
    ``None``; Integration never fabricates TTFT or throughput.
    """

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    timings = payload.get("timings") if isinstance(payload.get("timings"), dict) else {}

    prompt_tokens = _first_int(usage, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = _first_int(timings, "prompt_n", "tokens_evaluated")

    completion_tokens = _first_int(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _first_int(timings, "predicted_n", "tokens_predicted")

    total_tokens = _first_int(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    prompt_ms = _first_number(timings, "prompt_ms", "prompt_time_ms")
    completion_ms = _first_number(timings, "predicted_ms", "completion_ms", "decode_ms")
    prompt_tps = _first_number(timings, "prompt_per_second", "prompt_tokens_per_second")
    completion_tps = _first_number(
        timings,
        "predicted_per_second",
        "completion_tokens_per_second",
        "tokens_per_second",
    )
    cached_tokens = _first_int(timings, "cache_n", "tokens_cached")
    if cached_tokens is None:
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        cached_tokens = _first_int(details, "cached_tokens")
    completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    reasoning_tokens = _first_int(completion_details, "reasoning_tokens")

    # Non-streaming llama.cpp commonly reports prefill/decode timing but not a
    # true client-observed time-to-first-token.  Preserve a real TTFT only when
    # a build explicitly returns one.
    ttft_ms = _first_number(
        timings,
        "ttft_ms",
        "time_to_first_token_ms",
        "first_token_ms",
    )
    ttft_source = "server_reported" if ttft_ms is not None else "not_reported_non_streaming"

    return {
        "client_latency_ms": float(client_latency_ms),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "prompt_ms": prompt_ms,
        "completion_ms": completion_ms,
        "prompt_tokens_per_s": prompt_tps,
        "completion_tokens_per_s": completion_tps,
        "ttft_ms": ttft_ms,
        "ttft_source": ttft_source,
        "server_timings": deepcopy(timings),
        "server_usage": deepcopy(usage),
    }


class RuntimeProfiler:
    """Background state sampler used only for observability.

    Sampling is decoupled from the model worker so resource/network collection
    is not inserted directly into the measured inference critical section.
    """

    def __init__(self, state_provider: Any, runtime: Any, *, interval_s: float = 0.5) -> None:
        self.state_provider = state_provider
        self.runtime = runtime
        self.interval_s = max(0.1, float(interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self._timeline: list[dict[str, Any]] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="runtime-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.interval_s * 2.0))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        try:
            state = self.state_provider.get_state(self.runtime, force_refresh=True)
        except Exception:
            return
        row = {
            "wall_time_s": time.time(),
            "monotonic_s": time.monotonic(),
            "state": deepcopy(state),
        }
        with self._lock:
            self._latest = row
            self._timeline.append(row)
            # Bounded timeline: enough for a profiling run without unbounded RAM.
            if len(self._timeline) > 10000:
                del self._timeline[:1000]

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(self._latest)

    def timeline(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._timeline)


@dataclass
class NumericSummary:
    count: int
    mean: float | None
    p50: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None


def summarize_numeric(values: list[float]) -> NumericSummary:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return NumericSummary(0, None, None, None, None, None)

    def percentile(p: float) -> float:
        if len(clean) == 1:
            return clean[0]
        pos = (len(clean) - 1) * p
        lower = int(pos)
        upper = min(len(clean) - 1, lower + 1)
        weight = pos - lower
        return clean[lower] * (1.0 - weight) + clean[upper] * weight

    return NumericSummary(
        count=len(clean),
        mean=statistics.fmean(clean),
        p50=percentile(0.50),
        p95=percentile(0.95),
        minimum=clean[0],
        maximum=clean[-1],
    )


def build_profile_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "record_count": len(records),
        "completed": sum(r.get("status") == "completed" for r in records),
        "timeout": sum(r.get("status") == "timeout" for r in records),
        "failed": sum(r.get("status") == "failed" for r in records),
        "by_action": {},
        "by_final_route": {},
        "by_task_type": {},
    }
    for bucket_name, key in (
        ("by_action", "action_name"),
        ("by_final_route", "final_route"),
        ("by_task_type", "task_type"),
    ):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in records:
            groups.setdefault(str(row.get(key) or "UNKNOWN"), []).append(row)
        for group, rows in groups.items():
            summary[bucket_name][group] = {
                "count": len(rows),
                "completed": sum(r.get("status") == "completed" for r in rows),
                "timeout": sum(r.get("status") == "timeout" for r in rows),
                "total_wall_s": summarize_numeric(
                    [r.get("real_total_wall_s") for r in rows if r.get("real_total_wall_s") is not None]
                ).__dict__,
                "edge_inference_s": summarize_numeric(
                    [r.get("real_edge_stage_s") for r in rows if r.get("real_edge_stage_s") is not None]
                ).__dict__,
                "cloud_inference_s": summarize_numeric(
                    [r.get("real_cloud_stage_s") for r in rows if r.get("real_cloud_stage_s") is not None]
                ).__dict__,
            }
    return summary
