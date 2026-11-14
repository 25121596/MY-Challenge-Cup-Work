#!/usr/bin/env python3
"""Measure the ESW direct-cloud path with a fixed chat-token budget."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import platform
import statistics
import threading
import time
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = (
    "Reply with exactly one lowercase English word. Use plain text only. "
    "Do not return JSON, Markdown, code, labels, or explanations."
)
USER_PREFIX = "Machine telemetry review request: "
ESW_DIRECT_CLOUD_B64 = base64.b64encode(bytes.fromhex("010A0000")).decode("ascii")


def post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tokenize(model_url: str, text: str) -> int:
    return len(post_json(f"{model_url}/tokenize", {"content": text})["tokens"])


def build_messages(model_url: str, target_tokens: int) -> tuple[list[dict[str, str]], int]:
    words: list[str] = []
    for _ in range(target_tokens * 2):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PREFIX + " ".join(words)},
        ]
        prompt = post_json(f"{model_url}/apply-template", {"messages": messages})["prompt"]
        count = tokenize(model_url, prompt)
        if count == target_tokens:
            return messages, count
        if count > target_tokens:
            break
        words.append("telemetry")
    raise RuntimeError(f"unable to construct exactly {target_tokens} input tokens")


def rss_kb(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError):
        pass
    return None


def mem_available_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except FileNotFoundError:
        pass
    return None


class MemorySampler:
    def __init__(self, pids: list[int]) -> None:
        self.pids = pids
        self.start_rss = {str(pid): rss_kb(pid) for pid in pids}
        self.peak_rss = dict(self.start_rss)
        self.start_available = mem_available_kb()
        self.min_available = self.start_available
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.02):
            for pid in self.pids:
                value = rss_kb(pid)
                key = str(pid)
                if value is not None and (self.peak_rss[key] is None or value > self.peak_rss[key]):
                    self.peak_rss[key] = value
            available = mem_available_kb()
            if available is not None and (self.min_available is None or available < self.min_available):
                self.min_available = available

    def __enter__(self) -> "MemorySampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=1.0)


def stream_chat(url: str, payload: dict) -> tuple[float, float, float, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    first_byte_ms: float | None = None
    first_token_ms: float | None = None
    text_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=120.0) as response:
        first = response.read(1)
        if first:
            first_byte_ms = (time.perf_counter() - started) * 1000.0
        pending = first
        while True:
            line = pending + response.readline() if pending else response.readline()
            pending = b""
            if not line:
                break
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data = decoded[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            choice = event.get("choices", [{}])[0]
            content = choice.get("delta", {}).get("content", "")
            if content:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000.0
                text_parts.append(content)
    completed_ms = (time.perf_counter() - started) * 1000.0
    if first_byte_ms is None or first_token_ms is None:
        raise RuntimeError("stream ended without a valid generated token")
    return first_byte_ms, first_token_ms, completed_ms, "".join(text_parts)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "p0_ms": min(values),
        "p50_ms": percentile(values, 0.50),
        "p75_ms": percentile(values, 0.75),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model-url", default="http://127.0.0.1:8080")
    parser.add_argument("--route", choices=("cloud", "direct"), default="cloud")
    parser.add_argument("--model-pid", type=int, required=True)
    parser.add_argument("--cloud-pid", type=int)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--input-tokens", type=int, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    messages, actual_input_tokens = build_messages(args.model_url, args.input_tokens)
    envelope_url = f"{args.cloud_url}/api/v1/protocol/esw"
    chat_url = f"{args.cloud_url}/v1/chat/completions" if args.route == "cloud" else f"{args.model_url}/v1/chat/completions"
    base_envelope = {
        "schema_version": "edge-esw-envelope/v1",
        "device_id": "bench-direct-cloud",
        "trace_id": "cloud-alloffload-benchmark",
        "esw_payload_b64": ESW_DIRECT_CLOUD_B64,
    }
    chat_payload = {
        "model": "Qwen3-1.7B-IQ4_XS",
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_output_tokens,
        "seed": 42,
        "stream": True,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def one_request(index: int, measured: bool) -> dict[str, object]:
        request_id = f"cloud-alloffload-{'measure' if measured else 'warmup'}-{index:03d}"
        envelope = {**base_envelope, "request_id": request_id}
        pids = [args.model_pid]
        if args.cloud_pid is not None:
            pids.append(args.cloud_pid)
        sampler = MemorySampler(pids)
        started = time.perf_counter()
        with sampler:
            if args.route == "cloud":
                admission = post_json(envelope_url, envelope)
                admission_ms = (time.perf_counter() - started) * 1000.0
                if admission["esw_status"] != "ESW_OK" or admission["cloud_action"] != "direct_cloud":
                    raise RuntimeError(f"direct-cloud ESW admission failed: {admission}")
                esw_word_hex = admission["raw_word_hex"]
            else:
                admission_ms = 0.0
                esw_word_hex = None
            first_byte_ms, chat_ttft_ms, chat_total_ms, output = stream_chat(chat_url, chat_payload)
        total_ms = (time.perf_counter() - started) * 1000.0
        output_tokens = tokenize(args.model_url, output)
        stripped = output.strip()
        output_is_json = False
        if stripped:
            try:
                json.loads(stripped)
                output_is_json = True
            except json.JSONDecodeError:
                pass
        return {
            "request_id": request_id,
            "input_tokens": actual_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "output_tokens": output_tokens,
            "esw_word_hex": esw_word_hex,
            "esw_admission_ms": round(admission_ms, 3),
            "chat_first_byte_ms": round(first_byte_ms, 3),
            "chat_ttft_ms": round(chat_ttft_ms, 3),
            "e2e_ttft_ms": round(admission_ms + chat_ttft_ms, 3),
            "chat_total_ms": round(chat_total_ms, 3),
            "e2e_total_ms": round(total_ms, 3),
            "model_rss_start_kb": sampler.start_rss[str(args.model_pid)],
            "model_rss_peak_kb": sampler.peak_rss[str(args.model_pid)],
            "cloud_rss_start_kb": sampler.start_rss.get(str(args.cloud_pid)) if args.cloud_pid else None,
            "cloud_rss_peak_kb": sampler.peak_rss.get(str(args.cloud_pid)) if args.cloud_pid else None,
            "mem_available_start_kb": sampler.start_available,
            "mem_available_min_kb": sampler.min_available,
            "output_is_json": output_is_json,
            "output_text": output.replace("\n", " ").strip(),
        }

    for index in range(args.warmups):
        one_request(index + 1, measured=False)
    rows = [one_request(index + 1, measured=True) for index in range(args.samples)]
    csv_path = args.output_dir / "cloud_alloffload_raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    e2e_ttft = [float(row["e2e_ttft_ms"]) for row in rows]
    e2e_total = [float(row["e2e_total_ms"]) for row in rows]
    model_peak_kb = max(int(row["model_rss_peak_kb"] or 0) for row in rows)
    cloud_peak_kb = max(int(row["cloud_rss_peak_kb"] or 0) for row in rows)
    json_outputs = sum(bool(row["output_is_json"]) for row in rows)
    report = args.output_dir / "cloud_alloffload_report.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Cloud All-Offload Benchmark\n\n")
        route = "ESW-v1 DIRECT_CLOUD (`0x010A0000`) then cloud chat completion" if args.route == "cloud" else "direct llama-server chat completion"
        handle.write(f"- Route: {route}\n")
        handle.write("- ESW is a deterministic 4-byte control word; the one-token model output only validates SSE delivery\n")
        handle.write(f"- Samples: {args.samples}; warmups excluded: {args.warmups}\n")
        handle.write(f"- Input tokens: {actual_input_tokens} (chat template included)\n")
        handle.write(f"- Output cap: {args.max_output_tokens}; observed mean: {statistics.fmean(int(row['output_tokens']) for row in rows):.2f}\n")
        handle.write("- Prompt cache: disabled (`cache_prompt=false`)\n")
        handle.write("- Qwen thinking: disabled (`chat_template_kwargs.enable_thinking=false`)\n")
        handle.write(f"- JSON-shaped model outputs: {json_outputs}/{args.samples}\n")
        handle.write(f"- Model RSS peak: {model_peak_kb / 1024.0:.1f} MiB; cloud API RSS peak: {cloud_peak_kb / 1024.0:.1f} MiB\n\n")
        for name, values in (("End-to-end TTFT", e2e_ttft), ("End-to-end total", e2e_total)):
            stats = summarize(values)
            handle.write(f"## {name}\n\n")
            for key, value in stats.items():
                handle.write(f"- {key}: {value:.3f}\n")
            handle.write("\n")
        handle.write("This is a localhost proxy measurement. It excludes WAN RTT, TLS, remote queueing, and a remote cloud model.\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
