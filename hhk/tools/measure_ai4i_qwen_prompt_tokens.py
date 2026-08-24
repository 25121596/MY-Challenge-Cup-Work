#!/usr/bin/env python3
"""Measure legacy and compact AI4I Qwen prompts with llama-server's tokenizer."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge-io-protocol"))

from ai4i_qwen_encoding import encode_ai4i_qwen_review  # noqa: E402
from edge_inference import build_prompt  # noqa: E402


def tokenize(endpoint: str, content: str) -> int:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/tokenize",
        data=json.dumps({"content": content}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        result = json.loads(response.read())
    tokens = result.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("llama-server /tokenize response did not contain tokens")
    return len(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data" / "standard_inputs" / "ai4i_milling_inference_v1.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    task = rows[args.index]
    compact = encode_ai4i_qwen_review(task)
    legacy = build_prompt(task["task_type"], task)
    compact_tokens = tokenize(args.endpoint, compact)
    legacy_tokens = tokenize(args.endpoint, legacy)
    result = {
        "input": str(args.input),
        "index": args.index,
        "task_id": task.get("task_id"),
        "encoder": "ai4i_qwen_8d_v1",
        "legacy": {"characters": len(legacy), "tokens": legacy_tokens},
        "compact": {"characters": len(compact), "tokens": compact_tokens},
        "token_reduction_percent": round((1 - compact_tokens / legacy_tokens) * 100, 2),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
