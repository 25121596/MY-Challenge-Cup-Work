from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .accuracy_metrics import evaluate_prediction_rows
from .io_utils import read_jsonl, write_json, write_jsonl
from .output_validation import extract_json_object
from .student_training import render_sample


def _require_dependencies():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "学生评估需要额外依赖。请执行："
            "pip install -r requirements-distillation.txt"
        ) from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def _prompt(sample: dict[str, Any]) -> str:
    prompt_sample = dict(sample)
    prompt_sample["output"] = {}
    text = render_sample(prompt_sample)
    return text.rsplit("{}", 1)[0]


def compare_base_and_lora(
    model_path: str,
    lora_path: str,
    test_file: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = _require_dependencies()
    samples = read_jsonl(test_file)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs.update(device_map="auto", torch_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    def generate(text: str, *, adapter_enabled: bool) -> tuple[str, float]:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        context = model.disable_adapter() if not adapter_enabled else _nullcontext()
        with context, torch.no_grad():
            start = time.perf_counter()
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
        new_tokens = output[0][encoded["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        prompt = _prompt(sample)
        base_raw, base_ms = generate(prompt, adapter_enabled=False)
        lora_raw, lora_ms = generate(prompt, adapter_enabled=True)
        truth = sample.get("ground_truth")
        if not isinstance(truth, dict) and sample.get("target_source") == "ground_truth":
            truth = sample.get("output")
        rows.append({
            "task_id": sample.get("task_id"),
            "task_type": sample.get("task_type"),
            "ground_truth": truth,
            "teacher_output": sample.get("teacher_output"),
            "base_prediction": extract_json_object(base_raw),
            "lora_prediction": extract_json_object(lora_raw),
            "base_raw": base_raw,
            "lora_raw": lora_raw,
            "base_latency_ms": base_ms,
            "lora_latency_ms": lora_ms,
        })
        print(f"[{index}/{len(samples)}] {sample.get('task_id')}")

    write_jsonl(output_path, rows)
    base_report = evaluate_prediction_rows([
        {"prediction": row["base_prediction"], "ground_truth": row["ground_truth"]}
        for row in rows
    ])
    lora_report = evaluate_prediction_rows([
        {"prediction": row["lora_prediction"], "ground_truth": row["ground_truth"]}
        for row in rows
    ])
    report = {
        "model_path": model_path,
        "lora_path": lora_path,
        "sample_count": len(rows),
        "base": base_report.to_dict(),
        "lora": lora_report.to_dict(),
        "accuracy_claim_allowed": bool(
            base_report.category_accuracy is not None
            and lora_report.category_accuracy is not None
        ),
        "note": (
            "只有测试样本含独立ground_truth时，才能据此讨论真实判断准确率；"
            "否则仅能比较JSON有效率和教师一致性。"
        ),
    }
    write_json(report_path, report)
    return report


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
