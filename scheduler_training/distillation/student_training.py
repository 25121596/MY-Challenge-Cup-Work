from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_json


@dataclass(frozen=True)
class StudentTrainingConfig:
    model_path: str
    train_file: str
    output_dir: str
    epochs: float = 2.0
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ()
    load_in_4bit: bool = False


def _require_training_dependencies():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "学生LoRA训练需要额外依赖。请执行："
            "pip install -r requirements-distillation.txt"
        ) from exc
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def render_sample(sample: dict[str, Any]) -> str:
    return (
        "### Instruction\n"
        f"{sample['instruction']}\n\n"
        "### Input\n"
        f"{json.dumps(sample['input'], ensure_ascii=False, sort_keys=True)}\n\n"
        "### Response\n"
        f"{json.dumps(sample['output'], ensure_ascii=False, sort_keys=True)}"
    )


def _auto_target_modules(model) -> list[str]:
    preferred = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    found = sorted({name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name.rsplit(".", 1)[-1] in preferred})
    if found:
        return found
    fallback = sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules() if module.__class__.__name__ == "Linear"})
    if not fallback:
        raise ValueError("无法自动识别LoRA目标层，请通过--target-modules显式指定。")
    return fallback[-8:]


def train_student_lora(config: StudentTrainingConfig) -> dict[str, Any]:
    """参数化LoRA训练。当前管线不假设具体边端模型，运行前必须显式给出模型路径。"""
    deps = _require_training_dependencies()
    torch = deps["torch"]
    rows = read_jsonl(config.train_file)
    if not rows:
        raise ValueError("训练集为空。")

    tokenizer = deps["AutoTokenizer"].from_pretrained(
        config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16
    if config.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    model = deps["AutoModelForCausalLM"].from_pretrained(
        config.model_path, **load_kwargs
    )
    model.config.use_cache = False
    target_modules = list(config.target_modules) or _auto_target_modules(model)
    lora_config = deps["LoraConfig"](
        task_type=deps["TaskType"].CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = deps["get_peft_model"](model, lora_config)

    class JsonlDataset(torch.utils.data.Dataset):
        def __init__(self, samples: list[dict[str, Any]]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            text = render_sample(self.samples[index])
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = deps["TrainingArguments"](
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        fp16=bool(torch.cuda.is_available()),
        remove_unused_columns=False,
    )
    trainer = deps["Trainer"](
        model=model,
        args=args,
        train_dataset=JsonlDataset(rows),
    )
    result = trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "config": asdict(config),
        "target_modules": target_modules,
        "sample_count": len(rows),
        "train_metrics": result.metrics,
        "warning": "该适配器尚未自动部署到边端；必须先完成独立准确率和资源验收。",
    }
    write_json(output_dir / "phase4_training_metadata.json", metadata)
    return metadata
