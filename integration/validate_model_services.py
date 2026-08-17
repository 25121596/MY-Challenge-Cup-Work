from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    CLOUD_ENDPOINT,
    EDGE_ENDPOINT,
    MAX_TOKENS,
    REQUEST_TIMEOUT_S,
    ROOT,
    TEMPERATURE,
)
from model_client import ModelClient
from prompt_builder import build_common_prompt
from task_io import load_task, save_json


DEFAULT_INPUT = ROOT / "samples" / "test_task.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "model_services_validation.json"


def parse_args():
    parser = argparse.ArgumentParser(description="双模型基础调用测试")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main():
    args = parse_args()
    edge = ModelClient(EDGE_ENDPOINT, REQUEST_TIMEOUT_S)
    cloud = ModelClient(CLOUD_ENDPOINT, REQUEST_TIMEOUT_S)
    task = load_task(args.input)
    prompt = build_common_prompt(task)

    print(f"EDGE:  {edge.health()} | {edge.get_model_id()}")
    print(f"CLOUD: {cloud.health()} | {cloud.get_model_id()}")

    edge_result = edge.chat(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    cloud_result = cloud.chat(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    result = {
        "task": task,
        "edge": edge_result,
        "cloud": cloud_result,
    }
    save_json(args.output, result)

    print("\nEDGE:")
    print(edge_result["answer"])
    print("\nCLOUD:")
    print(cloud_result["answer"])
    print(f"\n保存到: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
