from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_task(path: str | Path) -> dict[str, Any]:
    """读取单条标准任务，列表时取第一条。"""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        if not payload:
            raise ValueError("输入 JSON 列表为空")
        task = payload[0]
    elif isinstance(payload, dict):
        if isinstance(payload.get("inputs"), list):
            if not payload["inputs"]:
                raise ValueError("inputs 列表为空")
            task = payload["inputs"][0]
        else:
            task = payload
    else:
        raise ValueError("输入 JSON 顶层必须是 object 或 list")

    if not isinstance(task, dict) or "data" not in task:
        raise ValueError("无法读取标准任务：任务必须为 object 且包含 data")
    return task


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8-sig")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
