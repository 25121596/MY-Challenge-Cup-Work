from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class DistillationDataError(ValueError):
    """蒸馏输入或中间文件格式错误。"""


def read_json(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到文件：{file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise DistillationDataError(f"文件不是合法JSON：{file_path}；{exc}") from exc


def extract_task_objects(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        rows = [item for item in raw if isinstance(item, dict)]
    elif isinstance(raw, dict) and isinstance(raw.get("inputs"), list):
        rows = [item for item in raw["inputs"] if isinstance(item, dict)]
    elif isinstance(raw, dict) and "task_id" in raw:
        rows = [raw]
    else:
        raise DistillationDataError(
            "任务输入应为任务数组、单条任务，或包含inputs数组的对象。"
        )
    if not rows:
        raise DistillationDataError("任务输入中没有JSON对象。")
    return rows


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    return extract_task_objects(read_json(path))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到JSONL文件：{file_path}")
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DistillationDataError(
                    f"{file_path}第{line_number}行不是合法JSON：{exc}"
                ) from exc
            if not isinstance(value, dict):
                raise DistillationDataError(
                    f"{file_path}第{line_number}行必须是JSON对象。"
                )
            rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with file_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
            count += 1
    return count


def load_ground_truth(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        truth = row.get("ground_truth", row.get("label", row.get("output")))
        if not task_id or not isinstance(truth, dict):
            raise DistillationDataError(
                "ground truth每行必须包含task_id和ground_truth对象。"
            )
        if task_id in result:
            raise DistillationDataError(f"ground truth包含重复task_id：{task_id}")
        result[task_id] = {
            "ground_truth": truth,
            "label_source": str(row.get("label_source", "ground_truth")),
        }
    return result
