from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


def register_student_candidate(
    registry_path: str | Path,
    *,
    model_name: str,
    base_model: str,
    adapter_path: str,
    evaluation_report: str | None = None,
    status: str = "candidate",
) -> dict[str, Any]:
    """仅登记候选模型，不自动替换未知边端模型。"""
    path = Path(registry_path)
    registry = read_json(path) if path.is_file() else {"active_model": None, "candidates": []}
    item = {
        "model_name": model_name,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "evaluation_report": evaluation_report,
        "status": status,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    registry.setdefault("candidates", []).append(item)
    write_json(path, registry)
    return item
