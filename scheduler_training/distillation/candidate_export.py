from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io_utils import load_tasks, write_json


@dataclass(frozen=True)
class CandidateExportReport:
    input_count: int
    completed_count: int
    selected_count: int
    selected_routes: tuple[str, ...]
    output_path: str
    incomplete_count: int


def export_distillation_candidates(
    input_path: str | Path,
    output_path: str | Path,
    *,
    strategy: str = "rule",
    ppo_model: str | None = None,
    routes: tuple[str, ...] = ("CLOUD", "COLLABORATIVE"),
    max_steps: int = 200000,
    seed: int = 42,
) -> CandidateExportReport:
    """运行Phase4.5调度并按最终路径导出原始hhk候选任务。

    这里只选择蒸馏候选，不把仿真准确率当成教师标签。
    """
    from scheduler import (
        AlwaysCloudScheduler,
        AlwaysLocalFirstScheduler,
        PPOScheduler,
        RuleScheduler,
    )
    from simulator import CloudEdgeSimulator

    strategy_key = strategy.strip().lower()
    if strategy_key == "rule":
        scheduler = RuleScheduler()
    elif strategy_key == "fixed_cloud":
        scheduler = AlwaysCloudScheduler()
    elif strategy_key == "fixed_local_first":
        scheduler = AlwaysLocalFirstScheduler()
    elif strategy_key == "ppo":
        if not ppo_model:
            raise ValueError("strategy=ppo时必须提供ppo_model")
        scheduler = PPOScheduler(ppo_model)
    else:
        raise ValueError(
            "strategy必须是rule、fixed_cloud、fixed_local_first或ppo"
        )

    raw_tasks = load_tasks(input_path)
    simulator = CloudEdgeSimulator(
        seed=seed,
        scheduler=scheduler,
        task_source="edge_input",
        edge_input_path=str(input_path),
    )
    simulator.run_until_complete(max_steps=max_steps)
    completed = simulator.get_completed_tasks()
    route_set = {route.upper() for route in routes}
    selected_ids: set[str] = set()
    if not completed.empty:
        for _, row in completed.iterrows():
            if str(row.get("route", "")).upper() in route_set:
                external_id = str(row.get("external_task_id", "")).strip()
                if external_id:
                    selected_ids.add(external_id)

    selected = [
        task for task in raw_tasks
        if str(task.get("task_id", "")).strip() in selected_ids
    ]
    write_json(output_path, selected)
    report = CandidateExportReport(
        input_count=len(raw_tasks),
        completed_count=len(completed),
        selected_count=len(selected),
        selected_routes=tuple(sorted(route_set)),
        output_path=str(output_path),
        incomplete_count=max(0, len(raw_tasks) - len(completed)),
    )
    write_json(Path(output_path).with_suffix(".manifest.json"), asdict(report))
    return report


# Compatibility for callers created before the current naming cleanup.
export_phase3_candidates = export_distillation_candidates
