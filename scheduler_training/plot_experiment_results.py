"""为 Scheduler Training 正式场景实验生成中文报告图表。

本模块只读取评测与训练结果，不参与 PPO 决策、奖励计算或仿真状态更新。
所有调度指标均属于场景仿真结果，不应描述为真实系统测量值。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PROJECT_DIR = Path(__file__).resolve().parent

SCENARIO_LABELS = {
    "normal_load": "常规负载",
    "edge_burst": "边端突发负载",
    "weak_network": "弱网波动",
}
SCENARIO_ORDER = tuple(SCENARIO_LABELS)

STRATEGY_LABELS = {
    "EDGE_ONLY": "纯边端基准",
    "PRE_ROUTE_ONLY": "仅前置路由",
    "POST_EDGE_ONLY": "仅边后规则",
    "TWO_LAYER": "两层调度",
}
STRATEGY_ORDER = tuple(STRATEGY_LABELS)
STRATEGY_COLORS = {
    "EDGE_ONLY": "#7A8491",
    "PRE_ROUTE_ONLY": "#3973B7",
    "POST_EDGE_ONLY": "#E69F35",
    "TWO_LAYER": "#138A72",
}

TASK_TYPE_LABELS = {
    "fault_diagnosis": "故障诊断",
    "quality_inspection": "质量检测",
    "safety_check": "安全检查",
    "data_analysis": "数据分析",
    "status_monitoring": "状态监测",
    "predictive_maintenance": "预测性维护",
    "generic_anomaly": "通用异常检测",
    "video_anomaly": "视频异常检测",
}
TASK_TYPE_ORDER = tuple(TASK_TYPE_LABELS)

CHART_STEMS = (
    "01_completion_rate",
    "02_p95_latency",
    "03_route_composition",
    "04_upload_volume",
    "05_training_reward",
    "06_task_type_cloud_heatmap",
    "07_expected_accuracy",
)


@dataclass(frozen=True)
class ResultTables:
    """绘图所需的四张输入表。"""

    summary: pd.DataFrame
    detailed: pd.DataFrame
    by_task_type: pd.DataFrame
    monitor: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} 缺少必需字段：{', '.join(missing)}")


def validate_tables(tables: ResultTables) -> None:
    """在绘图前检查输入契约，避免生成内容不完整的报告图。"""

    _require_columns(
        tables.summary,
        {
            "scenario_id",
            "strategy_id",
            "completion_rate",
            "latency_p95_s",
            "edge_ratio",
            "cloud_ratio",
            "collaborative_ratio",
            "upload_mb",
            "average_expected_accuracy",
        },
        "策略汇总 CSV",
    )
    _require_columns(
        tables.detailed,
        {
            "scenario_id",
            "strategy_id",
            "seed",
            "completion_rate",
            "average_expected_accuracy",
        },
        "逐种子明细 CSV",
    )
    _require_columns(
        tables.by_task_type,
        {
            "scenario_id",
            "strategy_id",
            "seed",
            "task_type",
            "finished_tasks",
            "cloud_ratio",
            "collaborative_ratio",
            "average_payload_mb",
        },
        "任务类型明细 CSV",
    )
    _require_columns(tables.monitor, {"r"}, "PPO monitor CSV")

    if tables.summary.empty or tables.detailed.empty or tables.by_task_type.empty:
        raise ValueError("正式实验结果为空，无法生成指标图。")
    if tables.monitor.empty:
        raise ValueError("PPO monitor CSV 没有回合奖励记录。")

    required_strategies = set(STRATEGY_ORDER)
    present_strategies = set(tables.summary["strategy_id"].astype(str))
    missing_strategies = required_strategies - present_strategies
    if missing_strategies:
        missing_labels = [STRATEGY_LABELS[item] for item in sorted(missing_strategies)]
        raise ValueError(f"策略汇总缺少正式四策略：{', '.join(missing_labels)}")


def load_result_tables(
    summary_path: Path,
    detailed_path: Path,
    by_task_type_path: Path,
    monitor_path: Path,
) -> ResultTables:
    """加载正式实验 CSV；monitor 首行的 SB3 元数据注释会被忽略。"""

    paths = (summary_path, detailed_path, by_task_type_path, monitor_path)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("找不到绘图输入文件：\n" + "\n".join(missing))

    tables = ResultTables(
        summary=pd.read_csv(summary_path),
        detailed=pd.read_csv(detailed_path),
        by_task_type=pd.read_csv(by_task_type_path),
        monitor=pd.read_csv(monitor_path, comment="#"),
    )
    validate_tables(tables)
    return tables


def _ordered_values(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    present = {str(value) for value in values}
    return [value for value in preferred if value in present] + sorted(present - set(preferred))


def _scenario_label(scenario_id: str) -> str:
    return SCENARIO_LABELS.get(scenario_id, scenario_id)


def _strategy_label(strategy_id: str) -> str:
    return STRATEGY_LABELS.get(strategy_id, strategy_id)


def _task_type_label(task_type: str) -> str:
    return TASK_TYPE_LABELS.get(task_type, task_type)


def derive_upload_comparison(tables: ResultTables) -> pd.DataFrame:
    """派生同批完成任务全量上云基准与两层调度上传量。

    全量上云基准按 EDGE_ONLY 同批已完成任务的 payload 总量计算，避免把
    未完成且没有逐任务 payload 记录的任务凭空计入。
    """

    edge_rows = tables.by_task_type[
        tables.by_task_type["strategy_id"].astype(str) == "EDGE_ONLY"
    ].copy()
    if edge_rows.empty:
        raise ValueError("任务类型明细中没有纯边端基准，无法派生全量上云通信量。")

    edge_rows["full_cloud_upload_mb"] = (
        pd.to_numeric(edge_rows["finished_tasks"], errors="raise")
        * pd.to_numeric(edge_rows["average_payload_mb"], errors="raise")
    )
    full_cloud_by_seed = (
        edge_rows.groupby(["scenario_id", "seed"], as_index=False)["full_cloud_upload_mb"]
        .sum()
        .groupby("scenario_id", as_index=False)["full_cloud_upload_mb"]
        .mean()
    )

    two_layer = tables.summary[
        tables.summary["strategy_id"].astype(str) == "TWO_LAYER"
    ][["scenario_id", "upload_mb"]].copy()
    two_layer = two_layer.groupby("scenario_id", as_index=False)["upload_mb"].mean()
    two_layer = two_layer.rename(columns={"upload_mb": "two_layer_upload_mb"})

    comparison = full_cloud_by_seed.merge(two_layer, on="scenario_id", how="inner")
    if comparison.empty:
        raise ValueError("全量上云基准与两层调度没有可匹配的实验场景。")
    comparison["reduction_rate"] = np.where(
        comparison["full_cloud_upload_mb"] > 0,
        1.0
        - comparison["two_layer_upload_mb"] / comparison["full_cloud_upload_mb"],
        np.nan,
    )
    return comparison


def derive_task_type_cloud_matrix(tables: ResultTables) -> pd.DataFrame:
    """按完成任务数加权计算两层调度的上云相关最终路径占比。"""

    rows = tables.by_task_type[
        tables.by_task_type["strategy_id"].astype(str) == "TWO_LAYER"
    ].copy()
    if rows.empty:
        raise ValueError("任务类型明细中没有两层调度数据。")

    finished = pd.to_numeric(rows["finished_tasks"], errors="raise")
    cloud_related = (
        pd.to_numeric(rows["cloud_ratio"], errors="raise")
        + pd.to_numeric(rows["collaborative_ratio"], errors="raise")
    )
    rows["cloud_related_tasks"] = finished * cloud_related
    grouped = rows.groupby(["task_type", "scenario_id"], as_index=False).agg(
        finished_tasks=("finished_tasks", "sum"),
        cloud_related_tasks=("cloud_related_tasks", "sum"),
    )
    grouped["cloud_related_ratio_pct"] = np.where(
        grouped["finished_tasks"] > 0,
        grouped["cloud_related_tasks"] / grouped["finished_tasks"] * 100.0,
        np.nan,
    )
    return grouped.pivot(
        index="task_type", columns="scenario_id", values="cloud_related_ratio_pct"
    )


def _configure_plotting() -> str:
    """配置 Matplotlib 中文字体和简洁的论文图表样式。"""

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    candidates = (
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "DengXian",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
    )
    font_name = next((name for name in candidates if name in available_fonts), None)
    if font_name is None:
        raise RuntimeError(
            "没有检测到可用中文字体。请安装微软雅黑、黑体或 Noto Sans CJK SC 后重试。"
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#AAB2BD",
            "axes.labelcolor": "#25313C",
            "axes.titleweight": "normal",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.color": "#394955",
            "ytick.color": "#394955",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return font_name


def _finish_axes(ax: Any, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9E0E6", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def _save_figure(
    fig: Any,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    saved: list[Path] = []
    for file_format in formats:
        destination = output_dir / f"{stem}.{file_format}"
        fig.savefig(destination, dpi=dpi, bbox_inches="tight")
        saved.append(destination)
    plt.close(fig)
    return saved


def _plot_completion_rate(
    tables: ResultTables,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    scenarios = _ordered_values(tables.summary["scenario_id"], SCENARIO_ORDER)
    strategies = _ordered_values(tables.summary["strategy_id"], STRATEGY_ORDER)
    x_positions = np.arange(len(scenarios), dtype=float)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for strategy_index, strategy_id in enumerate(strategies):
        means: list[float] = []
        standard_deviations: list[float] = []
        for scenario_id in scenarios:
            summary_values = tables.summary.loc[
                (tables.summary["scenario_id"].astype(str) == scenario_id)
                & (tables.summary["strategy_id"].astype(str) == strategy_id),
                "completion_rate",
            ]
            if summary_values.empty:
                raise ValueError(f"{scenario_id}/{strategy_id} 缺少完成率汇总。")
            mean_value = float(pd.to_numeric(summary_values, errors="raise").mean() * 100.0)
            raw_values = pd.to_numeric(
                tables.detailed.loc[
                    (tables.detailed["scenario_id"].astype(str) == scenario_id)
                    & (tables.detailed["strategy_id"].astype(str) == strategy_id),
                    "completion_rate",
                ],
                errors="raise",
            ).to_numpy(dtype=float) * 100.0
            if raw_values.size == 0:
                raise ValueError(f"{scenario_id}/{strategy_id} 缺少逐种子完成率。")

            means.append(mean_value)
            standard_deviations.append(float(raw_values.std(ddof=0)))

        centers = x_positions + (strategy_index - (len(strategies) - 1) / 2.0) * bar_width
        bars = ax.bar(
            centers,
            means,
            width=bar_width,
            yerr=standard_deviations,
            color=STRATEGY_COLORS[strategy_id],
            capsize=3,
            label=_strategy_label(strategy_id),
        )
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in means], padding=3, fontsize=8)

    ax.set_title("三种实验场景下四种调度策略的任务完成率", pad=18)
    ax.set_xlabel("实验场景", labelpad=10)
    ax.set_ylabel("任务完成率（%）", labelpad=10)
    ax.set_xticks(x_positions, [_scenario_label(item) for item in scenarios])
    ax.set_ylim(0, 105)
    ax.legend(title="调度策略", ncol=2, loc="upper right")
    _finish_axes(ax, grid_axis="y")
    fig.tight_layout()
    return _save_figure(fig, output_dir, CHART_STEMS[0], formats, dpi)


def _plot_p95_latency(
    tables: ResultTables,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    scenarios = _ordered_values(tables.summary["scenario_id"], SCENARIO_ORDER)
    x_positions = np.arange(len(scenarios), dtype=float)
    edge_values: list[float] = []
    two_layer_values: list[float] = []
    for scenario_id in scenarios:
        subset = tables.summary[tables.summary["scenario_id"].astype(str) == scenario_id]
        edge = subset.loc[subset["strategy_id"].astype(str) == "EDGE_ONLY", "latency_p95_s"]
        two = subset.loc[subset["strategy_id"].astype(str) == "TWO_LAYER", "latency_p95_s"]
        if edge.empty or two.empty:
            raise ValueError(f"{scenario_id} 缺少纯边端或两层调度 P95 时延。")
        edge_values.append(float(pd.to_numeric(edge, errors="raise").mean()))
        two_layer_values.append(float(pd.to_numeric(two, errors="raise").mean()))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bar_width = 0.32
    edge_bars = ax.bar(
        x_positions - bar_width / 2.0,
        edge_values,
        width=bar_width,
        color=STRATEGY_COLORS["EDGE_ONLY"],
        label="纯边端基准",
    )
    two_layer_bars = ax.bar(
        x_positions + bar_width / 2.0,
        two_layer_values,
        width=bar_width,
        color=STRATEGY_COLORS["TWO_LAYER"],
        label="两层调度",
    )
    ax.bar_label(edge_bars, labels=[f"{value:.3f}秒" for value in edge_values], padding=3, fontsize=9)
    ax.bar_label(two_layer_bars, labels=[f"{value:.3f}秒" for value in two_layer_values], padding=3, fontsize=9)

    for x_value, edge_value, two_value in zip(x_positions, edge_values, two_layer_values):
        reduction = (edge_value - two_value) / edge_value * 100.0 if edge_value else np.nan
        direction = "降低" if reduction >= 0 else "增加"
        ax.text(
            x_value,
            max(edge_value, two_value) + 0.055,
            f"{direction} {abs(reduction):.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("两层调度相对纯边端基准的第95百分位端到端时延变化", pad=18)
    ax.set_xlabel("实验场景", labelpad=10)
    ax.set_ylabel("第95百分位端到端时延（秒）", labelpad=10)
    ax.set_xticks(x_positions, [_scenario_label(item) for item in scenarios])
    ax.set_ylim(0, max(edge_values + two_layer_values) + 0.14)
    ax.legend(title="调度策略", loc="center left", bbox_to_anchor=(1.01, 0.5))
    _finish_axes(ax, grid_axis="y")
    fig.tight_layout()
    return _save_figure(fig, output_dir, CHART_STEMS[1], formats, dpi)


def _plot_route_composition(
    tables: ResultTables,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    rows = tables.summary[tables.summary["strategy_id"].astype(str) == "TWO_LAYER"].copy()
    scenarios = _ordered_values(rows["scenario_id"], SCENARIO_ORDER)
    rows = rows.set_index("scenario_id").reindex(scenarios)
    components = (
        ("edge_ratio", "边端直接完成", "#138A72"),
        ("cloud_ratio", "云端直接执行", "#3973B7"),
        ("collaborative_ratio", "边云协同复核", "#8E65A8"),
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    x_positions = np.arange(len(scenarios), dtype=float)
    bottoms = np.zeros(len(scenarios), dtype=float)
    for column, label, color in components:
        values = pd.to_numeric(rows[column], errors="raise").to_numpy(dtype=float) * 100.0
        bars = ax.bar(x_positions, values, bottom=bottoms, width=0.58, color=color, label=label)
        for bar, value, bottom in zip(bars, values, bottoms):
            if value >= 1.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bottom + value / 2.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if color != "#3973B7" else "white",
                )
            elif value > 0:
                ax.annotate(
                    f"{value:.1f}%",
                    (bar.get_x() + bar.get_width() / 2.0, bottom + value / 2.0),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=color,
                )
        bottoms += values

    ax.set_title("两层调度在不同场景下的最终执行路径构成", pad=18)
    ax.set_xlabel("实验场景", labelpad=10)
    ax.set_ylabel("最终执行路径占比（%）", labelpad=10)
    ax.set_xticks(x_positions, [_scenario_label(item) for item in scenarios])
    ax.set_ylim(0, 108)
    ax.legend(title="最终执行路径", loc="center left", bbox_to_anchor=(1.01, 0.5))
    _finish_axes(ax, grid_axis="y")
    fig.tight_layout()
    return _save_figure(fig, output_dir, CHART_STEMS[2], formats, dpi)


def _plot_upload_volume(
    tables: ResultTables,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    comparison = derive_upload_comparison(tables).set_index("scenario_id")
    scenarios = _ordered_values(comparison.index, SCENARIO_ORDER)
    comparison = comparison.reindex(scenarios)
    x_positions = np.arange(len(scenarios), dtype=float)
    full_cloud = comparison["full_cloud_upload_mb"].to_numpy(dtype=float)
    two_layer = comparison["two_layer_upload_mb"].to_numpy(dtype=float)
    reductions = comparison["reduction_rate"].to_numpy(dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bar_width = 0.32
    baseline_bars = ax.bar(
        x_positions - bar_width / 2.0,
        full_cloud,
        width=bar_width,
        color=STRATEGY_COLORS["EDGE_ONLY"],
        label="同批完成任务全量上云基准",
    )
    two_layer_bars = ax.bar(
        x_positions + bar_width / 2.0,
        two_layer,
        width=bar_width,
        color=STRATEGY_COLORS["TWO_LAYER"],
        label="两层调度实际上传量",
    )
    ax.bar_label(
        baseline_bars,
        labels=[f"{value:.1f}兆字节" for value in full_cloud],
        padding=3,
        fontsize=9,
    )
    ax.bar_label(
        two_layer_bars,
        labels=[f"{value:.1f}兆字节" for value in two_layer],
        padding=3,
        fontsize=9,
    )

    for x_value, baseline, actual, reduction in zip(
        x_positions, full_cloud, two_layer, reductions
    ):
        direction = "减少" if reduction >= 0 else "增加"
        ax.text(
            x_value,
            max(baseline, actual) + 48.0,
            f"{direction} {abs(reduction):.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("两层调度与全量上云基准的通信上传量对比", pad=18)
    ax.set_xlabel("实验场景", labelpad=10)
    ax.set_ylabel("累计上传数据量（兆字节/评测轮次）", labelpad=10)
    ax.set_xticks(x_positions, [_scenario_label(item) for item in scenarios])
    ax.set_ylim(0, max(full_cloud.max(), two_layer.max()) + 120.0)
    ax.legend(title="通信方案", loc="center left", bbox_to_anchor=(1.01, 0.5))
    _finish_axes(ax, grid_axis="y")
    fig.tight_layout()
    return _save_figure(fig, output_dir, CHART_STEMS[3], formats, dpi)


def _plot_training_reward(
    tables: ResultTables,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    rolling_window: int,
) -> list[Path]:
    rewards = pd.to_numeric(tables.monitor["r"], errors="raise").reset_index(drop=True)
    episodes = np.arange(1, len(rewards) + 1, dtype=int)
    rolling = rewards.rolling(window=rolling_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(episodes, rewards, color="#AEB7C0", lw=1.0, alpha=0.55, label="单回合奖励")
    ax.plot(
        episodes,
        rolling,
        color=STRATEGY_COLORS["TWO_LAYER"],
        lw=2.5,
        label=f"{rolling_window} 回合滑动平均",
    )
    final_value = float(rolling.iloc[-1])
    ax.scatter([episodes[-1]], [final_value], s=75, color=STRATEGY_COLORS["TWO_LAYER"], zorder=3)
    ax.annotate(
        f"末段滑动均值：{final_value:.1f}",
        (episodes[-1], final_value),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        color=STRATEGY_COLORS["TWO_LAYER"],
    )

    ax.set_title("近端策略优化（PPO）跨场景训练的回合奖励变化", pad=18)
    ax.set_xlabel("训练回合编号", labelpad=10)
    ax.set_ylabel("每回合累计奖励", labelpad=10)
    ax.legend(title="奖励序列", loc="lower right")
    _finish_axes(ax, grid_axis="y")
    fig.tight_layout()
    return _save_figure(fig, output_dir, CHART_STEMS[4], formats, dpi)


def _plot_task_type_heatmap(
    tables: ResultTables,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    matrix = derive_task_type_cloud_matrix(tables)
    scenarios = _ordered_values(matrix.columns, SCENARIO_ORDER)
    task_types = _ordered_values(matrix.index, TASK_TYPE_ORDER)
    matrix = matrix.reindex(index=task_types, columns=scenarios)
    values = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.8, 7.3))
    color_map = plt.get_cmap("YlGnBu").with_extremes(bad="#E7EBEF")
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=color_map, vmin=0, vmax=100, aspect="auto"
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("上云相关最终路径占比（%）", rotation=90, labelpad=12)

    ax.set_xticks(np.arange(len(scenarios)), [_scenario_label(item) for item in scenarios])
    ax.set_yticks(np.arange(len(task_types)), [_task_type_label(item) for item in task_types])
    ax.set_xlabel("实验场景", labelpad=10)
    ax.set_ylabel("任务类型", labelpad=10)
    ax.set_title("两层调度对不同任务类型的上云路径选择", pad=18)

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text_value = "无数据" if np.isnan(value) else f"{value:.1f}%"
            text_color = "white" if not np.isnan(value) and value >= 52 else "#25313C"
            ax.text(
                column_index,
                row_index,
                text_value,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return _save_figure(fig, output_dir, CHART_STEMS[5], formats, dpi)


def _plot_expected_accuracy(
    tables: ResultTables,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """绘制配置的路径级预期准确率先验，不表示真实模型Accuracy/F1。"""

    scenarios = _ordered_values(tables.summary["scenario_id"], SCENARIO_ORDER)
    strategies = _ordered_values(tables.summary["strategy_id"], STRATEGY_ORDER)
    x_positions = np.arange(len(scenarios), dtype=float)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for strategy_index, strategy_id in enumerate(strategies):
        means: list[float] = []
        standard_deviations: list[float] = []
        for scenario_id in scenarios:
            summary_values = tables.summary.loc[
                (tables.summary["scenario_id"].astype(str) == scenario_id)
                & (tables.summary["strategy_id"].astype(str) == strategy_id),
                "average_expected_accuracy",
            ]
            if summary_values.empty:
                raise ValueError(f"{scenario_id}/{strategy_id} 缺少平均预期准确率汇总。")
            means.append(
                float(pd.to_numeric(summary_values, errors="raise").mean() * 100.0)
            )

            raw_values = pd.to_numeric(
                tables.detailed.loc[
                    (tables.detailed["scenario_id"].astype(str) == scenario_id)
                    & (tables.detailed["strategy_id"].astype(str) == strategy_id),
                    "average_expected_accuracy",
                ],
                errors="raise",
            ).to_numpy(dtype=float) * 100.0
            if raw_values.size == 0:
                raise ValueError(f"{scenario_id}/{strategy_id} 缺少逐种子预期准确率。")
            standard_deviations.append(float(raw_values.std(ddof=0)))

        centers = x_positions + (
            strategy_index - (len(strategies) - 1) / 2.0
        ) * bar_width
        bars = ax.bar(
            centers,
            means,
            width=bar_width,
            yerr=standard_deviations,
            color=STRATEGY_COLORS[strategy_id],
            capsize=3,
            label=_strategy_label(strategy_id),
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}%" for value in means],
            padding=3,
            fontsize=8,
        )

    ax.set_title("三种实验场景下四种调度策略的平均预期准确率（仿真先验）", pad=18)
    ax.set_xlabel("实验场景", labelpad=10)
    ax.set_ylabel("平均预期准确率（%）", labelpad=10)
    ax.set_xticks(x_positions, [_scenario_label(item) for item in scenarios])
    ax.set_ylim(0, 105)
    ax.legend(title="调度策略", loc="center left", bbox_to_anchor=(1.01, 0.5))
    _finish_axes(ax, grid_axis="y")
    fig.tight_layout()
    return _save_figure(fig, output_dir, CHART_STEMS[6], formats, dpi)


def generate_charts(
    tables: ResultTables,
    output_dir: Path,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    rolling_window: int = 20,
    source_paths: dict[str, Path] | None = None,
) -> list[Path]:
    """生成七类中文图表和一份可追溯清单。"""

    validate_tables(tables)
    if dpi < 72:
        raise ValueError("DPI 不能低于 72。")
    if rolling_window < 1:
        raise ValueError("滑动窗口必须大于等于 1。")
    normalized_formats = tuple(dict.fromkeys(item.lower() for item in formats))
    unsupported = sorted(set(normalized_formats) - {"png", "svg", "pdf"})
    if unsupported:
        raise ValueError(f"不支持的图表格式：{', '.join(unsupported)}")

    font_name = _configure_plotting()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    saved.extend(_plot_completion_rate(tables, output_dir, normalized_formats, dpi))
    saved.extend(_plot_p95_latency(tables, output_dir, normalized_formats, dpi))
    saved.extend(_plot_route_composition(tables, output_dir, normalized_formats, dpi))
    saved.extend(_plot_upload_volume(tables, output_dir, normalized_formats, dpi))
    saved.extend(
        _plot_training_reward(tables, output_dir, normalized_formats, dpi, rolling_window)
    )
    saved.extend(_plot_task_type_heatmap(tables, output_dir, normalized_formats, dpi))
    saved.extend(_plot_expected_accuracy(tables, output_dir, normalized_formats, dpi))

    scenarios = _ordered_values(tables.summary["scenario_id"], SCENARIO_ORDER)
    strategies = _ordered_values(tables.summary["strategy_id"], STRATEGY_ORDER)
    manifest = {
        "schema_version": "scheduler-training-charts-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "font": font_name,
        "formats": list(normalized_formats),
        "dpi": dpi,
        "rolling_window": rolling_window,
        "scenarios": [{"id": item, "label": _scenario_label(item)} for item in scenarios],
        "strategies": [{"id": item, "label": _strategy_label(item)} for item in strategies],
        "seed_count": int(pd.to_numeric(tables.detailed["seed"], errors="raise").nunique()),
        "source_semantics": "Scheduler Training 场景仿真结果，不是实测系统指标。",
        "accuracy_semantics": "平均预期准确率是配置的路径级仿真先验，不是真实Accuracy/F1。",
        "input_files": {
            key: str(value.resolve()) for key, value in (source_paths or {}).items()
        },
        "generated_files": [path.name for path in saved],
    }
    manifest_path = output_dir / "chart_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Scheduler Training 中文实验指标图")
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_DIR / "results" / "scenario_formal_comparison.csv",
        help="四策略汇总 CSV",
    )
    parser.add_argument(
        "--detailed",
        type=Path,
        default=PROJECT_DIR / "results" / "scenario_formal_comparison_detailed.csv",
        help="逐随机种子明细 CSV",
    )
    parser.add_argument(
        "--by-task-type",
        type=Path,
        default=PROJECT_DIR / "results" / "scenario_formal_comparison_by_task_type.csv",
        help="按任务类型明细 CSV",
    )
    parser.add_argument(
        "--monitor",
        type=Path,
        default=PROJECT_DIR / "results" / "scenario_formal_monitor.csv",
        help="Stable-Baselines3 monitor CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "output",
        help="图表输出目录（默认：scheduler_training/output）",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=("png",),
        help="输出格式，可同时指定多种（默认：png）",
    )
    parser.add_argument("--dpi", type=int, default=300, help="位图分辨率（默认：300）")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="训练奖励滑动平均窗口（默认：20 回合）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_paths = {
        "summary": args.summary,
        "detailed": args.detailed,
        "by_task_type": args.by_task_type,
        "monitor": args.monitor,
    }
    try:
        tables = load_result_tables(
            args.summary,
            args.detailed,
            args.by_task_type,
            args.monitor,
        )
        saved = generate_charts(
            tables,
            args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
            rolling_window=args.rolling_window,
            source_paths=source_paths,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"绘图失败：{exc}") from exc

    print(f"[完成] 已在 {args.output_dir.resolve()} 生成 {len(saved)} 个图表文件：")
    for path in saved:
        print(f"  - {path.name}")
    print("  - chart_manifest.json（输入与生成物清单）")
    print("[说明] 图中调度指标来自场景仿真，不代表真实系统测量值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
