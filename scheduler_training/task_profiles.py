"""Phase4.5工业任务类型与仿真性能先验。

任务类型名称和默认优先级/截止时间来自hhk的task_types.json。
载荷范围、1.5B边端模型与7B云端模型的时延/准确率为当前仿真阶段的
可配置先验，不代表真实模型测量结果；后续接入真实模型后应由实测数据校准。

每类任务都保留4个Post-edge云端复核路径的仿真参数：
- collaborative_edge_latency_s：边端初步处理（特征提取）时间
- collaborative_cloud_latency_s：云端复核时间
- collaborative_accuracy：协同路径准确率
- collaborative_payload_ratio：上传压缩特征占原始载荷的比例
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskProfile:
    key: str
    name: str
    description: str
    default_priority: str
    default_deadline_s: float
    payload_min_mb: float
    payload_max_mb: float
    edge_confidence_min: float
    edge_confidence_max: float
    cloud_accuracy: float
    edge_base_latency_s: float
    cloud_base_latency_s: float
    complexity: float
    sampling_weight: float
    # Post-edge云端复核路径的仿真参数。
    collaborative_edge_latency_s: float = 0.08
    collaborative_cloud_latency_s: float = 0.10
    collaborative_accuracy: float = 0.97
    collaborative_payload_ratio: float = 0.20

    @property
    def payload_typical_mb(self) -> float:
        return math.sqrt(self.payload_min_mb * self.payload_max_mb)

    @property
    def edge_confidence_typical(self) -> float:
        return 0.5 * (
            self.edge_confidence_min + self.edge_confidence_max
        )


TASK_PROFILES: dict[str, TaskProfile] = {
    "fault_diagnosis": TaskProfile(
        key="fault_diagnosis",
        name="故障诊断",
        description="根据传感器数据判断设备是否存在故障",
        default_priority="high",
        default_deadline_s=0.20,
        payload_min_mb=0.02,
        payload_max_mb=0.30,
        edge_confidence_min=0.68,
        edge_confidence_max=0.88,
        cloud_accuracy=0.95,
        edge_base_latency_s=0.13,
        cloud_base_latency_s=0.13,
        complexity=0.62,
        sampling_weight=0.20,
        # 协同：边端做特征提取很快，云端复核需要时间。
        collaborative_edge_latency_s=0.07,
        collaborative_cloud_latency_s=0.09,
        collaborative_accuracy=0.965,
        collaborative_payload_ratio=0.18,
    ),
    "quality_inspection": TaskProfile(
        key="quality_inspection",
        name="质量检测",
        description="根据检测特征判断产品是否合格",
        default_priority="high",
        default_deadline_s=0.20,
        payload_min_mb=0.20,
        payload_max_mb=1.20,
        edge_confidence_min=0.62,
        edge_confidence_max=0.84,
        cloud_accuracy=0.96,
        edge_base_latency_s=0.17,
        cloud_base_latency_s=0.16,
        complexity=0.72,
        sampling_weight=0.14,
        # 视频/图像类：协同收益最大（边端压缩）。
        collaborative_edge_latency_s=0.10,
        collaborative_cloud_latency_s=0.12,
        collaborative_accuracy=0.978,
        collaborative_payload_ratio=0.15,
    ),
    "safety_check": TaskProfile(
        key="safety_check",
        name="安全识别",
        description="分析作业场景中的安全隐患",
        default_priority="high",
        default_deadline_s=0.15,
        payload_min_mb=0.12,
        payload_max_mb=0.80,
        edge_confidence_min=0.70,
        edge_confidence_max=0.88,
        cloud_accuracy=0.97,
        edge_base_latency_s=0.12,
        cloud_base_latency_s=0.12,
        complexity=0.68,
        sampling_weight=0.12,
        # 安全要求高：协同收益显著。
        collaborative_edge_latency_s=0.06,
        collaborative_cloud_latency_s=0.08,
        collaborative_accuracy=0.985,
        collaborative_payload_ratio=0.18,
    ),
    "data_analysis": TaskProfile(
        key="data_analysis",
        name="数据分析",
        description="分析历史监测数据趋势",
        default_priority="medium",
        default_deadline_s=0.35,
        payload_min_mb=0.03,
        payload_max_mb=0.35,
        edge_confidence_min=0.60,
        edge_confidence_max=0.82,
        cloud_accuracy=0.93,
        edge_base_latency_s=0.22,
        cloud_base_latency_s=0.17,
        complexity=0.78,
        sampling_weight=0.13,
        collaborative_edge_latency_s=0.12,
        collaborative_cloud_latency_s=0.10,
        collaborative_accuracy=0.95,
        collaborative_payload_ratio=0.20,
    ),
    "status_monitoring": TaskProfile(
        key="status_monitoring",
        name="状态监控",
        description="实时监控设备运行参数是否正常",
        default_priority="low",
        default_deadline_s=0.10,
        payload_min_mb=0.005,
        payload_max_mb=0.06,
        edge_confidence_min=0.84,
        edge_confidence_max=0.96,
        cloud_accuracy=0.97,
        edge_base_latency_s=0.065,
        cloud_base_latency_s=0.095,
        complexity=0.30,
        sampling_weight=0.10,
        # 简单任务：协同开销可能不值得。
        collaborative_edge_latency_s=0.04,
        collaborative_cloud_latency_s=0.06,
        collaborative_accuracy=0.975,
        collaborative_payload_ratio=0.25,
    ),
    "predictive_maintenance": TaskProfile(
        key="predictive_maintenance",
        name="预测性维护",
        description="根据历史数据预测设备维护时间点",
        default_priority="medium",
        default_deadline_s=0.50,
        payload_min_mb=0.05,
        payload_max_mb=0.55,
        edge_confidence_min=0.58,
        edge_confidence_max=0.80,
        cloud_accuracy=0.94,
        edge_base_latency_s=0.31,
        cloud_base_latency_s=0.21,
        complexity=0.92,
        sampling_weight=0.09,
        # 复杂任务：协同复用边端特征加速云端。
        collaborative_edge_latency_s=0.15,
        collaborative_cloud_latency_s=0.12,
        collaborative_accuracy=0.96,
        collaborative_payload_ratio=0.22,
    ),
    "generic_anomaly": TaskProfile(
        key="generic_anomaly",
        name="通用异常检测",
        description="对未知传感器类型执行通用异常检测",
        default_priority="high",
        default_deadline_s=0.20,
        payload_min_mb=0.01,
        payload_max_mb=0.22,
        edge_confidence_min=0.56,
        edge_confidence_max=0.80,
        cloud_accuracy=0.92,
        edge_base_latency_s=0.15,
        cloud_base_latency_s=0.14,
        complexity=0.70,
        sampling_weight=0.08,
        collaborative_edge_latency_s=0.08,
        collaborative_cloud_latency_s=0.09,
        collaborative_accuracy=0.955,
        collaborative_payload_ratio=0.20,
    ),
    "video_anomaly": TaskProfile(
        key="video_anomaly",
        name="视频异常检测",
        description="对视频预处理后的运动/亮度/边缘特征进行异常检测",
        default_priority="high",
        default_deadline_s=0.25,
        payload_min_mb=0.30,
        payload_max_mb=1.50,
        edge_confidence_min=0.58,
        edge_confidence_max=0.82,
        cloud_accuracy=0.95,
        edge_base_latency_s=0.23,
        cloud_base_latency_s=0.21,
        complexity=0.86,
        sampling_weight=0.06,
        # 视频类：协同最受益（边端压缩关键特征）。
        collaborative_edge_latency_s=0.10,
        collaborative_cloud_latency_s=0.10,
        collaborative_accuracy=0.98,
        collaborative_payload_ratio=0.10,
    ),
}

TASK_TYPE_ORDER = tuple(TASK_PROFILES.keys())
VALID_TASK_TYPES = frozenset(TASK_TYPE_ORDER)
PRIORITY_SCORE = {"low": 0.0, "medium": 0.5, "high": 1.0}


def get_task_profile(task_type: str) -> TaskProfile:
    """返回任务类型配置；未知类型按generic_anomaly处理。"""

    return TASK_PROFILES.get(
        str(task_type),
        TASK_PROFILES["generic_anomaly"],
    )


def sample_task_type(rng: np.random.Generator) -> str:
    names = list(TASK_TYPE_ORDER)
    weights = np.asarray(
        [TASK_PROFILES[name].sampling_weight for name in names],
        dtype=float,
    )
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def sample_log_uniform(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
) -> float:
    return float(
        np.exp(rng.uniform(np.log(minimum), np.log(maximum)))
    )


def sample_profile_values(
    task_type: str,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    """为typed_simulation生成一组有类型差异的任务参数。"""

    profile = get_task_profile(task_type)
    deadline_scale = float(rng.uniform(0.90, 1.55))
    edge_latency_jitter = float(rng.uniform(0.88, 1.15))
    cloud_latency_jitter = float(rng.uniform(0.88, 1.15))
    collab_edge_jitter = float(rng.uniform(0.88, 1.15))
    collab_cloud_jitter = float(rng.uniform(0.88, 1.15))

    return {
        "priority": profile.default_priority,
        "deadline_s": profile.default_deadline_s * deadline_scale,
        "size_mb": sample_log_uniform(
            rng,
            profile.payload_min_mb,
            profile.payload_max_mb,
        ),
        "confidence": float(
            rng.uniform(
                profile.edge_confidence_min,
                profile.edge_confidence_max,
            )
        ),
        "cloud_accuracy": profile.cloud_accuracy,
        "edge_base_latency_s": (
            profile.edge_base_latency_s * edge_latency_jitter
        ),
        "cloud_base_latency_s": (
            profile.cloud_base_latency_s * cloud_latency_jitter
        ),
        "complexity": float(
            np.clip(profile.complexity + rng.normal(0.0, 0.06), 0.0, 1.0)
        ),
        # 协同参数。
        "collaborative_edge_latency_s": (
            profile.collaborative_edge_latency_s * collab_edge_jitter
        ),
        "collaborative_cloud_latency_s": (
            profile.collaborative_cloud_latency_s * collab_cloud_jitter
        ),
        "collaborative_accuracy": profile.collaborative_accuracy,
        "collaborative_payload_ratio": profile.collaborative_payload_ratio,
    }


def profiles_as_dict() -> dict[str, dict[str, Any]]:
    return {
        key: asdict(profile)
        for key, profile in TASK_PROFILES.items()
    }
