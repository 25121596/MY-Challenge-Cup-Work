from __future__ import annotations

"""Task-type defaults used while parsing real edge inputs.

Latency, confidence, accuracy, complexity, and payload values in this file are
legacy profile priors. They are estimates, not real-system measurements, and
must eventually be calibrated from Integration traces.
"""

from dataclasses import dataclass
import math


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
    collaborative_edge_latency_s: float
    collaborative_cloud_latency_s: float
    collaborative_accuracy: float
    collaborative_payload_ratio: float

    @property
    def payload_typical_mb(self) -> float:
        return math.sqrt(self.payload_min_mb * self.payload_max_mb)

    @property
    def edge_confidence_typical(self) -> float:
        return 0.5 * (self.edge_confidence_min + self.edge_confidence_max)


TASK_PROFILES: dict[str, TaskProfile] = {
    "fault_diagnosis": TaskProfile(
        "fault_diagnosis", "故障诊断", "根据传感器数据判断设备是否存在故障",
        "high", 0.20, 0.02, 0.30, 0.68, 0.88, 0.95, 0.13, 0.13, 0.62,
        0.07, 0.09, 0.965, 0.18,
    ),
    "quality_inspection": TaskProfile(
        "quality_inspection", "质量检测", "根据检测特征判断产品是否合格",
        "high", 0.20, 0.20, 1.20, 0.62, 0.84, 0.96, 0.17, 0.16, 0.72,
        0.10, 0.12, 0.978, 0.15,
    ),
    "safety_check": TaskProfile(
        "safety_check", "安全识别", "分析作业场景中的安全隐患",
        "high", 0.15, 0.12, 0.80, 0.70, 0.88, 0.97, 0.12, 0.12, 0.68,
        0.06, 0.08, 0.985, 0.18,
    ),
    "data_analysis": TaskProfile(
        "data_analysis", "数据分析", "分析历史监测数据趋势",
        "medium", 0.35, 0.03, 0.35, 0.60, 0.82, 0.93, 0.22, 0.17, 0.78,
        0.12, 0.10, 0.95, 0.20,
    ),
    "status_monitoring": TaskProfile(
        "status_monitoring", "状态监控", "实时监控设备运行参数是否正常",
        "low", 0.10, 0.005, 0.06, 0.84, 0.96, 0.97, 0.065, 0.095, 0.30,
        0.04, 0.06, 0.975, 0.25,
    ),
    "predictive_maintenance": TaskProfile(
        "predictive_maintenance", "预测性维护", "根据历史数据预测设备维护时间点",
        "medium", 0.50, 0.05, 0.55, 0.58, 0.80, 0.94, 0.31, 0.21, 0.92,
        0.15, 0.12, 0.96, 0.22,
    ),
    "generic_anomaly": TaskProfile(
        "generic_anomaly", "通用异常检测", "对未知传感器类型执行通用异常检测",
        "high", 0.20, 0.01, 0.22, 0.56, 0.80, 0.92, 0.15, 0.14, 0.70,
        0.08, 0.09, 0.955, 0.20,
    ),
    "video_anomaly": TaskProfile(
        "video_anomaly", "视频异常检测", "对视频预处理后的运动/亮度/边缘特征进行异常检测",
        "high", 0.25, 0.30, 1.50, 0.58, 0.82, 0.95, 0.23, 0.21, 0.86,
        0.10, 0.10, 0.98, 0.10,
    ),
}

TASK_TYPE_ORDER = tuple(TASK_PROFILES)
VALID_TASK_TYPES = frozenset(TASK_TYPE_ORDER)
PRIORITY_SCORE = {"low": 0.0, "medium": 0.5, "high": 1.0}


def get_task_profile(task_type: str) -> TaskProfile:
    return TASK_PROFILES.get(str(task_type), TASK_PROFILES["generic_anomaly"])
