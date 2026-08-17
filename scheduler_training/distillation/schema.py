from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUIRED_OUTPUT_FIELDS = ("action", "level", "category", "summary", "reason")
VALID_LEVELS = ("low", "medium", "high", "critical")

TASK_INSTRUCTIONS: dict[str, str] = {
    "fault_diagnosis": "根据设备传感器数据完成故障诊断，判断故障类别、严重程度和处置动作。",
    "quality_inspection": "根据检测数据完成质量判断，识别缺陷类别、严重程度和处置动作。",
    "safety_check": "根据作业场景数据识别安全隐患、风险等级和处置动作。",
    "data_analysis": "根据监测数据识别趋势变化、异常模式和建议动作。",
    "status_monitoring": "根据运行参数判断设备状态、偏离程度和处置动作。",
    "predictive_maintenance": "根据历史与当前数据判断维护需求、风险等级和建议动作。",
    "generic_anomaly": "根据数值特征判断是否存在异常、异常类别和处置动作。",
    "video_anomaly": "根据视频预处理后、学生端可见的数值或文本特征判断异常事件。",
}

CATEGORY_HINTS: dict[str, list[str]] = {
    "fault_diagnosis": [
        "normal", "bearing_fault", "motor_overheat", "gearbox_wear",
        "vibration_abnormal", "lubrication_failure", "unknown_fault",
    ],
    "quality_inspection": [
        "normal", "surface_defect", "dimension_error", "material_flaw",
        "coating_issue", "assembly_defect", "unknown_defect",
    ],
    "safety_check": [
        "safe", "gas_leak", "fire_risk", "structural_issue",
        "electrical_hazard", "machinery_guard", "unknown_hazard",
    ],
    "data_analysis": [
        "normal", "trend_shift", "anomaly_pattern", "correlation_change",
        "threshold_breach", "data_quality_issue",
    ],
    "status_monitoring": [
        "normal", "slight_deviation", "parameter_drift",
        "calibration_needed", "sensor_degradation",
    ],
    "predictive_maintenance": [
        "normal", "bearing_wear", "lubrication_due", "filter_replacement",
        "belt_tension", "motor_degradation",
    ],
    "generic_anomaly": [
        "normal", "sensor_drift", "unexpected_pattern", "data_gap",
        "signal_noise", "outlier_spike",
    ],
    "video_anomaly": [
        "normal", "motion_anomaly", "brightness_change", "edge_distortion",
        "object_intrusion", "scene_change",
    ],
}


@dataclass(frozen=True)
class TeacherConfig:
    backend: str = "api"
    api_base: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen3-8b-teacher"
    api_key: str = "not-needed"
    temperature: float = 0.1
    max_tokens: int = 256
    timeout_s: float = 120.0
    retries: int = 1
    no_think: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeacherConfig":
        teacher = data.get("teacher", data)
        return cls(
            backend=str(teacher.get("backend", "api")),
            api_base=str(teacher.get("api_base", "http://127.0.0.1:8080/v1")),
            model=str(teacher.get("model", "qwen3-8b-teacher")),
            api_key=str(teacher.get("api_key", "not-needed")),
            temperature=float(teacher.get("temperature", 0.1)),
            max_tokens=int(teacher.get("max_tokens", 256)),
            timeout_s=float(teacher.get("timeout_s", 120.0)),
            retries=int(teacher.get("retries", 1)),
            no_think=bool(teacher.get("no_think", True)),
        )
