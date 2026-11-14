"""Phase4.5可插拔蒸馏模块。

该包与pre-route/Post-edge调度仿真解耦：它复用同一份 hhk 标准 input，负责教师伪标签、
训练集构建、学生模型参数化 LoRA 训练和基于真实标签的准确率评估。
"""

from .schema import REQUIRED_OUTPUT_FIELDS, VALID_LEVELS
from .output_validation import ValidationResult, validate_output

__all__ = [
    "REQUIRED_OUTPUT_FIELDS",
    "VALID_LEVELS",
    "ValidationResult",
    "validate_output",
]
