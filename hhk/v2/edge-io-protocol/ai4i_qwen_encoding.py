"""Stable compact prompt encoding for the AI4I Qwen review expert.

The tabular MoE remains the primary AI4I path.  This module only prepares a
small, schema-bound observation when the router explicitly requests a Qwen
review.  It never consumes labels or the raw ``readings`` training fields.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Sequence


AI4I_QWEN_ENCODING_VERSION = "ai4i_qwen_8d_v1"
AI4I_FEATURE_SCHEMA_ID = "ai4i_8d_v1"
AI4I_MODEL_ROUTE = "ai4i_tabular_moe_v1"
_TYPE_CODES = ("H", "L", "M")


def _scaled_zscore(value: float) -> str:
    """Encode a finite z-score as a signed tenth, with deterministic ties."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("AI4I Qwen encoding requires finite numeric features")
    scaled = int((Decimal(str(value)) * Decimal("10")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return f"{scaled:+d}"


def validate_ai4i_features(features: Sequence[float]) -> None:
    if not isinstance(features, (list, tuple)) or len(features) != 8:
        raise ValueError("AI4I Qwen encoding requires exactly eight features")
    type_code = features[:3]
    if any(isinstance(value, bool) or value not in (0, 1) for value in type_code) or sum(type_code) != 1:
        raise ValueError("AI4I Qwen encoding requires a one-hot Type")
    for value in features[3:]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("AI4I Qwen encoding requires finite numeric features")


def is_ai4i_online_task(task: dict) -> bool:
    """Identify only records bound to the frozen AI4I online feature contract."""
    data = task.get("data", {})
    return (data.get("feature_schema_id") == AI4I_FEATURE_SCHEMA_ID and
            data.get("model_route") == AI4I_MODEL_ROUTE)


def encode_ai4i_qwen_review(task: dict) -> str:
    """Return the complete, compact and versioned review prompt.

    Fields are ``T, A, P, R, Q, W`` in the frozen schema order.  A/P/R/Q/W
    are their z-scores multiplied by ten then rounded half-up.  For example,
    ``A=-4`` means an air-temperature z-score in [-0.45, -0.35).
    """
    data = task.get("data", {})
    if "ground_truth" in data or "readings" in data:
        raise ValueError("AI4I Qwen review accepts the features-only online contract")
    if not is_ai4i_online_task(task):
        raise ValueError("AI4I Qwen review requires the frozen schema and route")
    features = data.get("features")
    validate_ai4i_features(features)
    machine_type = _TYPE_CODES[features[:3].index(1)]
    values = (_scaled_zscore(value) for value in features[3:])
    encoded = " ".join(f"{name}={value}" for name, value in zip(("A", "P", "R", "Q", "W"), values))
    return (
        "AI4I_QWEN_8D_V1. AI4I failure review; 0=no failure pattern, 1=possible failure. "
        "T=type(H/L/M); A/P/R/Q/W=z*10. Output one digit only.\n"
        f"T={machine_type} {encoded}"
    )
