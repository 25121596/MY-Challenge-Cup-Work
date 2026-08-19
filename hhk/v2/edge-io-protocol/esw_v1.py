"""ESW-v1: a fixed-width edge scheduling word.

This module is deliberately independent from model inference and routing
policy.  JSON remains the complete audit record; ESW is its 32-bit fast-path
projection for a downstream scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Optional


SCHEMA_VERSION = 1
WIRE_SIZE_BYTES = 4


class LocalPrediction(IntEnum):
    UNKNOWN = 0
    NORMAL = 1
    FAULT = 2
    RESERVED = 3


class Review(IntEnum):
    NONE = 0
    RECOMMENDED = 1
    REQUIRED = 2
    RESERVED = 3


class Safety(IntEnum):
    UNKNOWN = 0
    P2 = 1
    P1 = 2
    P0 = 3


class EdgeAvailability(IntEnum):
    UNKNOWN = 0
    AVAILABLE = 1
    UNAVAILABLE = 2
    DEGRADED = 3


class NextAction(IntEnum):
    UNKNOWN = 0
    ACCEPT_LOCAL = 1
    DIRECT_CLOUD = 2
    CLOUD_REVIEW = 3


class DecodeStatus(str, Enum):
    OK = "ESW_OK"
    INVALID = "ESW_INVALID"
    UNSUPPORTED = "ESW_UNSUPPORTED"


@dataclass(frozen=True)
class EswFields:
    risk_q8: int = 0
    risk_valid: bool = False
    calibrated: bool = False
    local_prediction: LocalPrediction = LocalPrediction.UNKNOWN
    review: Review = Review.NONE
    safety: Safety = Safety.UNKNOWN
    edge_availability: EdgeAvailability = EdgeAvailability.UNKNOWN
    next_action: NextAction = NextAction.UNKNOWN


@dataclass(frozen=True)
class EswDecodeResult:
    status: DecodeStatus
    fields: EswFields
    raw_word: Optional[int]
    reason: Optional[str] = None


def quantize_risk(probability: float) -> int:
    """Quantize with round-half-up, never Python's banker's ``round``."""
    probability = min(1.0, max(0.0, float(probability)))
    return int(probability * 255.0 + 0.5)


def dequantize_risk(risk_q8: int) -> float:
    if not 0 <= risk_q8 <= 255:
        raise ValueError("risk_q8 must be in [0, 255]")
    return risk_q8 / 255.0


def qwen_unscored_fields(*, local_prediction: LocalPrediction = LocalPrediction.UNKNOWN,
                         review: Review = Review.REQUIRED,
                         safety: Safety = Safety.P1,
                         edge_availability: EdgeAvailability = EdgeAvailability.AVAILABLE,
                         next_action: NextAction = NextAction.CLOUD_REVIEW) -> EswFields:
    """Construct the only valid ESW representation of an uncalibrated Qwen result."""
    return EswFields(0, False, False, local_prediction, review, safety,
                     edge_availability, next_action)


def _fallback() -> EswFields:
    return EswFields(next_action=NextAction.CLOUD_REVIEW)


def _invalid(fields: EswFields) -> Optional[str]:
    if not 0 <= fields.risk_q8 <= 255:
        return "risk_q8_out_of_range"
    if not fields.risk_valid and fields.risk_q8 != 0:
        return "risk_q8_requires_risk_valid"
    if fields.risk_valid != fields.calibrated:
        return "risk_valid_must_equal_calibrated"
    if fields.local_prediction == LocalPrediction.RESERVED:
        return "reserved_local_prediction"
    if fields.review == Review.RESERVED:
        return "reserved_review"
    if fields.next_action == NextAction.DIRECT_CLOUD:
        if (fields.risk_q8 != 0 or fields.risk_valid or fields.calibrated
                or fields.local_prediction != LocalPrediction.UNKNOWN
                or fields.review != Review.NONE):
            return "direct_cloud_requires_no_edge_result"
    if fields.review == Review.REQUIRED and fields.next_action != NextAction.CLOUD_REVIEW:
        return "review_required_requires_cloud_review"
    if fields.next_action == NextAction.ACCEPT_LOCAL:
        if fields.edge_availability == EdgeAvailability.UNAVAILABLE:
            return "accept_local_requires_available_edge"
        if fields.local_prediction == LocalPrediction.UNKNOWN:
            return "accept_local_requires_prediction"
    if fields.safety == Safety.P0 and fields.next_action == NextAction.ACCEPT_LOCAL:
        return "p0_cannot_accept_local"
    if fields.edge_availability == EdgeAvailability.UNAVAILABLE:
        if fields.local_prediction != LocalPrediction.UNKNOWN:
            return "unavailable_edge_requires_unknown_prediction"
        if fields.risk_valid:
            return "unavailable_edge_cannot_claim_risk"
    if fields.risk_valid and fields.local_prediction == LocalPrediction.UNKNOWN:
        return "valid_risk_requires_prediction"
    return None


def encode_esw(fields: EswFields) -> int:
    """Return a canonical ESW-v1 word or raise ValueError for invalid semantics."""
    reason = _invalid(fields)
    if reason:
        raise ValueError(reason)
    return (
        fields.risk_q8
        | (int(fields.risk_valid) << 8)
        | (int(fields.calibrated) << 9)
        | (int(fields.local_prediction) << 10)
        | (int(fields.review) << 12)
        | (int(fields.safety) << 14)
        | (int(fields.edge_availability) << 16)
        | (int(fields.next_action) << 18)
        | (SCHEMA_VERSION << 24)
    )


def encode_esw_bytes(fields: EswFields) -> bytes:
    return encode_esw(fields).to_bytes(WIRE_SIZE_BYTES, byteorder="big")


def decode_esw(word: int) -> EswDecodeResult:
    if not 0 <= word <= 0xFFFFFFFF:
        return EswDecodeResult(DecodeStatus.INVALID, _fallback(), None, "word_out_of_range")
    version = (word >> 24) & 0x0F
    if version != SCHEMA_VERSION:
        return EswDecodeResult(DecodeStatus.UNSUPPORTED, _fallback(), word, "unknown_schema_version")
    if word & 0xF0F00000:
        return EswDecodeResult(DecodeStatus.INVALID, _fallback(), word, "nonzero_reserved_bits")
    try:
        fields = EswFields(
            risk_q8=word & 0xFF,
            risk_valid=bool((word >> 8) & 1),
            calibrated=bool((word >> 9) & 1),
            local_prediction=LocalPrediction((word >> 10) & 0x03),
            review=Review((word >> 12) & 0x03),
            safety=Safety((word >> 14) & 0x03),
            edge_availability=EdgeAvailability((word >> 16) & 0x03),
            next_action=NextAction((word >> 18) & 0x03),
        )
    except ValueError:
        return EswDecodeResult(DecodeStatus.INVALID, _fallback(), word, "invalid_enum")
    reason = _invalid(fields)
    if reason:
        return EswDecodeResult(DecodeStatus.INVALID, _fallback(), word, reason)
    return EswDecodeResult(DecodeStatus.OK, fields, word)


def decode_esw_bytes(payload: bytes) -> EswDecodeResult:
    if len(payload) != WIRE_SIZE_BYTES:
        return EswDecodeResult(DecodeStatus.INVALID, _fallback(), None, "invalid_packet_length")
    return decode_esw(int.from_bytes(payload, byteorder="big"))
