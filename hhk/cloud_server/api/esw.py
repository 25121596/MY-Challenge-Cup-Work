"""Cloud admission endpoint for the ESW-v1 fast control-plane payload."""
from __future__ import annotations

import base64
import binascii
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

_ESW_DIR = Path(__file__).resolve().parents[2] / "edge-io-protocol"
if str(_ESW_DIR) not in sys.path:
    sys.path.insert(0, str(_ESW_DIR))

from esw_v1 import DecodeStatus, NextAction, decode_esw_bytes
from storage.models import EswEnvelope, EswEnvelopeResult


router = APIRouter(tags=["esw"])


@router.post("/api/v1/protocol/esw", response_model=EswEnvelopeResult)
async def decode_esw_envelope(body: EswEnvelope) -> EswEnvelopeResult:
    """Validate and decode ESW without replacing the paired JSON audit record."""
    try:
        payload = base64.b64decode(body.esw_payload_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="esw_payload_b64 must be base64") from exc

    decoded = decode_esw_bytes(payload)
    safe_action = NextAction.CLOUD_REVIEW
    policy_reason = decoded.reason
    fields = None
    if decoded.status == DecodeStatus.OK:
        fields = {
            "risk_q8": decoded.fields.risk_q8,
            "risk_valid": decoded.fields.risk_valid,
            "calibrated": decoded.fields.calibrated,
            "failure_probability": decoded.fields.risk_q8 / 255.0 if decoded.fields.risk_valid else None,
            "local_prediction": decoded.fields.local_prediction.name.lower(),
            "review": decoded.fields.review.name.lower(),
            "safety": decoded.fields.safety.name.lower(),
            "edge_availability": decoded.fields.edge_availability.name.lower(),
            "next_action": decoded.fields.next_action.name.lower(),
        }
        if decoded.fields.next_action == NextAction.UNKNOWN:
            policy_reason = "action_unknown"
        else:
            safe_action = decoded.fields.next_action

    return EswEnvelopeResult(
        device_id=body.device_id,
        request_id=body.request_id,
        trace_id=body.trace_id,
        audit_review_id=body.audit_review_id,
        esw_status=decoded.status.value,
        raw_word_hex=f"{decoded.raw_word:08X}" if decoded.raw_word is not None else None,
        rejection_reason=decoded.reason,
        cloud_action=safe_action.name.lower(),
        policy_reason=policy_reason,
        fields=fields,
    )
