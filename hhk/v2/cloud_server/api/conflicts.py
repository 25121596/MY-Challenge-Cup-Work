"""`POST /api/v1/edge/conflicts` — Conflict arbitration endpoint.

When two edge devices disagree about a classification/detection and
cannot resolve the conflict locally (4-level rule cascade exhausted),
they escalate to the cloud for final arbitration.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from storage.device_state import device_state
from storage.models import ConflictArbitrationRequest, ConflictArbitrationResponse

logger = logging.getLogger("cloud_server.api.conflicts")

router = APIRouter(tags=["conflicts"])


@router.post(
    "/api/v1/edge/conflicts",
    response_model=ConflictArbitrationResponse,
)
async def conflict_arbitration(
    request: Request, body: ConflictArbitrationRequest
):
    """Arbitrate a conflict escalated by an edge device.

    The edge sends both its own classification and the peer's conflicting
    classification.  The cloud resolves the conflict using one of these
    strategies:

    1. **Confidence comparison** — adopt the decision from the node with
       the higher confidence score, so long as the gap is meaningful.
    2. **Sensor authority** — a node with a higher-resolution or
       specialized sensor overrides a generic one.
    3. **Cloud model review** — if available, re-run inference with a
       larger cloud model for a definitive answer.

    Returns the final decision and reasoning.
    """
    logger.info(
        "Conflict arbitration: id=%s, target=%s, %s(%s@%.2f) vs %s(%s@%.2f), reason=%s",
        body.conflict_id,
        body.target_id,
        body.our_node_id,
        body.our_decision,
        body.our_confidence,
        body.peer_node_id,
        body.peer_decision,
        body.peer_confidence,
        body.reason,
    )

    decision, reasoning, confidence = _arbitrate(body)

    # Log for audit trail
    device_state.log_conflict({
        **body.model_dump(),
        "resolution": decision,
        "reasoning": reasoning,
        "resolved_at": time.time(),
    })

    return ConflictArbitrationResponse(
        decision=decision,
        reasoning=reasoning,
        confidence=confidence,
    )


def _arbitrate(
    body: ConflictArbitrationRequest,
) -> tuple[str, str, float]:
    """Core arbitration logic.

    Strategy cascade:
    1. Confidence gap ≥ 0.15 → higher-confidence node wins.
    2. class_mismatch with close confidence → prefer "critical" (safety-first).
    3. Default → adopt the higher-confidence decision.
    """
    gap = abs(body.our_confidence - body.peer_confidence)

    # ── Rule 1: Large confidence gap ──
    if gap >= 0.15:
        if body.our_confidence > body.peer_confidence:
            winner = body.our_node_id
            decision = body.our_decision
            conf = body.our_confidence
        else:
            winner = body.peer_node_id
            decision = body.peer_decision
            conf = body.peer_confidence
        return (
            decision,
            f"Confidence gap ({gap:.2f}) resolves to {winner} (confidence={conf:.2f})",
            conf,
        )

    # ── Rule 2: Safety-first for close calls ──
    safety_order = {"critical": 3, "warning": 2, "normal": 1, "unknown": 0}

    our_safety = safety_order.get(body.our_decision.lower(), 0)
    peer_safety = safety_order.get(body.peer_decision.lower(), 0)

    if body.reason == "class_mismatch" and gap < 0.15:
        # If one says critical/warning and the other normal, err on the
        # side of caution
        if our_safety > peer_safety:
            return (
                body.our_decision,
                f"Safety-first: adopting {body.our_node_id}'s higher-severity decision ({body.our_decision} > {body.peer_decision})",
                body.our_confidence,
            )
        elif peer_safety > our_safety:
            return (
                body.peer_decision,
                f"Safety-first: adopting {body.peer_node_id}'s higher-severity decision ({body.peer_decision} > {body.our_decision})",
                body.peer_confidence,
            )

    # ── Rule 3: Higher confidence wins ──
    if body.our_confidence >= body.peer_confidence:
        return (
            body.our_decision,
            f"Default: {body.our_node_id} has higher confidence ({body.our_confidence:.2f} >= {body.peer_confidence:.2f})",
            body.our_confidence,
        )
    else:
        return (
            body.peer_decision,
            f"Default: {body.peer_node_id} has higher confidence ({body.peer_confidence:.2f} > {body.our_confidence:.2f})",
            body.peer_confidence,
        )
