"""Versioned edge-to-cloud escalation admission API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from storage.escalation_queue import escalation_queue
from storage.models import EscalationAck, EscalationRequest, ReviewFeedback, ReviewFeedbackAck


router = APIRouter(tags=["escalations"])


@router.post("/api/v1/escalations", response_model=EscalationAck,
             status_code=status.HTTP_202_ACCEPTED)
async def submit_escalation(body: EscalationRequest) -> EscalationAck:
    """Admit an edge review request with idempotency and deadline semantics."""
    record = escalation_queue.submit(body)
    return EscalationAck(**record)


@router.get("/api/v1/escalations/{review_id}")
async def get_escalation(review_id: str) -> dict:
    """Return admission and queue state; no inference completion is implied."""
    record = escalation_queue.get(review_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown review_id")
    return record


@router.post("/api/v1/escalations/{review_id}/feedback",
             response_model=ReviewFeedbackAck, status_code=status.HTTP_202_ACCEPTED)
async def submit_review_feedback(review_id: str, body: ReviewFeedback) -> ReviewFeedbackAck:
    """Store an explicit post-review label for later calibration analysis."""
    try:
        record = escalation_queue.add_feedback(review_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="unknown review_id")
    return ReviewFeedbackAck(**record)
