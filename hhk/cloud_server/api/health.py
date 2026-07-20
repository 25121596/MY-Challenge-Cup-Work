"""`GET /api/v1/health` — Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from config import config
from inference.model_manager import model_manager
from scheduler.decision_engine import decision_engine
from storage.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Return server health status including load and model capabilities.

    Called by edge devices to verify cloud connectivity and discover the
    maximum split layer supported by the cloud model.
    """
    # Check model backend health (non-blocking for the mock backend)
    backend_ok = await model_manager.health()

    return HealthResponse(
        status="ok" if backend_ok else "degraded",
        load=decision_engine.load,
        version=config.version,
        max_split_layer=config.max_split_layer,
    )
