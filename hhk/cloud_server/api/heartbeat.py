"""`POST /api/v1/edge/heartbeat` — Receive edge device status."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from storage.device_state import device_state
from storage.models import HeartbeatPayload
from scheduler.decision_engine import decision_engine

logger = logging.getLogger("cloud_server.api.heartbeat")

router = APIRouter(tags=["heartbeat"])


@router.post("/api/v1/edge/heartbeat")
async def heartbeat(request: Request, payload: HeartbeatPayload):
    """Ingest a heartbeat from an edge device.

    The payload contains CPU, memory, GPU, network, and inference metrics.
    The cloud scheduler analyses this data and may push tasks into the
    device's task queue.

    Returns 200 OK on success, 503 if the cloud is overloaded.
    """
    try:
        body = payload.model_dump()

        # Attach the remote IP for neighbor discovery
        client_ip = request.client.host if request.client else ""
        body["remote_ip"] = client_ip

        # Update global device state
        device_state.update_from_heartbeat(body)

        # Trigger scheduling decisions
        decision_engine.on_heartbeat(payload.device_id)

        logger.debug(
            "Heartbeat from %s: cpu=%s%%, mem=%s MB, tps=%s, rtt=%s ms",
            payload.device_id,
            payload.cpu.usage_pct,
            payload.memory.avail_mb,
            payload.inference.current_tps,
            payload.network.rtt_ms,
        )

        return Response(status_code=200)

    except Exception as exc:
        logger.exception("Heartbeat processing failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"error": "Service Unavailable", "detail": str(exc)},
        )
