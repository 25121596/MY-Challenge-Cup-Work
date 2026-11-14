"""`GET /api/v1/edge/neighbors` — Peer discovery endpoint.

Returns a list of other online edge devices in the same monitored area,
enabling P2P mesh formation.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from storage.device_state import device_state
from storage.models import NeighborInfo

logger = logging.getLogger("cloud_server.api.neighbors")

router = APIRouter(tags=["neighbors"])


@router.get("/api/v1/edge/neighbors", response_model=list[NeighborInfo])
async def get_neighbors(
    device_id: str = Query(..., description="Edge device requesting peers"),
):
    """Return online peer devices suitable for P2P communication.

    The cloud tracks which devices have recently sent heartbeats.  This
    endpoint returns all online peers except the requesting device.

    In production, this would be scoped to devices in the same monitored
    area (same asset/FOV).
    """
    if not device_id:
        return []

    requester = device_state.get(device_id)
    if requester is None:
        return []

    neighbors: list[NeighborInfo] = []
    online_ids = device_state.online_device_ids()

    for peer_id in online_ids:
        if peer_id == device_id:
            continue

        peer_state = device_state.get(peer_id)
        if peer_state is None:
            continue

        # If the requester has a monitored asset, scope to same asset
        if requester and requester.get("monitored_asset"):
            if peer_state.get("monitored_asset") != requester.get("monitored_asset"):
                continue

        info = device_state.get_neighbor_info(peer_id)
        if info:
            neighbors.append(NeighborInfo(**info))

    logger.debug(
        "Neighbors for %s: %d peers (of %d online)",
        device_id,
        len(neighbors),
        len(online_ids),
    )

    return neighbors
