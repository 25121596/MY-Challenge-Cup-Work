import base64
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import app


def envelope(payload: bytes) -> dict:
    return {
        "schema_version": "edge-esw-envelope/v1",
        "device_id": "edge-01",
        "request_id": "request-1",
        "trace_id": "trace-1",
        "audit_review_id": "review-1",
        "esw_payload_b64": base64.b64encode(payload).decode("ascii"),
    }


@pytest.mark.asyncio
async def test_valid_esw_decodes_and_routes_to_review():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/protocol/esw", json=envelope(bytes.fromhex("010DAB80")))
    assert response.status_code == 200
    body = response.json()
    assert body["esw_status"] == "ESW_OK"
    assert body["cloud_action"] == "cloud_review"
    assert body["fields"]["failure_probability"] == 128 / 255
    assert body["fields"]["local_prediction"] == "fault"


@pytest.mark.asyncio
async def test_invalid_esw_fails_closed_without_decoded_fields():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/protocol/esw", json=envelope(bytes.fromhex("01100000")))
    assert response.status_code == 200
    body = response.json()
    assert body["esw_status"] == "ESW_INVALID"
    assert body["rejection_reason"] == "nonzero_reserved_bits"
    assert body["cloud_action"] == "cloud_review"
    assert body["fields"] is None


@pytest.mark.asyncio
async def test_unknown_action_is_valid_but_routes_safely():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/v1/protocol/esw", json=envelope(bytes.fromhex("01000000")))
    assert response.status_code == 200
    body = response.json()
    assert body["esw_status"] == "ESW_OK"
    assert body["policy_reason"] == "action_unknown"
    assert body["cloud_action"] == "cloud_review"
