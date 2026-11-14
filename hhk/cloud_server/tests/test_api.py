"""Tests for the cloud server API endpoints.

Run with::

    cd D:\\llama.cpp\\cloud_server
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure cloud_server is importable
_SELF_DIR = Path(__file__).resolve().parent.parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture
def client():
    """Return an async client factory. Tests call ``async with client() as ac``."""
    def _make_client():
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")
    return _make_client


# ── 1. Health check ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health(client):
    async with client() as ac:
        resp = await ac.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "version" in data
    assert "max_split_layer" in data
    assert isinstance(data["load"], (int, float))


# ── 2. Heartbeat ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat(client):
    payload = {
        "device_id": "edge-test-01",
        "device_type": "jetson-orin-nano",
        "timestamp_ms": 1721200000000,
        "is_fluctuating": False,
        "cpu": {"usage_pct": 45.2, "core_count": 8, "freq_mhz": 0},
        "memory": {"total_mb": 8192, "avail_mb": 3200},
        "gpus": [
            {
                "name": "NVIDIA Orin",
                "total_mb": 4096,
                "free_mb": 2100,
                "used_mb": 1996,
                "device_type": 1,
            }
        ],
        "network": {
            "rtt_ms": 12,
            "packet_loss_pct": 0.0,
            "cloud_reachable": True,
        },
        "inference": {"current_tps": 45.3, "task_queue_len": 2},
    }
    async with client() as ac:
        resp = await ac.post("/api/v1/edge/heartbeat", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_heartbeat_missing_device_id(client):
    async with client() as ac:
        resp = await ac.post(
            "/api/v1/edge/heartbeat",
            json={"device_type": "test"},
        )
    assert resp.status_code == 422


# ── 3. Task polling ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tasks_empty(client):
    async with client() as ac:
        resp = await ac.get("/api/v1/edge/tasks", params={"device_id": "unknown"})
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_tasks_after_heartbeat_generates_tasks(client):
    payload = {
        "device_id": "edge-lowmem-01",
        "device_type": "jetson",
        "timestamp_ms": 1721200000000,
        "is_fluctuating": True,
        "cpu": {"usage_pct": 80.0, "core_count": 4},
        "memory": {"total_mb": 4096, "avail_mb": 512},
        "gpus": [],
        "network": {"rtt_ms": 600, "packet_loss_pct": 5.0, "cloud_reachable": True},
        "inference": {"current_tps": 5.0, "task_queue_len": 10},
    }
    async with client() as ac:
        await ac.post("/api/v1/edge/heartbeat", json=payload)
        resp = await ac.get(
            "/api/v1/edge/tasks", params={"device_id": "edge-lowmem-01"}
        )
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) > 0, f"Expected at least one task, got {tasks}"

    for task in tasks:
        assert "type" in task
        assert "task_id" in task
        assert task["type"] in (
            "cloud_review",
            "upload_features",
            "model_update",
            "rule_sync",
        )

    async with client() as ac:
        resp2 = await ac.get(
            "/api/v1/edge/tasks", params={"device_id": "edge-lowmem-01"}
        )
    assert resp2.json() == []


# ── 4. Split inference ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_split_inference(client):
    import base64

    dummy_data = b"\x00" * 8
    data_b64 = base64.b64encode(dummy_data).decode()

    payload = {
        "model": "qwen3-1.7b",
        "hidden": {
            "split_layer": 12,
            "current_heads": 16,
            "head_dim": 128,
            "batch_size": 1,
            "seq_len": 1,
            "dtype": "float32",
            "data_b64": data_b64,
            "data_bytes": 8,
            "checksum": "",
        },
        "context": {
            "n_past": 0,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_new_tokens": 16,
        },
        "device": {
            "device_id": "edge-01",
            "avail_mem_mb": 3200,
            "current_tps": 45.3,
            "network_rtt_ms": 12,
        },
    }
    async with client() as ac:
        resp = await ac.post("/api/v1/infer/split", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

    body = resp.text
    assert "data:" in body
    assert "[DONE]" in body


# ── 5. Feature upload ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_feature_upload(client):
    payload = {
        "prompt": "分析缺陷类型",
        "model": "qwen2-vl-7b",
        "temperature": 0.7,
        "max_tokens": 16,
        "stream": True,
        "media": {
            "type": "feature_vector",
            "width": 224,
            "height": 224,
            "channels": 3,
            "mime": "image/jpeg",
            "feature_shape": [49, 1024],
            "features_b64": "",
        },
    }
    async with client() as ac:
        resp = await ac.post("/api/v1/infer/features", json=payload)
    assert resp.status_code == 200
    assert "[DONE]" in resp.text


# ── 6. Chat completions ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_completions(client):
    payload = {
        "model": "deepseek-v2",
        "messages": [
            {"role": "system", "content": "你是一个工业质检助手"},
            {"role": "user", "content": "设备振动频率异常，是否停机？"},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 16,
        "stream": True,
    }
    async with client() as ac:
        resp = await ac.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "[DONE]" in resp.text


@pytest.mark.asyncio
async def test_chat_completions_validation(client):
    async with client() as ac:
        resp = await ac.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 422


# ── 7. Model updates ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_updates(client):
    async with client() as ac:
        resp = await ac.get(
            "/api/v1/edge/model-updates",
            params={
                "device_id": "edge-01",
                "current_lora_version": 0,
                "current_rule_version": 0,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "lora_updates" in data
    assert "rule_updates" in data
    assert isinstance(data["lora_updates"], list)
    assert isinstance(data["rule_updates"], list)


@pytest.mark.asyncio
async def test_model_updates_no_new(client):
    async with client() as ac:
        resp = await ac.get(
            "/api/v1/edge/model-updates",
            params={
                "device_id": "edge-01",
                "current_lora_version": 999,
                "current_rule_version": 999,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["lora_updates"] == []
    assert data["rule_updates"] == []


# ── 8. Neighbors ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_neighbors_empty(client):
    async with client() as ac:
        resp = await ac.get(
            "/api/v1/edge/neighbors", params={"device_id": "unknown"}
        )
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_neighbors_with_peers(client):
    hb1 = {
        "device_id": "edge-a",
        "device_type": "jetson",
        "timestamp_ms": 1721200000000,
        "cpu": {"usage_pct": 30, "core_count": 4},
        "memory": {"total_mb": 8192, "avail_mb": 4096},
        "gpus": [],
        "network": {"rtt_ms": 10, "packet_loss_pct": 0, "cloud_reachable": True},
        "inference": {"current_tps": 50, "task_queue_len": 0},
        "monitored_asset": "production_line_A",
    }
    hb2 = {**hb1, "device_id": "edge-b"}

    async with client() as ac:
        await ac.post("/api/v1/edge/heartbeat", json=hb1)
        await ac.post("/api/v1/edge/heartbeat", json=hb2)
        resp = await ac.get(
            "/api/v1/edge/neighbors", params={"device_id": "edge-a"}
        )
    assert resp.status_code == 200
    peers = resp.json()
    assert len(peers) >= 1
    assert any(p["node_id"] == "edge-b" for p in peers)


# ── 9. Conflicts ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conflict_arbitration(client):
    payload = {
        "conflict_id": "target_001_vs_edge-02",
        "target_id": "target_001",
        "target_type": "defect",
        "our_node_id": "edge-01",
        "our_decision": "critical",
        "our_confidence": 0.72,
        "peer_node_id": "edge-02",
        "peer_decision": "normal",
        "peer_confidence": 0.85,
        "reason": "class_mismatch",
        "severity": 3,
        "timestamp_ms": 1721200000000,
    }
    async with client() as ac:
        resp = await ac.post("/api/v1/edge/conflicts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "decision" in data
    assert "reasoning" in data
    assert data["decision"] in ("critical", "normal")


@pytest.mark.asyncio
async def test_conflict_large_confidence_gap(client):
    payload = {
        "conflict_id": "gap_test",
        "target_id": "t1",
        "target_type": "defect",
        "our_node_id": "edge-01",
        "our_decision": "normal",
        "our_confidence": 0.95,
        "peer_node_id": "edge-02",
        "peer_decision": "critical",
        "peer_confidence": 0.55,
        "reason": "class_mismatch",
        "severity": 2,
        "timestamp_ms": 1721200000000,
    }
    async with client() as ac:
        resp = await ac.post("/api/v1/edge/conflicts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "normal"


@pytest.mark.asyncio
async def test_conflict_safety_first(client):
    payload = {
        "conflict_id": "safety_test",
        "target_id": "t2",
        "target_type": "defect",
        "our_node_id": "edge-01",
        "our_decision": "critical",
        "our_confidence": 0.70,
        "peer_node_id": "edge-02",
        "peer_decision": "normal",
        "peer_confidence": 0.72,
        "reason": "class_mismatch",
        "severity": 3,
        "timestamp_ms": 1721200000000,
    }
    async with client() as ac:
        resp = await ac.post("/api/v1/edge/conflicts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "critical"


# ── Full integration smoke test ──────────────────────────────────


@pytest.mark.asyncio
async def test_full_flow(client):
    async with client() as ac:
        # 1. Health
        r = await ac.get("/api/v1/health")
        assert r.status_code == 200

        # 2. Heartbeat
        hb = {
            "device_id": "edge-fullflow",
            "device_type": "jetson-orin",
            "timestamp_ms": 1721200000000,
            "is_fluctuating": True,
            "cpu": {"usage_pct": 50, "core_count": 8},
            "memory": {"total_mb": 8192, "avail_mb": 500},
            "gpus": [{"name": "GPU", "total_mb": 4096, "free_mb": 500, "used_mb": 3596, "device_type": 1}],
            "network": {"rtt_ms": 600, "packet_loss_pct": 5, "cloud_reachable": True},
            "inference": {"current_tps": 5, "task_queue_len": 3},
        }
        r = await ac.post("/api/v1/edge/heartbeat", json=hb)
        assert r.status_code == 200

        # 3. Poll tasks
        r = await ac.get("/api/v1/edge/tasks", params={"device_id": "edge-fullflow"})
        assert r.status_code == 200
        tasks = r.json()
        assert len(tasks) > 0
        types = {t["type"] for t in tasks}
        assert "cloud_review" in types or "rule_sync" in types

        # 4. Chat
        r = await ac.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8,
            },
        )
        assert r.status_code == 200
        assert "[DONE]" in r.text

        # 5. Model updates
        r = await ac.get(
            "/api/v1/edge/model-updates",
            params={"device_id": "edge-fullflow", "current_lora_version": 0},
        )
        assert r.status_code == 200

        # 6. Neighbors
        r = await ac.get(
            "/api/v1/edge/neighbors", params={"device_id": "edge-fullflow"}
        )
        assert r.status_code == 200

        # 7. Conflict
        r = await ac.post(
            "/api/v1/edge/conflicts",
            json={
                "conflict_id": "c1",
                "target_id": "t1",
                "our_node_id": "n1",
                "our_decision": "critical",
                "our_confidence": 0.9,
                "peer_node_id": "n2",
                "peer_decision": "normal",
                "peer_confidence": 0.6,
                "reason": "class_mismatch",
                "severity": 3,
            },
        )
        assert r.status_code == 200
