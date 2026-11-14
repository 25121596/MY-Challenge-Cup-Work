"""Global device state store (in-memory).

Tracks the latest snapshot for every edge device that has sent a heartbeat.
Can be replaced with Redis or a database later.
"""

from __future__ import annotations

import time
import threading
from typing import Any


class DeviceState:
    """Thread-safe in-memory store for edge device state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # device_id → dict with last heartbeat + derived fields
        self._states: dict[str, dict[str, Any]] = {}
        # device_id → last_seen timestamp
        self._last_seen: dict[str, float] = {}
        # Conflict arbitration log
        self._conflict_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Heartbeat ingest
    # ------------------------------------------------------------------

    def update_from_heartbeat(self, payload: dict[str, Any]) -> None:
        """Ingest a heartbeat and update the device state snapshot."""
        device_id = payload.get("device_id", "")
        if not device_id:
            return

        # Flatten the heartbeat into a unified state dict
        cpu = payload.get("cpu", {})
        mem = payload.get("memory", {})
        net = payload.get("network", {})
        inf = payload.get("inference", {})
        gpus = payload.get("gpus", [])

        gpu_free_mb = 0.0
        gpu_total_mb = 0.0
        if gpus:
            gpu_free_mb = sum(g.get("free_mb", 0.0) for g in gpus)
            gpu_total_mb = sum(g.get("total_mb", 0.0) for g in gpus)

        with self._lock:
            self._states[device_id] = {
                "device_id": device_id,
                "device_type": payload.get("device_type", "unknown"),
                "timestamp_ms": payload.get("timestamp_ms", 0),
                "is_fluctuating": payload.get("is_fluctuating", False),
                # CPU
                "cpu_usage_pct": cpu.get("usage_pct", 0.0),
                "cpu_core_count": cpu.get("core_count", 0),
                # Memory
                "mem_total_mb": mem.get("total_mb", 0.0),
                "mem_avail_mb": mem.get("avail_mb", 0.0),
                # GPU
                "gpu_free_mb": gpu_free_mb,
                "gpu_total_mb": gpu_total_mb,
                # Network
                "rtt_ms": net.get("rtt_ms", 0.0),
                "packet_loss_pct": net.get("packet_loss_pct", 0.0),
                "cloud_reachable": net.get("cloud_reachable", True),
                # Inference
                "tps": inf.get("current_tps", 0.0),
                "task_queue_len": inf.get("task_queue_len", 0),
                # Extra
                "monitored_asset": payload.get("monitored_asset", ""),
                "camera_fov": payload.get("camera_fov", ""),
                "current_lora_version": payload.get("current_lora_version", 0),
                "model_version": payload.get("model_version", 0),
                # Remote IP (set by API layer if available)
                "remote_ip": payload.get("remote_ip", ""),
            }
            self._last_seen[device_id] = time.time()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, device_id: str) -> dict[str, Any] | None:
        """Return the latest state snapshot for *device_id*, or None."""
        with self._lock:
            return self._states.get(device_id)

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of all device states."""
        with self._lock:
            return dict(self._states)

    def last_seen(self, device_id: str) -> float | None:
        with self._lock:
            return self._last_seen.get(device_id)

    def online_device_ids(self) -> list[str]:
        """Return IDs of devices that have sent a heartbeat recently."""
        now = time.time()
        with self._lock:
            return [
                did
                for did, ts in self._last_seen.items()
                if now - ts < 120  # 2-minute timeout
            ]

    # ------------------------------------------------------------------
    # Neighbor info (derived from heartbeats)
    # ------------------------------------------------------------------

    def get_neighbor_info(self, device_id: str) -> dict[str, Any]:
        """Return neighbor-info dict suitable for the neighbors endpoint."""
        state = self.get(device_id)
        if state is None:
            return {}
        return {
            "node_id": device_id,
            "ip": state.get("remote_ip", ""),
            "udp_port": 15555,
            "tcp_port": 15556,
            "device_type": state.get("device_type", "unknown"),
            "asset": state.get("monitored_asset", ""),
            "fov": state.get("camera_fov", ""),
        }

    # ------------------------------------------------------------------
    # Conflict history
    # ------------------------------------------------------------------

    def log_conflict(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._conflict_log.append(entry)


# Singleton
device_state = DeviceState()
