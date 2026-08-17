from __future__ import annotations

"""Small stdlib HTTP sidecar for real-state/profiling collection.

Run this on a cloud/edge machine to expose local CPU/RAM/GPU metrics.  The same
agent can also receive the hhk heartbeat JSON at the exact hhk path
``/api/v1/edge/heartbeat`` and expose the latest snapshot back to the
orchestrator.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class HeartbeatStore:
    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.persist_path = None if persist_path is None else Path(persist_path)

    def put(self, payload: dict[str, Any]) -> None:
        device_id = str(payload.get("device_id", "")).strip()
        if not device_id:
            raise ValueError("heartbeat缺少device_id")
        row = dict(payload)
        row["receiver_timestamp_ms"] = int(time.time() * 1000)
        with self._lock:
            self._rows[device_id] = row
            if self.persist_path is not None:
                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                self.persist_path.write_text(
                    json.dumps(self._rows, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    def get(self, device_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if device_id:
                row = self._rows.get(device_id)
                return None if row is None else dict(row)
            if not self._rows:
                return None
            latest = max(
                self._rows.values(),
                key=lambda x: int(x.get("receiver_timestamp_ms", 0)),
            )
            return dict(latest)

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: dict(value) for key, value in self._rows.items()}


def _gpu_metrics() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            total, used, free, utilization = map(float, parts[1:5])
        except ValueError:
            continue
        rows.append(
            {
                "name": parts[0],
                "total_mb": total,
                "used_mb": used,
                "free_mb": free,
                "utilization_pct": utilization,
            }
        )
    return rows


def collect_local_metrics() -> dict[str, Any]:
    if psutil is not None:
        cpu_usage = float(psutil.cpu_percent(interval=0.05))
        memory = psutil.virtual_memory()
        total_mb = float(memory.total) / (1024.0 * 1024.0)
        avail_mb = float(memory.available) / (1024.0 * 1024.0)
        core_count = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    else:  # pragma: no cover - fallback for minimal installs
        core_count = int(os.cpu_count() or 1)
        try:
            load1 = float(os.getloadavg()[0])
            cpu_usage = min(100.0, max(0.0, load1 / max(core_count, 1) * 100.0))
        except (AttributeError, OSError):
            cpu_usage = 0.0
        total_mb = avail_mb = 0.0

    return {
        "timestamp_ms": int(time.time() * 1000),
        "cpu": {
            "usage_pct": round(cpu_usage, 3),
            "core_count": core_count,
        },
        "memory": {
            "avail_mb": round(avail_mb, 3),
            "total_mb": round(total_mb, 3),
        },
        "gpus": _gpu_metrics(),
    }


class MetricsAgentHandler(BaseHTTPRequestHandler):
    server_version = "CloudEdgeMetricsAgent/0.4"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        if getattr(self.server, "quiet", False):
            return
        super().log_message(format, *args)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(length) if length > 0 else b""

    @property
    def store(self) -> HeartbeatStore:
        return getattr(self.server, "heartbeat_store")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/health", "/api/v1/health"}:
            self._json(200, {"status": "ok", "version": "0.4"})
            return
        if parsed.path == "/metrics":
            self._json(200, collect_local_metrics())
            return
        if parsed.path == "/api/v1/edge/state":
            query = parse_qs(parsed.query)
            device_id = query.get("device_id", [None])[0]
            row = self.store.get(device_id)
            if row is None:
                self._json(404, {"error": "no heartbeat available"})
            else:
                self._json(200, row)
            return
        if parsed.path == "/api/v1/edge/states":
            self._json(200, {"devices": self.store.all()})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        body = self._read_body()
        if parsed.path == "/probe":
            self._json(
                200,
                {
                    "received_bytes": len(body),
                    "server_timestamp_ms": int(time.time() * 1000),
                },
            )
            return
        if parsed.path == "/api/v1/edge/heartbeat":
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
                if not isinstance(payload, dict):
                    raise ValueError("heartbeat必须为JSON object")
                self.store.put(payload)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"status": "accepted", "device_id": payload.get("device_id")})
            return
        self._json(404, {"error": "not found"})


def build_server(
    host: str,
    port: int,
    *,
    persist_path: str | Path | None = None,
    quiet: bool = False,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, int(port)), MetricsAgentHandler)
    setattr(server, "heartbeat_store", HeartbeatStore(persist_path))
    setattr(server, "quiet", bool(quiet))
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Integration real metrics/HHK heartbeat agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--persist-heartbeats", default="outputs/latest_hhk_heartbeats.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    server = build_server(
        args.host,
        args.port,
        persist_path=args.persist_heartbeats,
        quiet=args.quiet,
    )
    print(f"metrics agent listening on http://{args.host}:{args.port}")
    print("GET /metrics | POST /probe | POST /api/v1/edge/heartbeat")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
