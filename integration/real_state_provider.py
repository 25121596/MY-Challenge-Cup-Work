from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Protocol

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class ResourceMetrics:
    cpu_usage_pct: float = 0.0
    memory_available_mb: float = 0.0
    memory_total_mb: float = 0.0
    gpu_free_mb: float = 0.0
    gpu_total_mb: float = 0.0
    current_tps: float = 0.0
    gpu_usage_pct: float | None = None
    source: str = "unknown"
    timestamp_ms: int = 0

    @property
    def memory_available_ratio(self) -> float:
        if self.memory_total_mb <= 0:
            return 1.0
        return float(min(1.0, max(0.0, self.memory_available_mb / self.memory_total_mb)))

    @property
    def gpu_available_ratio(self) -> float | None:
        if self.gpu_total_mb <= 0:
            return None
        return float(min(1.0, max(0.0, self.gpu_free_mb / self.gpu_total_mb)))


@dataclass
class NetworkMetrics:
    reachable: bool = True
    rtt_ms: float = 0.0
    bandwidth_mbps: float = 25.0
    success_ratio: float = 1.0
    jitter_ms: float = 0.0
    failure_rate: float = 0.0
    packet_loss_pct: float = 0.0
    sample_count: int = 0
    rtt_window_ms: list[float] = field(default_factory=list)
    source: str = "fallback"
    timestamp_ms: int = 0


class RuntimeStateSource(Protocol):
    def queue_state(self) -> dict[str, Any]: ...


class LocalResourceSource:
    def __init__(self, label: str = "local") -> None:
        self.label = label

    @staticmethod
    def _gpu() -> tuple[float, float, float | None]:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free,utilization.gpu",
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
            return 0.0, 0.0, None
        if completed.returncode != 0:
            return 0.0, 0.0, None
        total = free = 0.0
        usages: list[float] = []
        for line in completed.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                total += float(parts[0])
                free += float(parts[1])
                usages.append(float(parts[2]))
            except ValueError:
                continue
        usage = None if not usages else sum(usages) / len(usages)
        return free, total, usage

    def sample(self) -> ResourceMetrics:
        if psutil is not None:
            cpu = float(psutil.cpu_percent(interval=0.02))
            mem = psutil.virtual_memory()
            avail = float(mem.available) / (1024.0 * 1024.0)
            total = float(mem.total) / (1024.0 * 1024.0)
        else:  # pragma: no cover
            cores = max(1, int(os.cpu_count() or 1))
            try:
                cpu = min(100.0, max(0.0, os.getloadavg()[0] / cores * 100.0))
            except (AttributeError, OSError):
                cpu = 0.0
            avail = total = 0.0
        gpu_free, gpu_total, gpu_usage = self._gpu()
        return ResourceMetrics(
            cpu_usage_pct=cpu,
            memory_available_mb=avail,
            memory_total_mb=total,
            gpu_free_mb=gpu_free,
            gpu_total_mb=gpu_total,
            gpu_usage_pct=gpu_usage,
            source=self.label,
            timestamp_ms=int(time.time() * 1000),
        )


class HttpResourceSource:
    def __init__(self, metrics_url: str, timeout_s: float = 1.5, label: str = "http") -> None:
        self.metrics_url = metrics_url.rstrip("/")
        if not self.metrics_url.endswith("/metrics"):
            self.metrics_url += "/metrics"
        self.timeout_s = float(timeout_s)
        self.label = label

    def sample(self) -> ResourceMetrics:
        with urllib.request.urlopen(self.metrics_url, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cpu = payload.get("cpu") or {}
        memory = payload.get("memory") or {}
        gpus = payload.get("gpus") or []
        gpu_total = sum(float(row.get("total_mb", 0.0) or 0.0) for row in gpus if isinstance(row, dict))
        gpu_free = sum(float(row.get("free_mb", 0.0) or 0.0) for row in gpus if isinstance(row, dict))
        gpu_usages = [float(row.get("utilization_pct", 0.0) or 0.0) for row in gpus if isinstance(row, dict)]
        gpu_usage = None if not gpu_usages else sum(gpu_usages) / len(gpu_usages)
        return ResourceMetrics(
            cpu_usage_pct=float(cpu.get("usage_pct", 0.0) or 0.0),
            memory_available_mb=float(memory.get("avail_mb", 0.0) or 0.0),
            memory_total_mb=float(memory.get("total_mb", 0.0) or 0.0),
            gpu_free_mb=gpu_free,
            gpu_total_mb=gpu_total,
            gpu_usage_pct=gpu_usage,
            source=self.label,
            timestamp_ms=int(payload.get("timestamp_ms", int(time.time() * 1000))),
        )


class HHKHeartbeatSource:
    """Reads the exact JSON emitted by hhk HeartbeatManager."""

    def __init__(self, state_url: str, timeout_s: float = 1.5) -> None:
        self.state_url = state_url
        self.timeout_s = float(timeout_s)
        self.last_network: NetworkMetrics | None = None

    def sample(self) -> ResourceMetrics:
        with urllib.request.urlopen(self.state_url, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cpu = payload.get("cpu") or {}
        memory = payload.get("memory") or {}
        inference = payload.get("inference") or {}
        network = payload.get("network") or {}
        gpus = payload.get("gpus") or []
        gpu_total = sum(float(row.get("total_mb", 0.0) or 0.0) for row in gpus if isinstance(row, dict))
        gpu_free = sum(float(row.get("free_mb", 0.0) or 0.0) for row in gpus if isinstance(row, dict))
        gpu_usages = [float(row.get("utilization_pct", 0.0) or 0.0) for row in gpus if isinstance(row, dict)]
        gpu_usage = None if not gpu_usages else sum(gpu_usages) / len(gpu_usages)
        rtt = float(network.get("rtt_ms", 0.0) or 0.0)
        reachable = bool(network.get("cloud_reachable", rtt >= 0))
        packet_loss_pct = float(network.get("packet_loss_pct", 0.0) or 0.0)
        success_ratio = max(0.0, min(1.0, 1.0 - packet_loss_pct / 100.0)) if reachable else 0.0
        self.last_network = NetworkMetrics(
            reachable=reachable,
            rtt_ms=max(0.0, rtt),
            bandwidth_mbps=25.0,
            success_ratio=success_ratio,
            failure_rate=1.0 - success_ratio,
            packet_loss_pct=packet_loss_pct,
            sample_count=1,
            rtt_window_ms=[max(0.0, rtt)] if reachable else [],
            source="hhk_heartbeat",
            timestamp_ms=int(payload.get("timestamp_ms", int(time.time() * 1000))),
        )
        return ResourceMetrics(
            cpu_usage_pct=float(cpu.get("usage_pct", 0.0) or 0.0),
            memory_available_mb=float(memory.get("avail_mb", 0.0) or 0.0),
            memory_total_mb=float(memory.get("total_mb", 0.0) or 0.0),
            gpu_free_mb=gpu_free,
            gpu_total_mb=gpu_total,
            current_tps=float(inference.get("current_tps", 0.0) or 0.0),
            gpu_usage_pct=gpu_usage,
            source="hhk_heartbeat",
            timestamp_ms=int(payload.get("timestamp_ms", int(time.time() * 1000))),
        )


class NetworkMonitor:
    def __init__(
        self,
        cloud_endpoint: str,
        *,
        cloud_metrics_base_url: str | None = None,
        fallback_bandwidth_mbps: float = 25.0,
        timeout_s: float = 1.5,
        probe_bytes: int = 64 * 1024,
        bandwidth_refresh_s: float = 10.0,
        history_size: int = 20,
    ) -> None:
        self.cloud_endpoint = cloud_endpoint.rstrip("/")
        self.cloud_metrics_base_url = (
            None if not cloud_metrics_base_url else cloud_metrics_base_url.rstrip("/")
        )
        self.fallback_bandwidth_mbps = float(fallback_bandwidth_mbps)
        self.timeout_s = float(timeout_s)
        self.probe_bytes = max(1024, int(probe_bytes))
        self.bandwidth_refresh_s = float(bandwidth_refresh_s)
        self._successes = 0
        self._attempts = 0
        self._last_bw = self.fallback_bandwidth_mbps
        self._last_bw_time = 0.0
        self._history = deque(maxlen=max(2, int(history_size)))
        self._lock = threading.Lock()

    def _measure_rtt(self) -> tuple[bool, float]:
        url = self.cloud_endpoint + "/health"
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as response:
                response.read(256)
            reachable = True
        except (urllib.error.URLError, TimeoutError, OSError):
            reachable = False
        rtt_ms = (time.perf_counter() - start) * 1000.0
        return reachable, rtt_ms

    def probe_payload(self, payload: bytes) -> tuple[float, float | None]:
        """POST actual bytes to the cloud metrics agent. Returns seconds, Mbps."""
        if self.cloud_metrics_base_url is None:
            start = time.perf_counter()
            self._measure_rtt()
            return time.perf_counter() - start, None
        url = self.cloud_metrics_base_url + "/probe"
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=max(self.timeout_s, 5.0)) as response:
            response.read()
        elapsed = max(1e-6, time.perf_counter() - start)
        mbps = len(payload) * 8.0 / elapsed / 1_000_000.0
        with self._lock:
            self._last_bw = max(0.1, float(mbps))
            self._last_bw_time = time.monotonic()
        return elapsed, mbps

    def _measure_bandwidth(self) -> float:
        now = time.monotonic()
        with self._lock:
            if now - self._last_bw_time < self.bandwidth_refresh_s:
                return self._last_bw
        if self.cloud_metrics_base_url is None:
            return self.fallback_bandwidth_mbps
        try:
            payload = b"0" * self.probe_bytes
            _, mbps = self.probe_payload(payload)
            return self.fallback_bandwidth_mbps if mbps is None else float(mbps)
        except Exception:
            return self.fallback_bandwidth_mbps

    def sample(self, hhk_network: NetworkMetrics | None = None) -> NetworkMetrics:
        external_packet_loss_pct = 0.0
        if hhk_network is not None:
            reachable = bool(hhk_network.reachable)
            rtt_ms = float(hhk_network.rtt_ms)
            source = hhk_network.source
            external_packet_loss_pct = float(getattr(hhk_network, "packet_loss_pct", 0.0) or 0.0)
        else:
            reachable, rtt_ms = self._measure_rtt()
            source = "http_health_probe"

        with self._lock:
            self._attempts += 1
            if reachable:
                self._successes += 1
            self._history.append((bool(reachable), float(rtt_ms)))
            recent = list(self._history)
            ratio = self._successes / max(1, self._attempts)

        rtt_window = [value for ok, value in recent if ok]
        if len(rtt_window) >= 2:
            jitter = sum(abs(b - a) for a, b in zip(rtt_window, rtt_window[1:])) / (len(rtt_window) - 1)
        else:
            jitter = 0.0
        recent_failure_rate = 1.0 - (sum(1 for ok, _ in recent if ok) / max(1, len(recent)))
        if external_packet_loss_pct > 0.0:
            recent_failure_rate = max(recent_failure_rate, min(1.0, external_packet_loss_pct / 100.0))

        bandwidth = self._measure_bandwidth() if reachable else 0.1
        return NetworkMetrics(
            reachable=reachable,
            rtt_ms=float(rtt_ms),
            bandwidth_mbps=float(bandwidth),
            success_ratio=float(ratio),
            jitter_ms=float(jitter),
            failure_rate=float(recent_failure_rate),
            packet_loss_pct=float(external_packet_loss_pct),
            sample_count=len(recent),
            rtt_window_ms=[float(v) for v in rtt_window],
            source=source,
            timestamp_ms=int(time.time() * 1000),
        )


class RealStateProvider:
    """Build the Scheduler state dict from real resources and runtime queues.

    The output keys intentionally match the 24D pre-route observation builder.
    Edge inference confidence is not part of this pre-route state.
    """

    def __init__(
        self,
        *,
        edge_source: Any,
        cloud_source: Any,
        network_monitor: NetworkMonitor,
        sample_ttl_s: float = 0.25,
    ) -> None:
        self.edge_source = edge_source
        self.cloud_source = cloud_source
        self.network_monitor = network_monitor
        self.sample_ttl_s = max(0.0, float(sample_ttl_s))
        self._lock = threading.Lock()
        self._last_sample_at = 0.0
        self._edge = ResourceMetrics(cpu_usage_pct=50.0, memory_available_mb=500.0, memory_total_mb=1000.0, source="initial")
        self._cloud = ResourceMetrics(cpu_usage_pct=50.0, memory_available_mb=500.0, memory_total_mb=1000.0, source="initial")
        self._network = NetworkMetrics()

    @staticmethod
    def resource_score(metrics: ResourceMetrics) -> float:
        cpu_available = min(1.0, max(0.0, 1.0 - metrics.cpu_usage_pct / 100.0))
        mem_available = metrics.memory_available_ratio
        gpu_available = metrics.gpu_available_ratio
        if gpu_available is None:
            score = 0.65 * cpu_available + 0.35 * mem_available
        else:
            score = 0.45 * cpu_available + 0.25 * mem_available + 0.30 * gpu_available
        return float(min(1.0, max(0.0, score)))

    @staticmethod
    def network_quality(metrics: NetworkMetrics) -> float:
        if not metrics.reachable:
            return 0.0
        # Smooth real-valued quality: 0 ms -> 1, 100 ms -> ~0.61, 500 ms -> ~0.08.
        rtt_component = math.exp(-max(0.0, metrics.rtt_ms) / 200.0)
        quality = 0.80 * rtt_component + 0.20 * min(1.0, max(0.0, metrics.success_ratio))
        return float(min(1.0, max(0.0, quality)))

    def _safe_resource(self, source: Any, fallback: ResourceMetrics) -> ResourceMetrics:
        try:
            return source.sample()
        except Exception:
            stale = ResourceMetrics(**asdict(fallback))
            stale.source = fallback.source + ":stale"
            return stale

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_sample_at < self.sample_ttl_s:
                return
            edge_previous = self._edge
            cloud_previous = self._cloud
        edge = self._safe_resource(self.edge_source, edge_previous)
        cloud = self._safe_resource(self.cloud_source, cloud_previous)
        hhk_network = getattr(self.edge_source, "last_network", None)
        try:
            network = self.network_monitor.sample(hhk_network=hhk_network)
        except Exception:
            network = self._network
        with self._lock:
            self._edge = edge
            self._cloud = cloud
            self._network = network
            self._last_sample_at = now

    def get_state(self, runtime: RuntimeStateSource, *, force_refresh: bool = False) -> dict[str, Any]:
        self.refresh(force=force_refresh)
        queue = runtime.queue_state()
        with self._lock:
            edge = self._edge
            cloud = self._cloud
            network = self._network
        return {
            "current_time": float(time.monotonic()),
            "current_time_s": float(time.monotonic()),
            "edge_queue": int(queue.get("edge_queue", 0)),
            "cloud_queue": int(queue.get("cloud_queue", 0)),
            "cloud_ready_queue": int(queue.get("cloud_ready_queue", queue.get("cloud_queue", 0))),
            "transmission_queue": int(queue.get("transmission_queue", 0)),
            "edge_busy": bool(queue.get("edge_busy", False)),
            "cloud_busy": bool(queue.get("cloud_busy", False)),
            "edge_workload_s": float(queue.get("edge_workload_s", 0.0)),
            "cloud_workload_s": float(queue.get("cloud_workload_s", 0.0)),
            "edge_resource": self.resource_score(edge),
            "cloud_resource": self.resource_score(cloud),
            "network_quality": self.network_quality(network),
            # pre-route-observation/2.0 saturates bandwidth values above 50 Mbps.
            "bandwidth_mbps": max(0.1, float(network.bandwidth_mbps)),
            "network_state": "GOOD" if self.network_quality(network) >= 0.5 else "BAD",
            "real_edge_metrics": asdict(edge),
            "real_cloud_metrics": asdict(cloud),
            "real_network_metrics": asdict(network),
            "state_source": "real",
        }
