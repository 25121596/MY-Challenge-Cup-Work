"""Cloud server configuration.

All settings can be overridden via environment variables prefixed with ``CLOUD_``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(f"CLOUD_{key}", default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(f"CLOUD_{key}", str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(f"CLOUD_{key}", str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(f"CLOUD_{key}", "").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


# Path to the existing demo project (scheduler logic).
DEMO_PROJECT_PATH = Path(
    os.environ.get(
        "CLOUD_DEMO_PATH",
        r"D:\大学\揭榜挂帅\cloud_edge_demo_phase1",
    )
)


@dataclass
class Config:
    """Server-wide configuration."""

    # --- Network ---
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8080))
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))

    # --- Model backend ---
    # Supported: "llama", "vllm", "mock"
    model_backend: str = field(
        default_factory=lambda: _env("MODEL_BACKEND", "mock")
    )
    # When backend is "llama", this is the base URL of llama-server
    llama_server_url: str = field(
        default_factory=lambda: _env("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
    )
    # When backend is "vllm"
    vllm_server_url: str = field(
        default_factory=lambda: _env("VLLM_SERVER_URL", "http://127.0.0.1:8000")
    )

    # Default model name to advertise
    default_model: str = field(
        default_factory=lambda: _env("DEFAULT_MODEL", "qwen3-1.7b")
    )

    # --- Scheduling ---
    # Confidence threshold for edge-local inference
    confidence_threshold: float = field(
        default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.78)
    )
    # Max split layer advertised to edges
    max_split_layer: int = field(
        default_factory=lambda: _env_int("MAX_SPLIT_LAYER", 28)
    )

    # PPO scheduling (optional)
    ppo_model_path: str = field(
        default_factory=lambda: _env(
            "PPO_MODEL_PATH",
            str(DEMO_PROJECT_PATH / "models" / "ppo_cloud_edge.zip"),
        )
    )
    use_ppo_scheduler: bool = field(
        default_factory=lambda: _env_bool("USE_PPO_SCHEDULER", False)
    )

    # --- Task generation thresholds ---
    network_degraded_rtt_ms: float = field(
        default_factory=lambda: _env_float("NETWORK_DEGRADED_RTT_MS", 500.0)
    )
    low_memory_threshold_mb: float = field(
        default_factory=lambda: _env_float("LOW_MEMORY_THRESHOLD_MB", 1024.0)
    )
    low_tps_threshold: float = field(
        default_factory=lambda: _env_float("LOW_TPS_THRESHOLD", 10.0)
    )
    max_escalation_queue: int = field(
        default_factory=lambda: _env_int("MAX_ESCALATION_QUEUE", 1000)
    )

    # --- LoRA updates ---
    lora_base_url: str = field(
        default_factory=lambda: _env(
            "LORA_BASE_URL", "http://localhost:8080/models/loras"
        )
    )
    latest_lora_version: int = field(
        default_factory=lambda: _env_int("LATEST_LORA_VERSION", 3)
    )

    # --- Version ---
    version: str = field(default_factory=lambda: _env("VERSION", "1.2.0"))


# Singleton
config = Config()
