from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
SCHEDULER_ROOT = (REPOSITORY_ROOT / "scheduler_training").resolve()
VERSION = "0.4.1"

DEFAULT_POLICY_BUNDLE = Path(
    os.environ.get(
        "SCHEDULER_POLICY_BUNDLE",
        str(SCHEDULER_ROOT / "models" / "releases" / "pre_route_v2" / "policy_bundle.json"),
    )
).resolve()

DEFAULT_EDGE_INPUT = Path(
    os.environ.get(
        "EDGE_INPUT",
        str(SCHEDULER_ROOT / "data" / "sample_edge_inputs.json"),
    )
).resolve()

EDGE_ENDPOINT = os.environ.get("EDGE_ENDPOINT", "http://127.0.0.1:8080").rstrip("/")
CLOUD_ENDPOINT = os.environ.get("CLOUD_ENDPOINT", "http://127.0.0.1:8081").rstrip("/")

# Optional real-state endpoints.
# EDGE_HHK_STATE_URL can point to the metrics agent that receives HHK
# heartbeats, e.g. http://127.0.0.1:8091/api/v1/edge/state?device_id=edge-node-01
EDGE_HHK_STATE_URL = os.environ.get("EDGE_HHK_STATE_URL", "").strip()
EDGE_METRICS_URL = os.environ.get("EDGE_METRICS_URL", "").strip()
CLOUD_METRICS_URL = os.environ.get("CLOUD_METRICS_URL", "").strip()

REQUEST_TIMEOUT_S = int(os.environ.get("REQUEST_TIMEOUT_S", "180"))
METRICS_TIMEOUT_S = float(os.environ.get("METRICS_TIMEOUT_S", "1.5"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
FALLBACK_BANDWIDTH_MBPS = float(os.environ.get("FALLBACK_BANDWIDTH_MBPS", "25.0"))
NETWORK_PROBE_BYTES = int(os.environ.get("NETWORK_PROBE_BYTES", str(64 * 1024)))
STATE_SAMPLE_TTL_S = float(os.environ.get("STATE_SAMPLE_TTL_S", "0.25"))

DEFAULT_SEED = 42
DEFAULT_EPISODE_STEPS = 300
DEFAULT_QUEUE_POLICY = "edf"
DEFAULT_NETWORK_MODE = "markov"
DEFAULT_RESOURCE_MODE = "consume_recover"
DEFAULT_REWARD_MODE = "deadline_aware"
DEFAULT_EDGE_INPUT_INTERVAL_S = float(os.environ.get("EDGE_INPUT_INTERVAL_S", "0.05"))

TRACE_SCHEMA_VERSION = "cloud-edge-real-trace/2.0"
PROFILE_SAMPLE_INTERVAL_S = float(os.environ.get("PROFILE_SAMPLE_INTERVAL_S", "0.5"))
NETWORK_HISTORY_SIZE = int(os.environ.get("NETWORK_HISTORY_SIZE", "20"))
