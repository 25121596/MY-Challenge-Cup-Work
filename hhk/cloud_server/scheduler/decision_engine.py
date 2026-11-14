"""CloudDecisionEngine — the central scheduling brain.

Receives heartbeats, updates global device state, and uses a combination of
rule-based logic (RuleScheduler from the demo project) and optional PPO to
decide what tasks to push to each edge device.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

import logging

from config import config, DEMO_PROJECT_PATH

# Ensure the demo project is on sys.path for imports
_demo_str = str(DEMO_PROJECT_PATH)
if _demo_str not in sys.path:
    sys.path.insert(0, _demo_str)

from storage.device_state import device_state
from storage.task_queue import task_queue_manager

logger = logging.getLogger("cloud_server.scheduler")


class CloudDecisionEngine:
    """Analyses device state and pushes tasks into per-device queues.

    Uses the existing RuleScheduler / PPOScheduler from the demo project
    for routing logic, plus heuristic rules for task generation.
    """

    def __init__(self) -> None:
        self._rule_scheduler: Any = None
        self._ppo_adapter: Any = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            from scheduler import RuleScheduler  # demo project

            self._rule_scheduler = RuleScheduler(
                confidence_threshold=config.confidence_threshold,
            )
        except Exception as exc:
            logger.warning("Could not load RuleScheduler: %s", exc)

        if config.use_ppo_scheduler:
            from scheduler.ppo_adapter import PPOAdapter

            self._ppo_adapter = PPOAdapter()
            if self._ppo_adapter._ensure_loaded():
                logger.info("PPO adapter loaded successfully")
            else:
                logger.warning(
                    "PPO adapter unavailable: %s", self._ppo_adapter._load_error
                )

        self._initialized = True

    # ------------------------------------------------------------------
    # Public API — called from the heartbeat handler
    # ------------------------------------------------------------------

    def on_heartbeat(self, device_id: str) -> None:
        """Process a heartbeat and generate tasks if appropriate."""
        self._ensure_initialized()
        state = device_state.get(device_id)
        if state is None:
            return

        decisions = self._make_decisions(device_id, state)
        for d in decisions:
            task_queue_manager.enqueue(
                device_id=device_id,
                task_type=d["type"],
                description=d.get("description", ""),
                priority=d.get("priority", 0),
                payload=d.get("payload", {}),
            )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _make_decisions(
        self, device_id: str, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate task decisions based on device state analysis.

        This is the core scheduling logic.  It examines the device snapshot
        and pushes tasks (cloud_review, rule_sync, model_update, etc.) when
        certain thresholds are crossed.
        """
        decisions: list[dict[str, Any]] = []

        rtt_ms = state.get("rtt_ms", 0.0)
        mem_avail_mb = state.get("mem_avail_mb", 0.0)
        tps = state.get("tps", 0.0)
        is_fluctuating = state.get("is_fluctuating", False)
        current_lora = state.get("current_lora_version", 0)

        # ── 1. Network degradation → switch to local-only mode ──
        if rtt_ms > config.network_degraded_rtt_ms:
            decisions.append({
                "type": "rule_sync",
                "description": "Network degraded — switch to local-only inference",
                "priority": 5,
                "payload": {
                    "rule_id": "network_degraded",
                    "rule_type": "override",
                    "target": "inference_mode",
                    "value": {"mode": "local_only"},
                },
            })

        # ── 2. Low memory → offload to cloud ──
        if mem_avail_mb < config.low_memory_threshold_mb:
            decisions.append({
                "type": "cloud_review",
                "description": "Low memory — offload inference to cloud",
                "priority": 4,
                "payload": {
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "route": "cloud",
                },
            })

        # ── 3. Low TPS → lower confidence threshold ──
        if tps < config.low_tps_threshold and tps > 0:
            decisions.append({
                "type": "rule_sync",
                "description": "Low TPS — relax confidence threshold",
                "priority": 3,
                "payload": {
                    "rule_id": "low_tps_adjust",
                    "rule_type": "threshold",
                    "target": "confidence_threshold",
                    "value": {"min": 0.65},
                },
            })

        # ── 4. Fluctuating device → adapt thresholds ──
        if is_fluctuating:
            decisions.append({
                "type": "rule_sync",
                "description": "Device fluctuating — lower anomaly threshold",
                "priority": 4,
                "payload": {
                    "rule_id": "fluctuation_adjust",
                    "rule_type": "threshold",
                    "target": "anomaly_score",
                    "value": {"min": 0.75, "max": 1.0},
                },
            })

        # ── 5. Low GPU memory → upload features instead of local infer ──
        gpu_free = state.get("gpu_free_mb", 0.0)
        gpu_total = state.get("gpu_total_mb", 1.0)
        if gpu_total > 0 and gpu_free / gpu_total < 0.2:
            decisions.append({
                "type": "upload_features",
                "description": "Low GPU memory — upload features to cloud",
                "priority": 3,
                "payload": {
                    "layer_idx": -1,
                    "targets": ["all"],
                },
            })

        # ── 6. Model version update ──
        if config.latest_lora_version > current_lora:
            decisions.append({
                "type": "model_update",
                "description": f"New LoRA v{config.latest_lora_version} available",
                "priority": 2,
                "payload": {
                    "url": f"{config.lora_base_url}/quality_v{config.latest_lora_version}.gguf",
                    "version": config.latest_lora_version,
                    "scale": 1.0,
                },
            })

        return decisions

    # ------------------------------------------------------------------
    # Route a *new inference task* using the scheduler (for simulator use)
    # ------------------------------------------------------------------

    def route_task(
        self, task: Any, state: dict[str, Any]
    ) -> tuple[int, str, dict[str, Any]]:
        """Use the rule or PPO scheduler to decide EDGE vs CLOUD for one task.

        Returns (action_int, reason, details).
        """
        self._ensure_initialized()

        if self._ppo_adapter and self._ppo_adapter.available:
            try:
                from observation import build_observation  # demo project

                obs = build_observation(task, state)
                action_idx = self._ppo_adapter.predict(obs)
                if action_idx == 0:
                    return 1, "PPO chose EDGE", {}
                elif action_idx == 1:
                    return 2, "PPO chose CLOUD", {}
            except Exception:
                pass  # fall through to rule scheduler

        if self._rule_scheduler is not None:
            return self._rule_scheduler.select_action(task, state)

        # Fallback: always EDGE
        return 1, "Fallback — default to EDGE", {}

    # ------------------------------------------------------------------
    # System load estimate
    # ------------------------------------------------------------------

    @property
    def load(self) -> float:
        """Rough load estimate: fraction of known devices with pending tasks."""
        online = device_state.online_device_ids()
        if not online:
            return 0.0
        pending_count = sum(
            1 for did in online if task_queue_manager.peek(did)
        )
        return round(pending_count / len(online) * 100, 1)

    def _new_task_id(self) -> str:
        return f"task-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


# Singleton
decision_engine = CloudDecisionEngine()
