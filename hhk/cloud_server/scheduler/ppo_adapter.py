"""Adapter to load and use trained PPO models from the phase-1 demo.

Wraps the existing ``PPOScheduler`` so the cloud server can optionally
use learned policies for task routing decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from config import config, DEMO_PROJECT_PATH

# Ensure the demo project is on sys.path
if str(DEMO_PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(DEMO_PROJECT_PATH))


class PPOAdapter:
    """Thin wrapper around the demo project's PPOScheduler.

    Handles lazy loading so the server starts even if stable-baselines3
    is not installed (falls back to RuleScheduler).
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or config.ppo_model_path
        self._scheduler: Any = None
        self._load_error: str | None = None
        self.available = False

    def _ensure_loaded(self) -> bool:
        if self._scheduler is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            from scheduler import PPOScheduler  # type: ignore[import-not-found]

            self._scheduler = PPOScheduler(
                model_path=self.model_path,
                deterministic=True,
            )
            self.available = True
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def predict(self, observation: Any) -> int:
        """Return 0=EDGE, 1=CLOUD, or -1 on error."""
        if not self._ensure_loaded():
            return -1

        try:
            import numpy as np

            action_index, _states = self._scheduler.model.predict(
                observation, deterministic=True
            )
            return int(np.asarray(action_index).item())
        except Exception:
            return -1

    @property
    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "model_path": self.model_path,
            "error": self._load_error,
        }
