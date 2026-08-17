from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from observation import OBSERVATION_NAMES
from ppo_core import CloudEdgePPOCore


class CloudEdgeGymEnv(gym.Env):
    """Phase4.5 pre-route Gymnasium wrapper (24D / Discrete(2))."""

    metadata = {"render_modes": []}

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.core = CloudEdgePPOCore(**kwargs)
        # Gym indices: 0=LOCAL_FIRST, 1=DIRECT_CLOUD.
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(OBSERVATION_NAMES),),
            dtype=np.float32,
        )

    @property
    def simulator(self):
        return self.core.simulator

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        return self.core.reset(seed=seed)

    def step(self, action: int):
        return self.core.step(int(action))

    def close(self) -> None:
        self.core.simulator = None
        self.core.context = None
