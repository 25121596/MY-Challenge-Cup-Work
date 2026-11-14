"""State-provider interfaces shared by simulation and real deployment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from system_state import SystemState

if TYPE_CHECKING:
    from simulator import CloudEdgeSimulator


class StateProvider(ABC):
    @abstractmethod
    def get_state(self) -> SystemState:
        raise NotImplementedError


class SimStateProvider(StateProvider):
    """Adapter exposing simulator state through the common SystemState contract."""

    def __init__(self, simulator: "CloudEdgeSimulator") -> None:
        self.simulator = simulator

    def get_state(self) -> SystemState:
        return self.simulator.get_system_state()
