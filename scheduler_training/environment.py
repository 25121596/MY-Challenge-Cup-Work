"""Phase4.5 dynamics facade.

The old monolithic EnvironmentDynamics is replaced by pluggable resource and
network dynamics.  This compatibility facade keeps old construction/property
access valid for auxiliary scripts while the simulator uses the separate APIs.
"""

from __future__ import annotations

import numpy as np

from dynamics import LegacyNetworkDynamics, LegacyResourceDynamics

VALID_RESOURCE_MODES = {"legacy", "consume_recover"}
VALID_NETWORK_MODES = {"legacy", "markov"}


class EnvironmentDynamics:
    def __init__(
        self,
        slot_duration: float = 0.05,
        resource_mode: str = "consume_recover",
        network_mode: str = "markov",
    ) -> None:
        if slot_duration <= 0:
            raise ValueError("slot_duration必须大于0")
        self.slot_duration = float(slot_duration)
        self.resource_mode = resource_mode
        self.network_mode = network_mode
        self.resources = LegacyResourceDynamics(resource_mode)
        self.network = LegacyNetworkDynamics(network_mode)

    @property
    def edge_resource(self) -> float:
        return self.resources.edge_resource

    @edge_resource.setter
    def edge_resource(self, value: float) -> None:
        self.resources.edge_resource = float(value)

    @property
    def cloud_resource(self) -> float:
        return self.resources.cloud_resource

    @cloud_resource.setter
    def cloud_resource(self, value: float) -> None:
        self.resources.cloud_resource = float(value)

    @property
    def network_quality(self) -> float:
        return self.network.network_quality

    @network_quality.setter
    def network_quality(self, value: float) -> None:
        self.network.network_quality = float(value)
        self.network.network_state = "GOOD" if value >= 0.5 else "BAD"

    @property
    def network_state(self) -> str:
        return self.network.network_state

    @property
    def bandwidth_mbps(self) -> float:
        return self.network.bandwidth_mbps

    # Parameter compatibility proxies used by regression checks.
    def __getattr__(self, name: str):
        if hasattr(self.resources, name):
            return getattr(self.resources, name)
        if hasattr(self.network, name):
            return getattr(self.network, name)
        raise AttributeError(name)

    def update(
        self,
        *,
        edge_busy: bool,
        cloud_busy: bool,
        edge_executed: bool,
        cloud_executed: bool,
        rng: np.random.Generator,
        delta_t_s: float | None = None,
    ) -> None:
        dt = self.slot_duration if delta_t_s is None else float(delta_t_s)
        # Phase4's consume/recover mode used executed flags; in continuous time
        # execution over an interval is equivalent to the node being busy.
        resource_edge_busy = edge_executed if self.resource_mode == "consume_recover" and delta_t_s is None else edge_busy
        resource_cloud_busy = cloud_executed if self.resource_mode == "consume_recover" and delta_t_s is None else cloud_busy
        self.resources.advance(
            delta_t_s=dt,
            edge_busy=resource_edge_busy,
            cloud_busy=resource_cloud_busy,
            rng=rng,
        )
        self.network.advance(delta_t_s=dt, rng=rng)

    def reset(self) -> None:
        self.resources.reset()
        self.network.reset()
