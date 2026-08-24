"""Authoritative asset routing for the edge/cloud runtime.

Asset importance is operational metadata, not a model score and not a safety
priority. It is read only from versioned registry configuration; task inputs
cannot nominate themselves for cloud handling.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "asset-routing/1.0"


class AssetRoutingPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class AssetRoutingDecision:
    enabled: bool
    route: str | None
    reason: str
    matched_by: str | None
    policy_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "route": self.route,
            "reason": self.reason,
            "matched_by": self.matched_by,
            "policy_version": self.policy_version,
        }


class AssetRoutingPolicy:
    """Designated assets go to cloud; all other assets run at the edge first."""

    def __init__(self, payload: Mapping[str, Any], *, source_path: Path | None = None) -> None:
        self.payload = dict(payload)
        self.source_path = None if source_path is None else source_path.resolve()
        self._validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "AssetRoutingPolicy":
        source = Path(path).resolve()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetRoutingPolicyError(f"cannot read asset routing policy {source}: {exc}") from exc
        if not isinstance(payload, dict):
            raise AssetRoutingPolicyError("asset routing policy must be a JSON object")
        return cls(payload, source_path=source)

    @classmethod
    def disabled(cls) -> "AssetRoutingPolicy":
        return cls({
            "schema_version": SCHEMA_VERSION,
            "policy_version": "asset-routing-disabled",
            "enabled": False,
            "default_route": "LOCAL_FIRST",
            "designated_cloud_device_ids": [],
            "designated_cloud_device_types": [],
        })

    @property
    def policy_version(self) -> str:
        return str(self.payload["policy_version"])

    def _validate(self) -> None:
        if self.payload.get("schema_version") != SCHEMA_VERSION:
            raise AssetRoutingPolicyError(f"unsupported schema: {self.payload.get('schema_version')!r}")
        if not str(self.payload.get("policy_version", "")).strip():
            raise AssetRoutingPolicyError("policy_version is required")
        if not isinstance(self.payload.get("enabled"), bool):
            raise AssetRoutingPolicyError("enabled must be boolean")
        if self.payload.get("default_route") != "LOCAL_FIRST":
            raise AssetRoutingPolicyError("default_route must be LOCAL_FIRST")
        for key in ("designated_cloud_device_ids", "designated_cloud_device_types"):
            values = self.payload.get(key)
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise AssetRoutingPolicyError(f"{key} must be a list of non-empty strings")

    def decide(self, task_input: Mapping[str, Any]) -> AssetRoutingDecision:
        if not bool(self.payload["enabled"]):
            return AssetRoutingDecision(False, None, "ASSET_ROUTING_DISABLED", None, self.policy_version)
        data = task_input.get("data")
        data = data if isinstance(data, Mapping) else {}
        device_id = data.get("device_id")
        device_type = data.get("device_type")
        if isinstance(device_id, str) and device_id in self.payload["designated_cloud_device_ids"]:
            return AssetRoutingDecision(True, "DIRECT_CLOUD", "DESIGNATED_ASSET_DIRECT_CLOUD", "device_id", self.policy_version)
        if isinstance(device_type, str) and device_type in self.payload["designated_cloud_device_types"]:
            return AssetRoutingDecision(True, "DIRECT_CLOUD", "DESIGNATED_ASSET_DIRECT_CLOUD", "device_type", self.policy_version)
        return AssetRoutingDecision(True, "LOCAL_FIRST", "UNDESIGNATED_ASSET_LOCAL_FIRST", None, self.policy_version)
