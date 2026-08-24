from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asset_routing_policy import AssetRoutingPolicy, AssetRoutingPolicyError  # noqa: E402


class AssetRoutingPolicyTest(unittest.TestCase):
    def test_empty_registry_keeps_all_assets_local_first(self):
        policy = AssetRoutingPolicy.from_json(ROOT / "config" / "asset_routing_v1.json")
        result = policy.decide({"data": {"device_id": "machine-01", "device_type": "milling"}})
        self.assertTrue(result.enabled)
        self.assertEqual(result.route, "LOCAL_FIRST")
        self.assertEqual(result.reason, "UNDESIGNATED_ASSET_LOCAL_FIRST")

    def test_only_registry_match_can_select_direct_cloud(self):
        policy = AssetRoutingPolicy({
            "schema_version": "asset-routing/1.0",
            "policy_version": "test-registry-v1",
            "enabled": True,
            "default_route": "LOCAL_FIRST",
            "designated_cloud_device_ids": ["line-a-controller"],
            "designated_cloud_device_types": [],
        })
        selected = policy.decide({
            "asset_importance": "important",
            "data": {"device_id": "line-a-controller"},
        })
        self.assertEqual(selected.route, "DIRECT_CLOUD")
        self.assertEqual(selected.matched_by, "device_id")
        ignored_claim = policy.decide({
            "asset_importance": "important",
            "data": {"device_id": "unknown-device"},
        })
        self.assertEqual(ignored_claim.route, "LOCAL_FIRST")

    def test_invalid_default_route_is_rejected(self):
        with self.assertRaises(AssetRoutingPolicyError):
            AssetRoutingPolicy({
                "schema_version": "asset-routing/1.0",
                "policy_version": "invalid",
                "enabled": True,
                "default_route": "DIRECT_CLOUD",
                "designated_cloud_device_ids": [],
                "designated_cloud_device_types": [],
            })
