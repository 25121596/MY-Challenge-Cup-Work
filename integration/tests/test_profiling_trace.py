from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from profiling import RuntimeProfiler, TRACE_SCHEMA_VERSION, parse_llama_profile
from policy_bundle_loader import load_policy_bundle
from real_state_provider import NetworkMonitor, RealStateProvider
from real_ppo_runtime import CyclicTestPolicy, RealPPORuntime
from real_task_runtime import RealTaskManager
from runtime_task import RuntimeTask
from service_time_estimator import ServiceTimeEstimator
from tests.test_runtime_scheduler import StaticResourceSource, FakeNetworkMonitor
from trace_collector import TRACE_CSV_FIELDS, TraceCollector
from tests.bundle_fixture import create_test_bundle


class SequencedNetworkMonitor(NetworkMonitor):
    def __init__(self, rows):
        super().__init__(
            "http://fake",
            cloud_metrics_base_url=None,
            fallback_bandwidth_mbps=25.0,
            history_size=4,
        )
        self.rows = iter(rows)

    def _measure_rtt(self):
        return next(self.rows)


class ProfilingTraceTest(unittest.TestCase):
    def test_llama_profile_parser_supports_usage_and_timings(self):
        payload = {
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 55,
                "total_tokens": 75,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "timings": {
                "prompt_n": 20,
                "prompt_ms": 400.0,
                "prompt_per_second": 50.0,
                "predicted_n": 55,
                "predicted_ms": 2750.0,
                "predicted_per_second": 20.0,
            },
        }
        row = parse_llama_profile(payload, client_latency_ms=3200.0)
        self.assertEqual(row["prompt_tokens"], 20)
        self.assertEqual(row["completion_tokens"], 55)
        self.assertEqual(row["total_tokens"], 75)
        self.assertEqual(row["cached_tokens"], 4)
        self.assertEqual(row["prompt_tokens_per_s"], 50.0)
        self.assertEqual(row["completion_tokens_per_s"], 20.0)
        self.assertIsNone(row["ttft_ms"])
        self.assertEqual(row["ttft_source"], "not_reported_non_streaming")

    def test_network_window_calculates_jitter_and_failure_rate(self):
        monitor = SequencedNetworkMonitor([
            (True, 10.0),
            (True, 14.0),
            (False, 100.0),
            (True, 20.0),
        ])
        rows = [monitor.sample() for _ in range(4)]
        last = rows[-1]
        self.assertEqual(last.sample_count, 4)
        self.assertAlmostEqual(last.failure_rate, 0.25, places=6)
        # Successful RTT window [10,14,20] -> mean absolute delta=(4+6)/2=5ms.
        self.assertAlmostEqual(last.jitter_ms, 5.0, places=6)
        self.assertEqual(last.rtt_window_ms, [10.0, 14.0, 20.0])
        self.assertEqual(last.bandwidth_mbps, 25.0)

    def test_trace_collector_exports_stable_schema(self):
        collector = TraceCollector()
        record = {
            "task_id": "T-1",
            "task_type": "fault_diagnosis",
            "action_name": "LOCAL_FIRST",
            "final_route": "EDGE",
            "status": "completed",
            "real_total_wall_s": 0.5,
            "observation_24d": [0.0] * 24,
            "resource_snapshots": {"decision": {"state": {"edge_resource": 0.7}}},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jsonl = root / "trace.jsonl"
            csv = root / "trace.csv"
            summary = root / "summary.json"
            collector.write_jsonl(jsonl, [record])
            collector.write_csv(csv, [record])
            collector.write_summary(summary, [record], integration_version="0.4.0")

            jsonl_row = json.loads(jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(jsonl_row["schema_version"], TRACE_SCHEMA_VERSION)
            header = csv.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
            self.assertEqual(header, TRACE_CSV_FIELDS)
            summary_row = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(summary_row["schema_version"], TRACE_SCHEMA_VERSION)
            self.assertEqual(summary_row["record_count"], 1)

    def test_runtime_record_contains_stage_profiles_and_snapshots(self):
        raw = {
            "task_id": "PROFILE-001",
            "node_id": "edge-smoke",
            "task_type": "fault_diagnosis",
            "priority": "medium",
            "max_latency_ms": 1000,
            "data": {"sensor": 1, "value": 2.0},
        }
        task = RuntimeTask(
            task_id=1, arrival_time=0, size_mb=0.05, deadline_s=1.0, confidence=0.8,
            external_task_id=raw["task_id"], node_id=raw["node_id"], task_type=raw["task_type"],
            priority=raw["priority"], complexity=0.5, edge_base_latency_s=0.02,
            cloud_base_latency_s=0.02, collaborative_edge_latency_s=0.015,
            collaborative_cloud_latency_s=0.02, collaborative_accuracy=0.96,
            collaborative_payload_ratio=0.2, edge_input=raw, source="edge_input",
        )

        class ProfileClient:
            def __init__(self, name):
                self.name = name
            def chat(self, prompt, *, max_tokens=128, temperature=0.1):
                return {
                    "answer": f"{self.name}-ok",
                    "confidence": 0.95,
                    "level": "warning",
                    "category": "known_fault",
                    "output_valid": True,
                    "ood": False,
                    "profile": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cached_tokens": 0,
                        "prompt_ms": 10.0,
                        "completion_ms": 20.0,
                        "prompt_tokens_per_s": 1000.0,
                        "completion_tokens_per_s": 250.0,
                        "ttft_ms": None,
                    },
                }

        network = FakeNetworkMonitor()
        provider = RealStateProvider(
            edge_source=StaticResourceSource(20.0, 0.8),
            cloud_source=StaticResourceSource(30.0, 0.7),
            network_monitor=network,
            sample_ttl_s=0.0,
        )
        manager = RealTaskManager(
            edge_client=ProfileClient("edge"),
            cloud_client=ProfileClient("cloud"),
            network_monitor=network,
            estimator=ServiceTimeEstimator(alpha=0.5),
            timeout_scan_s=0.005,
        )
        profiler = RuntimeProfiler(provider, manager, interval_s=0.1)
        manager.set_profiler(profiler)
        bundle_temp = tempfile.TemporaryDirectory()
        bundle = load_policy_bundle(create_test_bundle(bundle_temp.name))
        runtime = RealPPORuntime(
            policy=CyclicTestPolicy((0,)),
            policy_bundle=bundle,
            state_provider=provider,
            task_manager=manager,
        )
        try:
            profiler.start()
            result = runtime.run_tasks([task], use_source_timestamps=False, drain_timeout_s=2.0, verbose=False)
            row = result.records[0]
            self.assertEqual(row["schema_version"], TRACE_SCHEMA_VERSION)
            self.assertEqual(row["edge_prompt_tokens"], 10)
            self.assertEqual(row["edge_completion_tokens"], 5)
            self.assertIn("decision", row["resource_snapshots"])
            self.assertIn("edge_start", row["resource_snapshots"])
            self.assertIn("edge_finish", row["resource_snapshots"])
        finally:
            profiler.stop()
            manager.close()
            bundle_temp.cleanup()


if __name__ == "__main__":
    unittest.main()
