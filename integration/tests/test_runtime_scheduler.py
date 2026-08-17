from __future__ import annotations

import sys
from pathlib import Path
import importlib.util
import json
import threading
import time
import tempfile
from types import SimpleNamespace
import numpy as np
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_input_loader import load_edge_inputs  # noqa: E402
from policy_bundle_loader import PolicyBundleError, load_policy_bundle  # noqa: E402
from pre_route_observation import build_pre_route_observation  # noqa: E402
from real_ppo_runtime import CyclicTestPolicy, RealPPORuntime  # noqa: E402
from real_state_provider import (  # noqa: E402
    HHKHeartbeatSource,
    NetworkMetrics,
    RealStateProvider,
    ResourceMetrics,
)
from real_metrics_agent import build_server  # noqa: E402
from real_task_runtime import EDFTaskQueue, RealTaskEnvelope, RealTaskManager  # noqa: E402
from runtime_task import RuntimeTask  # noqa: E402
from service_time_estimator import ServiceTimeEstimator  # noqa: E402
from tests.bundle_fixture import create_test_bundle  # noqa: E402


class StaticResourceSource:
    def __init__(self, cpu: float, mem_ratio: float, gpu_ratio: float | None = None):
        self.cpu = cpu
        self.mem_ratio = mem_ratio
        self.gpu_ratio = gpu_ratio
        self.last_network = None

    def sample(self):
        total = 1000.0
        gpu_total = 0.0 if self.gpu_ratio is None else 1000.0
        gpu_free = 0.0 if self.gpu_ratio is None else gpu_total * self.gpu_ratio
        return ResourceMetrics(
            cpu_usage_pct=self.cpu,
            memory_available_mb=total * self.mem_ratio,
            memory_total_mb=total,
            gpu_free_mb=gpu_free,
            gpu_total_mb=gpu_total,
            source="static",
            timestamp_ms=int(time.time() * 1000),
        )


class FakeNetworkMonitor:
    def __init__(self, *, reachable: bool = True):
        self.reachable = reachable

    def sample(self, hhk_network=None):
        return NetworkMetrics(
            reachable=self.reachable,
            rtt_ms=20.0,
            bandwidth_mbps=30.0,
            success_ratio=1.0,
            source="fake",
            timestamp_ms=int(time.time() * 1000),
        )

    def probe_payload(self, payload: bytes):
        time.sleep(0.004)
        return 0.004, max(1.0, len(payload) * 8 / 0.004 / 1_000_000)


class FailingTransferNetwork(FakeNetworkMonitor):
    def probe_payload(self, payload: bytes):
        raise RuntimeError("simulated transfer failure")


class FakeModelClient:
    def __init__(
        self,
        name: str,
        delay_s: float = 0.015,
        confidences: tuple[float, ...] = (0.95,),
    ):
        self.name = name
        self.delay_s = delay_s
        self.confidences = confidences
        self.calls: list[str] = []

    def chat(self, prompt: str, *, max_tokens=128, temperature=0.1):
        self.calls.append(prompt)
        start = time.perf_counter()
        time.sleep(self.delay_s)
        confidence = self.confidences[(len(self.calls) - 1) % len(self.confidences)]
        return {
            "endpoint": f"fake://{self.name}",
            "model": self.name,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "answer": f"{self.name}-ok",
            "confidence": confidence,
            "level": "warning",
            "category": "known_fault",
            "output_valid": True,
            "ood": False,
            "usage": {},
        }


class EmptyRuntimeState:
    def queue_state(self):
        return {
            "edge_queue": 0,
            "cloud_queue": 0,
            "cloud_ready_queue": 0,
            "transmission_queue": 0,
            "edge_busy": False,
            "cloud_busy": False,
            "edge_workload_s": 0.0,
            "cloud_workload_s": 0.0,
        }


class RuntimeSchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._bundle_temp = tempfile.TemporaryDirectory()
        cls.bundle = load_policy_bundle(create_test_bundle(cls._bundle_temp.name))

    @classmethod
    def tearDownClass(cls):
        cls._bundle_temp.cleanup()

    def make_task(self, index: int, deadline_s: float = 1.0):
        task_type = (
            "fault_diagnosis",
            "quality_inspection",
            "safety_check",
        )[index % 3]
        raw = {
            "task_id": f"SMOKE-{index:03d}",
            "node_id": "edge-smoke",
            "task_type": task_type,
            "priority": "high" if index % 2 else "medium",
            "max_latency_ms": int(deadline_s * 1000),
            "data": {"sensor": index, "value": 1.2 + index},
        }
        return RuntimeTask(
            task_id=index + 1,
            arrival_time=0,
            size_mb=0.05,
            deadline_s=deadline_s,
            confidence=0.8,
            external_task_id=raw["task_id"],
            node_id=raw["node_id"],
            task_type=task_type,
            priority=raw["priority"],
            complexity=0.5,
            edge_base_latency_s=0.02,
            cloud_base_latency_s=0.02,
            collaborative_edge_latency_s=0.015,
            collaborative_cloud_latency_s=0.02,
            collaborative_accuracy=0.96,
            collaborative_payload_ratio=0.2,
            edge_input=raw,
            source="edge_input",
        )

    def test_integration_input_loader_builds_runtime_tasks(self):
        payload = [{
            "task_id": "INPUT-001",
            "node_id": "edge-smoke",
            "task_type": "fault_diagnosis",
            "priority": "high",
            "max_latency_ms": 500,
            "payload_size_mb": 0.25,
            "data": {"sensor": 1, "value": 2.5},
        }]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tasks.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            tasks, report = load_edge_inputs(path)
        self.assertEqual(report.loaded_count, 1)
        self.assertIsInstance(tasks[0], RuntimeTask)
        self.assertEqual(tasks[0].external_task_id, "INPUT-001")
        self.assertEqual(tasks[0].deadline_s, 0.5)
        self.assertEqual(tasks[0].size_mb, 0.25)

    def test_real_state_builds_scheduler_24d_observation(self):
        provider = RealStateProvider(
            edge_source=StaticResourceSource(25.0, 0.75, 0.70),
            cloud_source=StaticResourceSource(40.0, 0.60, 0.50),
            network_monitor=FakeNetworkMonitor(),
            sample_ttl_s=0.0,
        )
        state = provider.get_state(EmptyRuntimeState(), force_refresh=True)
        observation = build_pre_route_observation(
            self.make_task(0),
            state,
            observation_names=self.bundle.observation_names,
        )
        self.assertEqual(observation.shape, (24,))
        self.assertGreater(state["edge_resource"], 0.0)
        self.assertGreater(state["network_quality"], 0.0)
        self.assertEqual(state["bandwidth_mbps"], 30.0)

        scheduler_dir = ROOT.parent / "scheduler_training"
        scheduler_path = str(scheduler_dir)
        isolated_names = ("config", "models", "task_profiles", "observation")
        saved_modules = {name: sys.modules.pop(name, None) for name in isolated_names}
        sys.path.insert(0, scheduler_path)
        try:
            module = importlib.import_module("observation")
            scheduler_observation = module.build_observation(self.make_task(0), state)
            np.testing.assert_allclose(observation, scheduler_observation, rtol=0.0, atol=1e-7)
        finally:
            for name in isolated_names:
                sys.modules.pop(name, None)
            sys.path.remove(scheduler_path)
            for name, saved in saved_modules.items():
                if saved is not None:
                    sys.modules[name] = saved

    def test_edf_queue_orders_by_absolute_deadline(self):
        queue = EDFTaskQueue("test")
        now = time.monotonic()
        long_task = self.make_task(1, deadline_s=2.0)
        short_task = self.make_task(2, deadline_s=0.5)
        e1 = RealTaskEnvelope(long_task, long_task.edge_input, now, 1, now + 2.0)
        e2 = RealTaskEnvelope(short_task, short_task.edge_input, now, 2, now + 0.5)
        queue.put(e1)
        queue.put(e2)
        snapshot = queue.snapshot()
        self.assertEqual([row.task_id for row in snapshot], ["SMOKE-002", "SMOKE-001"])
        queue.close()

    def test_two_actions_produce_edge_cloud_and_post_edge_collaborative_routes(self):
        edge = FakeModelClient("edge", 0.015, confidences=(0.95, 0.40))
        cloud = FakeModelClient("cloud", 0.018)
        network = FakeNetworkMonitor()
        provider = RealStateProvider(
            edge_source=StaticResourceSource(20.0, 0.8),
            cloud_source=StaticResourceSource(25.0, 0.75),
            network_monitor=network,
            sample_ttl_s=0.0,
        )
        manager = RealTaskManager(
            edge_client=edge,
            cloud_client=cloud,
            network_monitor=network,
            estimator=ServiceTimeEstimator(alpha=0.5),
            timeout_scan_s=0.005,
        )
        runtime = RealPPORuntime(
            policy=CyclicTestPolicy((0, 1, 0)),
            policy_bundle=self.bundle,
            state_provider=provider,
            task_manager=manager,
        )
        tasks = [self.make_task(i, deadline_s=0.8) for i in range(6)]
        try:
            result = runtime.run_tasks(
                tasks,
                arrival_interval_s=0.006,
                use_source_timestamps=False,
                drain_timeout_s=3.0,
                verbose=False,
            )
            self.assertEqual(result.submitted_count, 6)
            self.assertEqual(result.completed_count, 6)
            self.assertEqual(result.timeout_count, 0)
            self.assertEqual(
                {r["action_name"] for r in result.records},
                {"LOCAL_FIRST", "DIRECT_CLOUD"},
            )
            self.assertEqual(
                {r["final_route"] for r in result.records},
                {"EDGE", "CLOUD", "COLLABORATIVE"},
            )
            self.assertTrue(all(len(r["observation_24d"]) == 24 for r in result.records))
            self.assertTrue(
                any(
                    (r.get("edge_queue_at_decision") or 0)
                    + (r.get("cloud_queue_at_decision") or 0) > 0
                    for r in result.records[1:]
                ),
                "new tasks should observe backlog while real workers are busy",
            )
            collab = next(r for r in result.records if r["final_route"] == "COLLABORATIVE")
            self.assertEqual(collab["post_edge_decision"], "CLOUD_REVIEW")
            self.assertIsNotNone(collab["transmission_start_t"])
            self.assertIsNotNone(collab["transmission_finish_t"])
            self.assertIsNotNone(collab["cloud_review_running_t"])
            self.assertGreater(collab["real_edge_stage_s"], 0.0)
            self.assertGreater(collab["real_cloud_stage_s"], 0.0)
        finally:
            manager.close()


    def test_metrics_agent_accepts_hhk_heartbeat(self):
        server = build_server("127.0.0.1", 0, quiet=True)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = {
                "device_id": "edge-smoke",
                "device_type": "pc",
                "timestamp_ms": int(time.time() * 1000),
                "cpu": {"usage_pct": 42.0, "core_count": 8},
                "memory": {"avail_mb": 6000, "total_mb": 16000},
                "gpus": [{"name": "gpu", "total_mb": 8000, "free_mb": 5000}],
                "network": {
                    "rtt_ms": 18,
                    "packet_loss_pct": 2,
                    "cloud_reachable": True,
                },
                "inference": {"current_tps": 31.5, "task_queue_len": 1},
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/v1/edge/heartbeat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
            source = HHKHeartbeatSource(
                f"http://127.0.0.1:{port}/api/v1/edge/state?device_id=edge-smoke"
            )
            metrics = source.sample()
            self.assertEqual(metrics.cpu_usage_pct, 42.0)
            self.assertEqual(metrics.current_tps, 31.5)
            self.assertIsNotNone(source.last_network)
            self.assertEqual(source.last_network.rtt_ms, 18.0)
            self.assertAlmostEqual(source.last_network.success_ratio, 0.98, places=3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)

    def test_wall_clock_deadline_timeout(self):
        edge = FakeModelClient("slow-edge", 0.08)
        cloud = FakeModelClient("cloud", 0.01)
        network = FakeNetworkMonitor()
        provider = RealStateProvider(
            edge_source=StaticResourceSource(20.0, 0.8),
            cloud_source=StaticResourceSource(20.0, 0.8),
            network_monitor=network,
            sample_ttl_s=0.0,
        )
        manager = RealTaskManager(
            edge_client=edge,
            cloud_client=cloud,
            network_monitor=network,
            timeout_scan_s=0.003,
        )
        runtime = RealPPORuntime(
            policy=CyclicTestPolicy((0,)),
            policy_bundle=self.bundle,
            state_provider=provider,
            task_manager=manager,
        )
        try:
            result = runtime.run_tasks(
                [self.make_task(9, deadline_s=0.025)],
                arrival_interval_s=0.01,
                use_source_timestamps=False,
                drain_timeout_s=1.0,
                verbose=False,
            )
            self.assertEqual(result.timeout_count, 1)
            self.assertEqual(result.records[0]["status"], "timeout")
        finally:
            manager.close()

    def test_direct_cloud_is_safely_overridden_when_cloud_is_unreachable(self):
        network = FakeNetworkMonitor(reachable=False)
        provider = RealStateProvider(
            edge_source=StaticResourceSource(20.0, 0.8),
            cloud_source=StaticResourceSource(20.0, 0.8),
            network_monitor=network,
            sample_ttl_s=0.0,
        )
        manager = RealTaskManager(
            edge_client=FakeModelClient("edge"),
            cloud_client=FakeModelClient("cloud"),
            network_monitor=network,
        )
        runtime = RealPPORuntime(
            policy=CyclicTestPolicy((1,)),
            policy_bundle=self.bundle,
            state_provider=provider,
            task_manager=manager,
        )
        env = manager.register_arrival(self.make_task(10, deadline_s=1.0))
        _, _, requested, effective, info = runtime._decision_for(env)
        self.assertEqual(requested, 1)
        self.assertEqual(effective, 0)
        self.assertEqual(
            info["runtime_override"],
            "DIRECT_CLOUD_TO_LOCAL_FIRST_CLOUD_UNREACHABLE",
        )
        manager.close()

    def test_integration_and_scheduler_post_edge_evaluators_are_consistent(self):
        module_name = "scheduler_post_edge_policy_contract_test"
        source = ROOT.parent / "scheduler_training" / "post_edge_policy.py"
        spec = importlib.util.spec_from_file_location(module_name, source)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            scheduler_policy = module.PostEdgePolicy(self.bundle.post_edge_policy.rules)
            task = SimpleNamespace(
                task_type="fault_diagnosis",
                absolute_deadline_s=100.0,
            )
            cases = (
                {"confidence": 0.95, "level": "warning", "category": "known", "output_valid": True, "ood": False},
                {"confidence": 0.40, "level": "warning", "category": "known", "output_valid": True, "ood": False},
                {"confidence": 0.95, "level": "critical", "category": "known", "output_valid": True, "ood": False},
                {"confidence": None, "level": "", "category": "", "output_valid": False, "ood": False},
            )
            for signals in cases:
                integration_result = self.bundle.post_edge_policy.evaluate(
                    task=task,
                    edge_result=signals,
                    now_s=1.0,
                    cloud_review_estimated_s=0.1,
                ).to_dict()
                scheduler_result = scheduler_policy.evaluate(
                    task=task,
                    edge_result=signals,
                    now_s=1.0,
                    cloud_review_estimated_s=0.1,
                ).to_dict()
                self.assertEqual(integration_result, scheduler_result)
        finally:
            sys.modules.pop(module_name, None)

    def test_policy_bundle_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = create_test_bundle(temp)
            model = Path(temp) / "pre_route_ppo.zip"
            model.write_bytes(model.read_bytes() + b"tampered")
            with self.assertRaises(PolicyBundleError):
                load_policy_bundle(manifest)

    def test_cloud_review_transfer_failure_falls_back_to_edge_result(self):
        network = FailingTransferNetwork()
        provider = RealStateProvider(
            edge_source=StaticResourceSource(20.0, 0.8),
            cloud_source=StaticResourceSource(20.0, 0.8),
            network_monitor=network,
            sample_ttl_s=0.0,
        )
        manager = RealTaskManager(
            edge_client=FakeModelClient("edge", confidences=(0.3,)),
            cloud_client=FakeModelClient("cloud"),
            network_monitor=network,
            timeout_scan_s=0.005,
        )
        runtime = RealPPORuntime(
            policy=CyclicTestPolicy((0,)),
            policy_bundle=self.bundle,
            state_provider=provider,
            task_manager=manager,
        )
        try:
            result = runtime.run_tasks(
                [self.make_task(20, deadline_s=1.0)],
                use_source_timestamps=False,
                drain_timeout_s=2.0,
                verbose=False,
            )
            row = result.records[0]
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["post_edge_policy_decision"], "CLOUD_REVIEW")
            self.assertEqual(row["post_edge_decision"], "ACCEPT_EDGE")
            self.assertEqual(row["final_route"], "EDGE")
            self.assertIn("CLOUD_REVIEW_FAILURE_TO_ACCEPT_EDGE", row["runtime_override"])
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
