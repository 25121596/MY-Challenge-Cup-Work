from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dynamics import LegacyNetworkDynamics, LegacyResourceDynamics
from edf_queue import EDFTaskQueue, QueueItem
from edge_input_adapter import load_edge_inputs
from models import Action, PPO_ACTION_TO_ROUTE, ROUTE_TO_PPO_ACTION, Task
from observation import OBSERVATION_NAMES, build_observation
from policy_bundle import publish_policy_bundle, validate_policy_bundle
from post_edge_policy import DEFAULT_RULES_PATH, PostEdgeDecision, PostEdgePolicy
from ppo_core import CloudEdgePPOCore
from reward import RewardConfig
from simulator import CloudEdgeSimulator
from task_profiles import TASK_TYPE_ORDER


class SchedulerCoreTests(unittest.TestCase):
    def test_observation_and_action_contract(self):
        sim = CloudEdgeSimulator(seed=1, arrival_probability=1.0)
        context, _, _ = sim.advance_to_next_decision()
        self.assertIsNotNone(context)
        obs = build_observation(context["task"], context["state_before"])
        self.assertEqual(len(OBSERVATION_NAMES), 24)
        self.assertEqual(obs.shape, (24,))
        self.assertNotIn("task_confidence", OBSERVATION_NAMES)
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(Action.LOCAL_FIRST.value, 1)
        self.assertEqual(Action.DIRECT_CLOUD.value, 2)
        self.assertEqual(PPO_ACTION_TO_ROUTE, {0: Action.LOCAL_FIRST, 1: Action.DIRECT_CLOUD})
        self.assertEqual(ROUTE_TO_PPO_ACTION[Action.LOCAL_FIRST], 0)
        self.assertEqual(ROUTE_TO_PPO_ACTION[Action.DIRECT_CLOUD], 1)

    def test_sample_edge_inputs_and_rule_rollout(self):
        root = Path(__file__).resolve().parents[1]
        sample_path = root / "data" / "sample_edge_inputs.json"
        tasks, report = load_edge_inputs(sample_path)
        self.assertEqual(report.loaded_count, 96)
        self.assertEqual(len(TASK_TYPE_ORDER), 8)
        self.assertEqual({task.task_type for task in tasks}, set(TASK_TYPE_ORDER))

        simulator = CloudEdgeSimulator(
            seed=42,
            scheduler=None,
            queue_policy="edf",
        )
        simulator.run(1000)
        metrics = simulator.get_metrics()
        self.assertGreater(metrics["generated_tasks"], 0)
        self.assertGreaterEqual(metrics["timeout_rate"], 0.0)
        self.assertLessEqual(metrics["timeout_rate"], 1.0)
        self.assertLessEqual(
            set(simulator.get_task_type_metrics()["task_type"]),
            set(TASK_TYPE_ORDER),
        )

    def test_all_sample_edge_inputs_finish(self):
        root = Path(__file__).resolve().parents[1]
        simulator = CloudEdgeSimulator(
            seed=42,
            task_source="edge_input",
            edge_input_path=str(root / "data" / "sample_edge_inputs.json"),
            queue_policy="edf",
        )
        simulator.run_until_complete(max_steps=20_000)
        self.assertEqual(simulator.get_metrics()["finished_tasks"], 96)

    def test_non_preemptive_edf_waiting_order(self):
        queue = EDFTaskQueue("edf")
        a = Task(1, 0.0, 0.1, 5.0, 0.8)
        b = Task(2, 1.0, 0.1, 1.0, 0.8)
        c = Task(3, 0.5, 0.1, 2.5, 0.8)
        for task in (a, b, c):
            queue.push(QueueItem(task, Action.LOCAL_FIRST, "EDGE_WAITING", 1.0, 1.0))
        self.assertEqual(
            [queue.pop().task.task_id, queue.pop().task.task_id, queue.pop().task.task_id],
            [2, 3, 1],
        )

        sim = CloudEdgeSimulator(seed=3, arrival_probability=0.0)
        running = Task(10, 0.0, 0.1, 10.0, 0.9)
        urgent = Task(11, 0.1, 0.1, 0.2, 0.9)
        sim._pending_decision_task = running
        state = sim.get_system_state()
        details = sim.service_time_model.estimate(running, state)
        sim.apply_decision(
            {"task": running, "state_before": state},
            Action.LOCAL_FIRST,
            "test",
            details,
        )
        active_id = sim.edge_active.task.task_id
        sim.edge_queue.push(
            QueueItem(urgent, Action.LOCAL_FIRST, "EDGE_WAITING", 0.1, 0.1)
        )
        self.assertEqual(sim.edge_active.task.task_id, active_id)
        self.assertEqual(sim.edge_queue.items()[0].task.task_id, 11)

    def test_continuous_service_not_remaining_slots(self):
        sim = CloudEdgeSimulator(seed=4, arrival_probability=0.0)
        task = Task(
            task_id=1,
            arrival_time=0.0,
            size_mb=0.02,
            deadline_s=5.0,
            confidence=0.95,
            edge_base_latency_s=0.137,
            complexity=0.63,
        )
        sim._pending_decision_task = task
        state = sim.get_system_state()
        details = sim.service_time_model.estimate(task, state)
        sim.apply_decision(
            {"task": task, "state_before": state},
            Action.LOCAL_FIRST,
            "test",
            details,
        )
        finish = sim.edge_active.metadata["finish_time_s"]
        self.assertFalse(
            math.isclose(
                finish / sim.slot_duration,
                round(finish / sim.slot_duration),
                abs_tol=1e-8,
            )
        )
        sim._advance_clock(finish)
        sim._process_internal_events_now()
        self.assertEqual(len(sim.completed_tasks), 1)
        self.assertEqual(sim.completed_tasks[0]["route"], "EDGE")
        self.assertAlmostEqual(sim.completed_tasks[0]["latency_s"], finish, places=9)

    def test_post_edge_rule_triggers_collaborative_lifecycle(self):
        sim = CloudEdgeSimulator(seed=5, arrival_probability=0.0)
        task = Task(
            task_id=1,
            arrival_time=0.0,
            size_mb=0.1,
            deadline_s=10.0,
            confidence=0.50,
            task_type="fault_diagnosis",
        )
        sim._pending_decision_task = task
        state = sim.get_system_state()
        details = sim.service_time_model.estimate(task, state)
        sim.apply_decision(
            {"task": task, "state_before": state},
            Action.LOCAL_FIRST,
            "test",
            details,
        )
        self.assertEqual(task.execution_stage, "EDGE_RUNNING")
        for _ in range(10):
            if sim.completed_tasks:
                break
            next_time = sim._next_event_time(include_arrival=False)
            self.assertIsNotNone(next_time)
            sim._advance_clock(next_time)
            sim._process_internal_events_now()
        self.assertEqual(len(sim.completed_tasks), 1)
        result = sim.completed_tasks[0]
        self.assertEqual(result["pre_route_action"], "LOCAL_FIRST")
        self.assertEqual(result["post_edge_decision"], "CLOUD_REVIEW")
        self.assertEqual(result["post_edge_rule_version"], "post-edge-ai4i-v2")
        self.assertEqual(result["route"], "COLLABORATIVE")
        self.assertFalse(result["timeout"])
        self.assertEqual(result["execution_stage"], "FINISHED")
        self.assertIsNotNone(result["edge_stage_start_time"])
        self.assertIsNotNone(result["transmission_start_time"])
        self.assertIsNotNone(result["transmission_ready_time"])
        self.assertIsNotNone(result["cloud_stage_start_time"])
        self.assertAlmostEqual(result["component_sum_s"], result["latency_s"], places=8)

    def test_rule_config_and_simulator_use_same_policy(self):
        standalone = PostEdgePolicy.from_json(DEFAULT_RULES_PATH)
        sim = CloudEdgeSimulator(seed=6, arrival_probability=0.0)
        self.assertEqual(sim.post_edge_policy.rule_version, standalone.rule_version)
        task = Task(1, 0.0, 0.1, 10.0, 0.50, task_type="fault_diagnosis")
        edge_result = sim._simulated_edge_result(task)
        direct = standalone.evaluate(
            task=task,
            edge_result=edge_result,
            now_s=0.1,
            cloud_review_estimated_s=0.2,
        )
        through_simulator = sim.post_edge_policy.evaluate(
            task=task,
            edge_result=edge_result,
            now_s=0.1,
            cloud_review_estimated_s=0.2,
        )
        self.assertEqual(direct, through_simulator)
        self.assertEqual(direct.decision, PostEdgeDecision.CLOUD_REVIEW)

    def test_policy_bundle_manifest_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dummy_model = root / "dummy.zip"
            dummy_model.write_bytes(b"not-a-real-sb3-model")
            manifest = publish_policy_bundle(
                model_path=dummy_model,
                release_dir=root / "release",
                rules_path=DEFAULT_RULES_PATH,
                policy_version="test-policy-v2",
                training_metadata={"timesteps": 1},
            )
            report = validate_policy_bundle(manifest, load_ppo=False)
            self.assertTrue(report["valid"])
            self.assertEqual(report["observation_dimension"], 24)
            self.assertEqual(
                report["action_mapping"],
                {"0": "LOCAL_FIRST", "1": "DIRECT_CLOUD"},
            )
            self.assertEqual(
                report["calibration_status"],
                "legacy_parameters_not_real_calibrated",
            )

    def test_legacy_numeric_parameters_unchanged(self):
        resource = LegacyResourceDynamics("consume_recover")
        self.assertEqual(resource.edge_recovery_per_s, 0.045)
        self.assertEqual(resource.cloud_recovery_per_s, 0.0375)
        self.assertEqual(resource.edge_consumption_per_s, 0.090)
        self.assertEqual(resource.cloud_consumption_per_s, 0.075)
        self.assertEqual(resource.edge_noise_std_per_s, 0.015)
        self.assertEqual(resource.cloud_noise_std_per_s, 0.010)
        self.assertEqual((resource.edge_resource, resource.cloud_resource), (0.75, 0.85))

        network = LegacyNetworkDynamics("markov")
        self.assertEqual((network.min_bandwidth_mbps, network.max_bandwidth_mbps), (5.0, 50.0))
        self.assertEqual((network.p_good_to_bad_per_s, network.p_bad_to_good_per_s), (0.10, 0.30))
        self.assertEqual((network.good_network_quality, network.bad_network_quality), (1.00, 0.20))
        self.assertEqual(network.network_quality, 0.70)

        config = RewardConfig()
        self.assertEqual(config.latency_weight, 2.0)
        self.assertEqual(config.queue_penalty_weight, 0.2)
        self.assertEqual(config.timeout_penalty, 20.0)
        self.assertEqual(config.accuracy_penalty_weight, 3.0)
        self.assertEqual(config.communication_penalty_weight, 0.25)
        self.assertEqual(config.edge_compute_penalty_weight, 0.05)
        self.assertEqual(config.cloud_compute_penalty_weight, 0.05)

    def test_core_two_action_rollout(self):
        core = CloudEdgePPOCore(seed=7, episode_steps=2, arrival_probability=1.0)
        obs, info = core.reset()
        self.assertEqual(obs.shape, (24,))
        self.assertEqual(info["time_engine"], "continuous_event")
        names = []
        for action in (0, 1):
            obs, reward, terminated, truncated, step_info = core.step(action)
            self.assertEqual(obs.shape, (24,))
            self.assertTrue(np.isfinite(reward))
            names.append(step_info["action_name"])
            if terminated or truncated:
                break
        self.assertEqual(names, ["LOCAL_FIRST", "DIRECT_CLOUD"])


if __name__ == "__main__":
    unittest.main()
