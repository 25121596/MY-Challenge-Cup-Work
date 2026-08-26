from __future__ import annotations

from collections import Counter
import unittest

import numpy as np

from experiment_scenarios import (
    AlwaysAcceptEdgePolicy,
    DEFAULT_SCENARIO_FILES,
    ScenarioArrivalModel,
    load_experiment_scenarios,
    scenario_simulator_kwargs,
    select_training_scenario,
)
from ppo_core import CloudEdgePPOCore
from scheduler import AlwaysCloudScheduler, AlwaysLocalFirstScheduler
from simulator import CloudEdgeSimulator


class ExperimentScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = load_experiment_scenarios(DEFAULT_SCENARIO_FILES)
        cls.by_id = {scenario.scenario_id: scenario for scenario in cls.scenarios}

    def test_standard_scenario_contract(self):
        self.assertEqual(
            tuple(self.by_id),
            ("normal_load", "edge_burst", "weak_network"),
        )
        self.assertEqual(
            [scenario.training_weight for scenario in self.scenarios],
            [0.4, 0.4, 0.2],
        )
        self.assertTrue(
            all(scenario.task_source == "typed_simulation" for scenario in self.scenarios)
        )
        self.assertTrue(
            all(scenario.slot_duration_s == 0.05 for scenario in self.scenarios)
        )

        burst = self.by_id["edge_burst"]
        self.assertEqual(burst.arrival_period_s, 15.0)
        self.assertEqual(
            [segment.probability for segment in burst.arrival_segments],
            [0.08, 0.40, 0.08],
        )
        weak = self.by_id["weak_network"]
        self.assertEqual(weak.network_initial_state, "BAD")
        self.assertEqual(
            (weak.min_bandwidth_mbps, weak.max_bandwidth_mbps),
            (5.0, 20.0),
        )

    def test_scenario_arrivals_are_seed_reproducible(self):
        scenario = self.by_id["edge_burst"]

        def sample(seed: int) -> list[float]:
            model = ScenarioArrivalModel(scenario, seed=seed)
            rng = np.random.default_rng(seed)
            model.reset(rng=rng)
            values: list[float] = []
            for _ in range(40):
                next_time = model.next_arrival_time_s()
                self.assertIsNotNone(next_time)
                values.append(float(next_time))
                model.consume_arrival(rng=rng, now_s=float(next_time))
            return values

        self.assertEqual(sample(81), sample(81))
        self.assertNotEqual(sample(81), sample(82))

    def test_weighted_episode_selection_is_reproducible(self):
        first = [
            select_training_scenario(self.scenarios, seed=seed).scenario_id
            for seed in range(200)
        ]
        second = [
            select_training_scenario(self.scenarios, seed=seed).scenario_id
            for seed in range(200)
        ]
        self.assertEqual(first, second)
        counts = Counter(first)
        self.assertEqual(set(counts), set(self.by_id))
        self.assertGreater(counts["normal_load"], counts["weak_network"])
        self.assertGreater(counts["edge_burst"], counts["weak_network"])

    def test_edge_only_is_a_true_no_cloud_baseline(self):
        scenario = self.by_id["normal_load"]
        simulator = CloudEdgeSimulator(
            seed=42,
            scheduler=AlwaysLocalFirstScheduler(),
            post_edge_policy=AlwaysAcceptEdgePolicy(),
            queue_policy="edf",
            **scenario_simulator_kwargs(scenario, seed=42),
        )
        simulator.run(200)
        metrics = simulator.get_metrics()
        self.assertEqual(metrics["local_first_ratio"], 1.0)
        self.assertEqual(metrics["direct_cloud_ratio"], 0.0)
        self.assertEqual(metrics["cloud_review_ratio"], 0.0)
        self.assertEqual(metrics["cloud_ratio"], 0.0)
        self.assertEqual(metrics["collaborative_ratio"], 0.0)
        self.assertEqual(metrics["upload_mb"], 0.0)

    def test_post_edge_only_never_directly_routes_to_cloud(self):
        scenario = self.by_id["normal_load"]
        simulator = CloudEdgeSimulator(
            seed=43,
            scheduler=AlwaysLocalFirstScheduler(),
            queue_policy="edf",
            **scenario_simulator_kwargs(scenario, seed=43),
        )
        simulator.run(200)
        metrics = simulator.get_metrics()
        self.assertEqual(metrics["local_first_ratio"], 1.0)
        self.assertEqual(metrics["direct_cloud_ratio"], 0.0)
        self.assertEqual(metrics["cloud_ratio"], 0.0)
        self.assertGreater(
            metrics["post_edge_review_tasks"] + metrics["post_edge_accept_tasks"],
            0,
        )

    def test_same_seed_pairs_the_workload_across_strategies(self):
        scenario = self.by_id["edge_burst"]

        def workload(scheduler: object) -> list[tuple]:
            simulator = CloudEdgeSimulator(
                seed=91,
                scheduler=scheduler,
                post_edge_policy=AlwaysAcceptEdgePolicy(),
                queue_policy="edf",
                **scenario_simulator_kwargs(scenario, seed=91),
            )
            simulator.run(120)
            records = simulator.get_records()
            return list(
                records[
                    [
                        "time_s",
                        "task_type",
                        "task_size_mb",
                        "deadline_s",
                        "latent_edge_confidence",
                    ]
                ].itertuples(index=False, name=None)
            )

        self.assertEqual(
            workload(AlwaysLocalFirstScheduler()),
            workload(AlwaysCloudScheduler()),
        )

    def test_cross_scenario_core_preserves_24d_two_action_contract(self):
        core = CloudEdgePPOCore(
            seed=44,
            episode_steps=2,
            experiment_scenarios=self.scenarios,
        )
        observation, reset_info = core.reset()
        self.assertEqual(observation.shape, (24,))
        self.assertIn(reset_info["scenario_id"], self.by_id)
        _, _, _, _, first = core.step(0)
        self.assertEqual(first["action_name"], "LOCAL_FIRST")
        _, _, _, truncated, second = core.step(1)
        self.assertEqual(second["action_name"], "DIRECT_CLOUD")
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
