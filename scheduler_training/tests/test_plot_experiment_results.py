import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_experiment_results import (
    CHART_STEMS,
    ResultTables,
    derive_task_type_cloud_matrix,
    derive_upload_comparison,
    generate_charts,
    validate_tables,
)


SCENARIOS = ("normal_load", "edge_burst", "weak_network")
STRATEGIES = ("EDGE_ONLY", "PRE_ROUTE_ONLY", "POST_EDGE_ONLY", "TWO_LAYER")
TASK_TYPES = (
    "fault_diagnosis",
    "quality_inspection",
    "safety_check",
    "data_analysis",
    "status_monitoring",
    "predictive_maintenance",
    "generic_anomaly",
    "video_anomaly",
)


def build_tables() -> ResultTables:
    summary_rows = []
    detailed_rows = []
    by_task_type_rows = []
    completion_by_strategy = {
        "EDGE_ONLY": 0.30,
        "PRE_ROUTE_ONLY": 0.72,
        "POST_EDGE_ONLY": 0.35,
        "TWO_LAYER": 0.82,
    }
    accuracy_by_strategy = {
        "EDGE_ONLY": 0.76,
        "PRE_ROUTE_ONLY": 0.82,
        "POST_EDGE_ONLY": 0.77,
        "TWO_LAYER": 0.84,
    }
    for scenario_index, scenario_id in enumerate(SCENARIOS):
        for strategy_id in STRATEGIES:
            is_two_layer = strategy_id == "TWO_LAYER"
            summary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "strategy_id": strategy_id,
                    "completion_rate": completion_by_strategy[strategy_id]
                    - scenario_index * 0.02,
                    "latency_p95_s": 0.58 if strategy_id == "EDGE_ONLY" else 0.39,
                    "edge_ratio": 0.64 if is_two_layer else 1.0,
                    "cloud_ratio": 0.28 if is_two_layer else 0.0,
                    "collaborative_ratio": 0.08 if is_two_layer else 0.0,
                    "upload_mb": 10.0 if is_two_layer else 0.0,
                    "average_expected_accuracy": accuracy_by_strategy[strategy_id]
                    - scenario_index * 0.01,
                }
            )
            for seed in (42, 43):
                detailed_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "strategy_id": strategy_id,
                        "seed": seed,
                        "completion_rate": completion_by_strategy[strategy_id]
                        - scenario_index * 0.02
                        + (seed - 42.5) * 0.01,
                        "average_expected_accuracy": accuracy_by_strategy[strategy_id]
                        - scenario_index * 0.01
                        + (seed - 42.5) * 0.002,
                    }
                )

        for strategy_id in ("EDGE_ONLY", "TWO_LAYER"):
            for seed in (42, 43):
                for task_type in TASK_TYPES:
                    by_task_type_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "strategy_id": strategy_id,
                            "seed": seed,
                            "task_type": task_type,
                            "finished_tasks": 10,
                            "cloud_ratio": 0.30 if strategy_id == "TWO_LAYER" else 0.0,
                            "collaborative_ratio": 0.10 if strategy_id == "TWO_LAYER" else 0.0,
                            "average_payload_mb": 0.50,
                        }
                    )

    monitor = pd.DataFrame({"r": [-160.0 + index * 2.0 for index in range(30)]})
    return ResultTables(
        summary=pd.DataFrame(summary_rows),
        detailed=pd.DataFrame(detailed_rows),
        by_task_type=pd.DataFrame(by_task_type_rows),
        monitor=monitor,
    )


class PlotExperimentResultsTests(unittest.TestCase):
    def test_validate_tables_requires_all_four_strategies(self):
        tables = build_tables()
        tables.summary.drop(
            tables.summary[tables.summary["strategy_id"] == "PRE_ROUTE_ONLY"].index,
            inplace=True,
        )
        with self.assertRaisesRegex(ValueError, "正式四策略"):
            validate_tables(tables)

    def test_upload_baseline_is_derived_from_completed_payload(self):
        comparison = derive_upload_comparison(build_tables()).set_index("scenario_id")
        self.assertAlmostEqual(
            float(comparison.loc["normal_load", "full_cloud_upload_mb"]), 40.0
        )
        self.assertAlmostEqual(
            float(comparison.loc["normal_load", "two_layer_upload_mb"]), 10.0
        )
        self.assertAlmostEqual(float(comparison.loc["normal_load", "reduction_rate"]), 0.75)

    def test_task_type_matrix_uses_cloud_and_collaborative_routes(self):
        matrix = derive_task_type_cloud_matrix(build_tables())
        self.assertAlmostEqual(
            float(matrix.loc["fault_diagnosis", "normal_load"]), 40.0
        )

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "Matplotlib 未安装，仅跳过图像渲染测试。",
    )
    def test_generate_seven_charts_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            try:
                files = generate_charts(
                    build_tables(),
                    output_dir,
                    formats=("png",),
                    dpi=100,
                    rolling_window=5,
                )
            except RuntimeError as exc:
                if "中文字体" in str(exc):
                    self.skipTest(str(exc))
                raise

            self.assertEqual(len(files), len(CHART_STEMS))
            self.assertEqual({path.stem for path in files}, set(CHART_STEMS))
            self.assertTrue(all(path.stat().st_size > 0 for path in files))
            manifest = json.loads(
                (output_dir / "chart_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], "scheduler-training-charts-v1")
            self.assertIn("场景仿真", manifest["source_semantics"])
            self.assertIn("不是真实Accuracy/F1", manifest["accuracy_semantics"])


if __name__ == "__main__":
    unittest.main()
