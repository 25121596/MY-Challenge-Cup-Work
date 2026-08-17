"""Phase4.5仿真记录与评价指标。"""

from __future__ import annotations

import pandas as pd


class MetricsTracker:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.completed_tasks: list[dict] = []

    def add_record(self, record: dict) -> None:
        self.records.append(record)

    def add_completed_task(self, task_result: dict) -> None:
        self.completed_tasks.append(task_result)

    def get_records_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def get_completed_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.completed_tasks)

    @staticmethod
    def _quantile(dataframe: pd.DataFrame, column: str, q: float) -> float:
        if dataframe.empty or column not in dataframe:
            return float("nan")
        values = dataframe[column].dropna()
        return float(values.quantile(q)) if not values.empty else float("nan")

    def summarize(self, *, generated_tasks: int, in_system_tasks: int) -> dict:
        records = self.get_records_df()
        finished = self.get_completed_df()
        finished_tasks = len(finished)
        timeout_tasks = int(finished["timeout"].sum()) if not finished.empty else 0
        successful = (
            finished[finished["timeout"] == False] if not finished.empty else finished
        )
        successful_tasks = len(successful)

        processed_rate = finished_tasks / max(generated_tasks, 1)
        completion_rate = successful_tasks / max(generated_tasks, 1)
        timeout_rate = timeout_tasks / max(generated_tasks, 1)
        finished_timeout_rate = timeout_tasks / max(finished_tasks, 1)
        average_success_latency = (
            float(successful["latency_s"].mean())
            if not successful.empty
            else float("nan")
        )
        average_accuracy = (
            float(finished["expected_accuracy"].mean())
            if not finished.empty
            else float("nan")
        )

        if records.empty:
            average_reward = cumulative_reward = 0.0
            average_edge_queue = average_cloud_queue = 0.0
            edge_ratio = cloud_ratio = collaborative_ratio = 0.0
            local_first_ratio = direct_cloud_ratio = 0.0
            average_deadline_ratio = upload_mb = 0.0
        else:
            action_records = records[records["task_id"].notna()]
            if action_records.empty:
                local_first_ratio = direct_cloud_ratio = 0.0
            else:
                local_first_ratio = float((action_records["action_name"] == "LOCAL_FIRST").mean())
                direct_cloud_ratio = float((action_records["action_name"] == "DIRECT_CLOUD").mean())
            if finished.empty:
                edge_ratio = cloud_ratio = collaborative_ratio = upload_mb = 0.0
            else:
                edge_ratio = float((finished["route"] == "EDGE").mean())
                cloud_ratio = float((finished["route"] == "CLOUD").mean())
                collaborative_ratio = float((finished["route"] == "COLLABORATIVE").mean())
                upload_mb = float(finished["upload_mb"].fillna(0.0).sum())
            average_reward = float(records["reward"].mean())
            cumulative_reward = float(records["cumulative_reward"].iloc[-1])
            average_edge_queue = float(records["edge_queue"].mean())
            average_cloud_queue = float(records["cloud_queue"].mean())
            average_deadline_ratio = float(records["deadline_ratio"].mean())

        latency_p50 = self._quantile(successful, "latency_s", 0.50)
        latency_p95 = self._quantile(successful, "latency_s", 0.95)
        latency_p99 = self._quantile(successful, "latency_s", 0.99)
        waiting_p50 = self._quantile(successful, "waiting_time_s", 0.50)
        waiting_p95 = self._quantile(successful, "waiting_time_s", 0.95)
        waiting_p99 = self._quantile(successful, "waiting_time_s", 0.99)
        tail_ratio = (
            latency_p99 / latency_p50
            if latency_p50 == latency_p50 and latency_p50 > 0
            else float("nan")
        )
        action_ratio_sum = edge_ratio + cloud_ratio + collaborative_ratio

        return {
            "generated_tasks": generated_tasks,
            "finished_tasks": finished_tasks,
            "successful_tasks": successful_tasks,
            "timeout_tasks": timeout_tasks,
            "processed_rate": processed_rate,
            "completion_rate": completion_rate,
            "timeout_rate": timeout_rate,
            "finished_timeout_rate": finished_timeout_rate,
            "average_success_latency": average_success_latency,
            "latency_p50_s": latency_p50,
            "latency_p95_s": latency_p95,
            "latency_p99_s": latency_p99,
            "waiting_p50_s": waiting_p50,
            "waiting_p95_s": waiting_p95,
            "waiting_p99_s": waiting_p99,
            "tail_ratio_p99_p50": tail_ratio,
            "average_expected_accuracy": average_accuracy,
            "average_deadline_ratio": average_deadline_ratio,
            "average_reward": average_reward,
            "cumulative_reward": cumulative_reward,
            "average_edge_queue": average_edge_queue,
            "average_cloud_queue": average_cloud_queue,
            "edge_ratio": edge_ratio,
            "cloud_ratio": cloud_ratio,
            "collaborative_ratio": collaborative_ratio,
            "local_first_ratio": local_first_ratio,
            "direct_cloud_ratio": direct_cloud_ratio,
            "action_ratio_sum": action_ratio_sum,
            "upload_mb": upload_mb,
            "in_system_tasks": in_system_tasks,
        }

    def summarize_by_task_type(self) -> pd.DataFrame:
        finished = self.get_completed_df()
        if finished.empty:
            return pd.DataFrame()

        rows: list[dict] = []
        for task_type, group in finished.groupby("task_type", dropna=False):
            successful = group[group["timeout"] == False]
            rows.append(
                {
                    "task_type": task_type,
                    "finished_tasks": len(group),
                    "successful_tasks": len(successful),
                    "timeout_rate": float(group["timeout"].mean()),
                    "average_latency_s": (
                        float(successful["latency_s"].mean())
                        if not successful.empty
                        else float("nan")
                    ),
                    "latency_p95_s": (
                        float(successful["latency_s"].quantile(0.95))
                        if not successful.empty
                        else float("nan")
                    ),
                    "average_accuracy": float(group["expected_accuracy"].mean()),
                    "edge_ratio": float((group["route"] == "EDGE").mean()),
                    "cloud_ratio": float((group["route"] == "CLOUD").mean()),
                    "collaborative_ratio": float(
                        (group["route"] == "COLLABORATIVE").mean()
                    ),
                    "average_upload_mb": float(group.get("upload_mb", 0.0).fillna(0.0).mean())
                    if "upload_mb" in group
                    else 0.0,
                    "average_payload_mb": float(group["size_mb"].mean()),
                    "average_deadline_s": float(group["deadline_s"].mean()),
                }
            )
        return pd.DataFrame(rows).sort_values("task_type").reset_index(drop=True)

    def reset(self) -> None:
        self.records.clear()
        self.completed_tasks.clear()
