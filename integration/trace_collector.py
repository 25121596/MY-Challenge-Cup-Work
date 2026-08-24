from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from profiling import TRACE_SCHEMA_VERSION, build_profile_summary


# Stable, ordered CSV schema. Nested/raw diagnostic objects are serialized as JSON.
TRACE_CSV_FIELDS = [
    "schema_version", "bundle_schema_version", "policy_version", "post_edge_rule_version",
    "task_id", "task_type", "priority", "deadline_s",
    "input_size_bytes", "size_mb", "input_tokens", "complexity", "task_profile_confidence",
    "arrival_wall_time_s", "arrival_t", "decision_t", "first_start_t", "finish_t",
    "requested_action_index", "requested_action_name", "action_index", "action_name",
    "final_route", "runtime_override", "policy_action_probabilities", "policy_value_estimate",
    "post_edge_decision_t", "post_edge_policy_decision", "post_edge_decision",
    "post_edge_reason", "post_edge_rule_id", "post_edge_review_recommended",
    "post_edge_cloud_review_feasible", "post_edge_confidence_threshold",
    "post_edge_failure_probability", "post_edge_review_state",
    "edge_confidence", "edge_failure_probability", "edge_local_prediction", "edge_review_state",
    "edge_safety_priority", "edge_safety_trigger", "edge_action", "edge_model_route",
    "edge_level", "edge_category", "edge_output_valid", "edge_ood",
    "edge_escalation_hint",
    "status", "execution_stage", "finish_reason", "error",
    "edge_queue_at_decision", "cloud_queue_at_decision", "edge_workload_s_at_decision",
    "cloud_workload_s_at_decision", "edge_resource_at_decision", "cloud_resource_at_decision",
    "network_quality_at_decision", "bandwidth_mbps_at_decision", "rtt_ms_at_decision",
    "jitter_ms_at_decision", "network_failure_rate_at_decision", "packet_loss_pct_at_decision",
    "edge_cpu_pct_at_decision", "edge_memory_available_mb_at_decision",
    "edge_memory_total_mb_at_decision", "edge_gpu_usage_pct_at_decision",
    "edge_gpu_free_mb_at_decision", "edge_gpu_total_mb_at_decision", "edge_tps_at_decision",
    "cloud_cpu_pct_at_decision", "cloud_memory_available_mb_at_decision",
    "cloud_memory_total_mb_at_decision", "cloud_gpu_usage_pct_at_decision",
    "cloud_gpu_free_mb_at_decision", "cloud_gpu_total_mb_at_decision", "cloud_tps_at_decision",
    "edge_queue_wait_s", "cloud_queue_wait_s", "cloud_review_queue_wait_s",
    "edge_running_t", "transmission_start_t", "transmission_finish_t", "cloud_running_t",
    "cloud_review_running_t", "real_edge_stage_s", "real_transfer_stage_s",
    "real_cloud_stage_s", "real_total_wall_s", "deadline_overrun_s",
    "real_transfer_bytes", "real_transfer_mbps",
    "edge_model_id", "edge_endpoint", "edge_prompt_bytes", "edge_prompt_tokens", "edge_completion_tokens", "edge_total_tokens", "edge_cached_tokens", "edge_reasoning_tokens",
    "edge_prompt_ms", "edge_completion_ms", "edge_prompt_tps", "edge_completion_tps", "edge_ttft_ms",
    "cloud_model_id", "cloud_endpoint", "cloud_prompt_bytes", "cloud_prompt_tokens", "cloud_completion_tokens", "cloud_total_tokens", "cloud_cached_tokens", "cloud_reasoning_tokens",
    "cloud_prompt_ms", "cloud_completion_ms", "cloud_prompt_tps", "cloud_completion_tps", "cloud_ttft_ms",
    "observation_24d", "resource_snapshots", "network_rtt_window_ms", "final_answer",
]


class TraceCollector:
    def __init__(self, *, schema_version: str = TRACE_SCHEMA_VERSION) -> None:
        self.schema_version = schema_version

    def normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        row = dict(record)
        row["schema_version"] = self.schema_version
        return row

    def write_jsonl(self, path: str | Path, records: list[dict[str, Any]]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(self.normalize_record(record), ensure_ascii=False) + "\n")

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return value

    def write_csv(self, path: str | Path, records: list[dict[str, Any]]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRACE_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                row = self.normalize_record(record)
                writer.writerow({key: self._csv_value(row.get(key)) for key in TRACE_CSV_FIELDS})

    def write_summary(self, path: str | Path, records: list[dict[str, Any]], **metadata: Any) -> dict[str, Any]:
        payload = build_profile_summary(records)
        payload.update(metadata)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
