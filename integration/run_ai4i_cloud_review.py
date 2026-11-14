from __future__ import annotations

import argparse
import time
from pathlib import Path

from ai4i_cloud_review import (
    AI4ICloudReviewClient,
    append_jsonl,
    evaluate_results,
    failed_result,
    latest_results,
    load_failure_cutoff,
    read_jsonl,
    select_edge_rows,
    sha256_text,
    sha256_file,
    validate_sources,
    write_json,
    write_jsonl,
    write_report,
)
from config import CLOUD_ENDPOINT, REQUEST_TIMEOUT_S, ROOT


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "ai4i_edge_cloud_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在真实 llama-server 云模型上复核 AI4I 边端结果并生成边云指标"
    )
    parser.add_argument("--edge-predictions", required=True, help="HHK V2 test_predictions.jsonl")
    parser.add_argument("--edge-metrics", required=True, help="HHK V2 metrics.json")
    parser.add_argument("--qwen-inputs", required=True, help="AI4I Qwen review JSONL")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cloud-endpoint", default=CLOUD_ENDPOINT)
    parser.add_argument("--timeout-s", type=float, default=float(REQUEST_TIMEOUT_S))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--all-test", action="store_true", help="云端复核全部测试集，而非仅 Post-edge 子集")
    parser.add_argument("--pilot-per-label", type=int, help="每个真实标签抽取 N 条，执行分层 pilot")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--report-only", action="store_true", help="不调用模型，只从已有结果重建报告")
    parser.add_argument("--edge-model-version", default="ai4i_tabular_moe_v1")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def build_deliverables(
    *,
    edge_rows,
    target_rows,
    results_path: Path,
    output_dir: Path,
    cutoff: float,
    edge_model_version: str,
    batch_scope: str,
    source_hashes: dict[str, str],
) -> dict:
    results = latest_results(results_path)
    metrics, feedback = evaluate_results(
        edge_rows,
        target_rows,
        results,
        cutoff=cutoff,
        edge_model_version=edge_model_version,
        batch_scope=batch_scope,
        source_hashes=source_hashes,
    )
    write_jsonl(output_dir / "edge_cloud_feedback.jsonl", feedback)
    write_json(output_dir / "edge_cloud_metrics.json", metrics)
    write_report(output_dir / "edge_cloud_report.md", metrics)
    return metrics


def main() -> int:
    args = parse_args()
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts 必须大于 0")
    if args.progress_every <= 0:
        raise ValueError("--progress-every 必须大于 0")

    edge_path = Path(args.edge_predictions).resolve()
    metrics_path = Path(args.edge_metrics).resolve()
    qwen_path = Path(args.qwen_inputs).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "cloud_review_results.jsonl"

    edge_rows = read_jsonl(edge_path)
    prompt_rows = read_jsonl(qwen_path)
    cutoff = load_failure_cutoff(metrics_path)
    prompts = validate_sources(edge_rows, prompt_rows, cutoff)
    target_rows = select_edge_rows(
        edge_rows,
        review_only=not args.all_test,
        pilot_per_label=args.pilot_per_label,
        max_tasks=args.max_tasks,
    )
    batch_scope = "all_test" if args.all_test else "post_edge_review"
    if args.pilot_per_label is not None:
        batch_scope += "_stratified_pilot"
    if args.max_tasks is not None:
        batch_scope += "_limited"
    source_hashes = {
        "edge_predictions_sha256": sha256_file(edge_path),
        "edge_metrics_sha256": sha256_file(metrics_path),
        "qwen_inputs_sha256": sha256_file(qwen_path),
    }

    print(f"测试集: {len(edge_rows)} 条")
    print(f"Post-edge 复核任务: {sum(row.get('cloud_review') is True for row in edge_rows)} 条")
    print(f"本批目标: {len(target_rows)} 条 ({batch_scope})")
    print(f"边端阈值: {cutoff:.12f}")
    print(f"输出目录: {output_dir}")

    interrupted = False
    if not args.report_only:
        client = AI4ICloudReviewClient(
            args.cloud_endpoint,
            timeout_s=args.timeout_s,
            seed=args.seed,
        )
        service = client.inspect_service()
        print(f"云端: {service.endpoint} | model={service.model_id} | health={service.health}")
        existing = latest_results(results_path)
        completed = 0
        try:
            for index, edge_row in enumerate(target_rows, start=1):
                task_id = str(edge_row["task_id"])
                prompt = str(prompts[task_id]["prompt"])
                previous = existing.get(task_id)
                same_request = (
                    previous is not None
                    and previous.get("request", {}).get("prompt_sha256")
                    == sha256_text(prompt)
                    and previous.get("request", {}).get("model_id") == service.model_id
                )
                if same_request and (previous.get("output_valid") or not args.retry_failed):
                    completed += 1
                    continue

                last_error: BaseException | None = None
                result = None
                for attempt in range(1, args.max_attempts + 1):
                    try:
                        result = client.review(task_id, prompt)
                        result["attempt_count"] = attempt
                        if result["output_valid"] or attempt == args.max_attempts:
                            break
                    except Exception as exc:  # preserve one auditable terminal error per task
                        last_error = exc
                        if attempt < args.max_attempts:
                            time.sleep(min(0.5 * attempt, 2.0))
                if result is None:
                    assert last_error is not None
                    result = failed_result(
                        task_id,
                        prompt,
                        endpoint=service.endpoint,
                        model_id=service.model_id,
                        seed=args.seed,
                        exc=last_error,
                        attempt_count=args.max_attempts,
                    )
                append_jsonl(results_path, result)
                existing[task_id] = result
                completed += 1
                if completed % args.progress_every == 0 or completed == len(target_rows):
                    status = "OK" if result.get("output_valid") else "FAILED"
                    print(f"[{completed}/{len(target_rows)}] {task_id}: {status}")
        except KeyboardInterrupt:
            interrupted = True
            print("\n收到中断，保留已完成 JSONL，并生成当前阶段报告。")

    metrics = build_deliverables(
        edge_rows=edge_rows,
        target_rows=target_rows,
        results_path=results_path,
        output_dir=output_dir,
        cutoff=cutoff,
        edge_model_version=args.edge_model_version,
        batch_scope=batch_scope,
        source_hashes=source_hashes,
    )
    execution = metrics["cloud_execution"]
    comparison = metrics["edge_cloud_comparison"]
    print("\n交付物已生成:")
    for name in (
        "cloud_review_results.jsonl",
        "edge_cloud_feedback.jsonl",
        "edge_cloud_metrics.json",
        "edge_cloud_report.md",
    ):
        print(f"  {output_dir / name}")
    print(
        f"结果: attempted={execution['attempted_tasks']}, valid={execution['valid_outputs']}, "
        f"agreement={comparison['agreement_count']}, disagreement={comparison['disagreement_count']}"
    )
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
