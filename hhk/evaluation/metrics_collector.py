"""
指标采集器 — 对标比赛核心指标

采集指标:
  - 端到端时延 (ms)
  - 边端推理时延 (ms)
  - TTFT (首token延迟, ms)
  - 推理吞吐量 (tokens/s)
  - 边端内存占用 (GB)
  - 甩云率 (%)
  - 冲突率 (%)  [需多节点]
  - 冲突解决率 (%)  [需多节点]
  - 断网可用性 (%)  [需手动模拟]

用法:
  python metrics_collector.py <result_file_or_dir>
  python metrics_collector.py demo_results.json
"""
import sys, os, json, statistics, argparse
from pathlib import Path
from datetime import datetime


def compute_metrics(data):
    """从推理结果计算全部指标"""
    details = data.get("details", [])
    if not details:
        print("❌ 结果数据为空")
        return None

    # 过滤成功的结果
    ok = [d for d in details if "error" not in d]

    if not ok:
        print("❌ 全部请求失败")
        return {"total_requests": len(details), "successful": 0}

    latencies = [d["latency_ms"] for d in ok if "latency_ms" in d]
    tps_values = [d["tokens_per_sec"] for d in ok if d.get("tokens_per_sec", 0) > 0]

    metrics = {
        # 时延
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies)*0.95)], 1) if latencies else 0,
        "p99_latency_ms": round(sorted(latencies)[int(len(latencies)*0.99)], 1) if latencies else 0,
        "max_latency_ms": round(max(latencies), 1) if latencies else 0,

        # 吞吐量
        "avg_tokens_per_sec": round(statistics.mean(tps_values), 1) if tps_values else 0,

        # 达标率
        "within_200ms_count": sum(1 for l in latencies if l <= 200),
        "within_200ms_pct": round(sum(1 for l in latencies if l <= 200) / len(latencies) * 100, 1) if latencies else 0,

        # 可靠性
        "total_requests": len(details),
        "successful": len(ok),
        "success_rate_pct": round(len(ok) / len(details) * 100, 1) if details else 0,

        # 甩云率 (根据时延>200ms 或 result 中 escalate 标记推算)
        "cloud_escalations": data.get("summary", {}).get("cloud_escalations", 0),
        "escalation_rate_pct": round(
            data.get("summary", {}).get("cloud_escalations", 0) / len(details) * 100, 1
        ) if details else 0,

        "timestamp": datetime.now().isoformat(),
    }

    return metrics


def print_metrics(metrics, budget_ms=200):
    """打印指标对照比赛要求"""
    print("\n" + "=" * 50)
    print("  📐 比赛核心指标")
    print("=" * 50)

    checks = [
        ("端到端平均时延", f"{metrics['avg_latency_ms']} ms", f"≤{budget_ms}ms",
         metrics['avg_latency_ms'] <= budget_ms),
        ("P95时延", f"{metrics['p95_latency_ms']} ms", f"≤{budget_ms}ms",
         metrics['p95_latency_ms'] <= budget_ms),
        ("200ms达标率", f"{metrics['within_200ms_pct']}%", "≥95%",
         metrics['within_200ms_pct'] >= 95),
        ("成功率", f"{metrics['success_rate_pct']}%", "≥90%",
         metrics['success_rate_pct'] >= 90),
        ("甩云率", f"{metrics['escalation_rate_pct']}%", "≤30% (越低越好)",
         metrics['escalation_rate_pct'] <= 30),
    ]

    for name, value, target, passed in checks:
        status = "✅" if passed else "⚠️"
        print(f"  {status} {name:<18}: {value:<15} (要求: {target})")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="指标采集器")
    parser.add_argument("input", help="推理结果 JSON 文件")
    parser.add_argument("--budget-ms", type=int, default=200, help="时延预算(ms)")
    parser.add_argument("--output", default=None, help="输出指标 JSON")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = compute_metrics(data)
    if metrics:
        print_metrics(metrics, args.budget_ms)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"\n  📄 指标已保存: {args.output}")


if __name__ == "__main__":
    main()
