"""
评测报告生成器 — 根据采集的指标生成比赛提交用报告

用法:
  python report_generator.py <metrics.json> [--template report_template.md]
"""
import sys, os, json, argparse
from datetime import datetime


def generate_report(metrics_json, benchmark_json=None, memcheck_json=None):
    """生成 Markdown 格式评测报告"""

    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 可选: 合并 benchmark 和 memcheck 结果
    bench_data = None
    if benchmark_json and os.path.exists(benchmark_json):
        with open(benchmark_json, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

    memcheck_data = None
    if memcheck_json and os.path.exists(memcheck_json):
        with open(memcheck_json, "r", encoding="utf-8") as f:
            memcheck_data = json.load(f)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# 云边协同 AI 系统 — 性能评测报告

> 赛题: XH-202606 面向云边协同场景的分布式人工智能感知与决策
> 生成时间: {now}
> 测试模型: Qwen3-1.7B-IQ4_XS

---

## 一、执行摘要

本报告记录了边端轻量大模型在工业场景下的性能评测结果。

| 核心指标 | 实测值 | 赛题要求 | 达标 |
|---|---|---|---|
| 端到端平均时延 | {metrics.get('avg_latency_ms', 'N/A')} ms | ≤200ms | {'✅' if metrics.get('avg_latency_ms', 999) <= 200 else '❌'} |
| P95时延 | {metrics.get('p95_latency_ms', 'N/A')} ms | ≤200ms | {'✅' if metrics.get('p95_latency_ms', 999) <= 200 else '⚠️'} |
| 200ms达标率 | {metrics.get('within_200ms_pct', 'N/A')}% | ≥95% | {'✅' if metrics.get('within_200ms_pct', 0) >= 95 else '⚠️'} |
| 请求成功率 | {metrics.get('success_rate_pct', 'N/A')}% | ≥90% | {'✅' if metrics.get('success_rate_pct', 0) >= 90 else '❌'} |
| 单次推理内存 | {memcheck_data[0].get('ram_delta_peak', 'N/A') if memcheck_data else 'N/A'} GB | ≤1.5GB | {'✅' if (memcheck_data and memcheck_data[0].get('ram_delta_peak', 999) <= 1.5) else '⚠️'} |
| 能力保持率 | {bench_data[0].get('total_score', 'N/A') if bench_data else 'N/A'}/100 | ≥80% | {'✅' if (bench_data and bench_data[0].get('total_score', 0) >= 80) else '⚠️'} |

## 二、时延分布

| 指标 | 值 |
|---|---|
| 平均时延 | {metrics.get('avg_latency_ms', 'N/A')} ms |
| P50 (中位数) | {metrics.get('p50_latency_ms', 'N/A')} ms |
| P95 | {metrics.get('p95_latency_ms', 'N/A')} ms |
| P99 | {metrics.get('p99_latency_ms', 'N/A')} ms |
| 最大时延 | {metrics.get('max_latency_ms', 'N/A')} ms |

## 三、吞吐量

| 指标 | 值 |
|---|---|
| 平均生成速度 | {metrics.get('avg_tokens_per_sec', 'N/A')} t/s |

## 四、云边协同效率

| 指标 | 值 |
|---|---|
| 甩云率 | {metrics.get('escalation_rate_pct', 'N/A')}% |
| 甩云次数 | {metrics.get('cloud_escalations', 'N/A')} |

## 五、综合能力评测 (benchmark.py)

| 能力维度 | 得分 |
|---|---|
| 💻 代码 | {bench_data[0].get('section_details', {}).get('🧠 通用能力', {}).get('details', [{}])[0].get('avg_score', 'N/A') if bench_data else 'N/A'} |
| 🔢 数学 | {bench_data[0].get('section_details', {}).get('🧠 通用能力', {}).get('details', [{}])[1].get('avg_score', 'N/A') if bench_data else 'N/A'} |
| 📖 语言推理 | {bench_data[0].get('section_details', {}).get('🧠 通用能力', {}).get('details', [{}])[2].get('avg_score', 'N/A') if bench_data else 'N/A'} |
| 📊 信息处理 | {bench_data[0].get('section_details', {}).get('🧠 通用能力', {}).get('details', [{}])[3].get('avg_score', 'N/A') if bench_data else 'N/A'} |
| 🔧 故障诊断 | {bench_data[0].get('section_details', {}).get('🏭 工业场景', {}).get('details', [{}])[0].get('avg_score', 'N/A') if bench_data else 'N/A'} |
| 🔒 安全识别 | {bench_data[0].get('section_details', {}).get('🏭 工业场景', {}).get('details', [{}])[1].get('avg_score', 'N/A') if bench_data else 'N/A'} |
| **综合** | **{bench_data[0].get('total_score', 'N/A') if bench_data else 'N/A'}/100** |

## 六、内存占用

| 指标 | 值 |
|---|---|
| RAM 峰值增量 | {memcheck_data[0].get('ram_delta_peak', 'N/A') if memcheck_data else 'N/A'} GB |
| VRAM 峰值增量 | {memcheck_data[0].get('vram_delta_peak', 'N/A') if memcheck_data else 'N/A'} MB |
| 边缘预估总占用 | {memcheck_data[0].get('edge_ram_estimate', 'N/A') if memcheck_data else 'N/A'} GB |

## 七、结论

基于以上评测结果，边端轻量模型 Qwen3-1.7B-IQ4_XS：
- {'✅ 满足' if metrics.get('avg_latency_ms', 999) <= 200 else '⚠️ 未完全满足'} 端到端时延要求
- {'✅ 满足' if (memcheck_data and memcheck_data[0].get('ram_delta_peak', 999) <= 1.5) else '⚠️ 需进一步优化'} 内存限制
- 建议: 后续使用 4B 或 8B 模型提升能力保持率
"""

    return report


def main():
    parser = argparse.ArgumentParser(description="评测报告生成器")
    parser.add_argument("metrics", help="metrics_collector 输出的 JSON")
    parser.add_argument("--benchmark", default=None, help="benchmark 输出的 JSON")
    parser.add_argument("--memcheck", default=None, help="memcheck 输出的 JSON")
    parser.add_argument("--output", default="EVALUATION_REPORT.md", help="输出报告路径")

    args = parser.parse_args()

    report = generate_report(args.metrics, args.benchmark, args.memcheck)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 报告已生成: {args.output}")
    print(f"\n{report[:500]}...")


if __name__ == "__main__":
    main()
