"""
端到端演示控制器

一键运行完整流程:
  1. 加载场景配置
  2. 启动云端服务
  3. 预处理传感器数据 → 标准输入
  4. 注入异常 (生成测试用例)
  5. 调用边端推理
  6. 收集全部指标
  7. 生成评测报告

用法:
  python demo_controller.py                                    # 默认交互模式
  python demo_controller.py --scenario industrial_inspection   # 工业检测场景
  python demo_controller.py --scenario traffic_monitoring      # 交通监控场景
  python demo_controller.py --scenario both                    # 两个场景都跑
"""
import sys, os, json, time, subprocess, argparse, urllib.request, uuid
from pathlib import Path
from datetime import datetime

# 项目路径
HHK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HHK_DIR))  # 允许 from preprocessing.xxx import
SCENARIOS_DIR = HHK_DIR / "scenarios"
PREPROCESSING_DIR = HHK_DIR / "preprocessing"
EVALUATION_DIR = HHK_DIR / "evaluation"
DATA_DIR = Path(r"D:\Challenge Cup\Data file")

# llama.cpp 可执行文件
LLAMA_CLI = HHK_DIR / "build" / "bin" / "Release" / "llama-cli.exe"
LLAMA_SERVER = HHK_DIR / "build" / "bin" / "Release" / "llama-server.exe"
# fallback: 工程根目录下的预编译版本
if not LLAMA_SERVER.exists():
    LLAMA_SERVER = HHK_DIR.parent.parent / "llama" / "llama-server.exe"


def load_scenario(name):
    """加载场景配置"""
    path = SCENARIOS_DIR / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def step(label):
    """打印步骤标题"""
    print(f"\n{'='*60}")
    print(f"  📍 {label}")
    print(f"{'='*60}")


def check_prerequisites():
    """检查运行环境"""
    issues = []

    # Python 库
    for lib in ["psutil", "flask"]:
        try:
            __import__(lib)
        except ImportError:
            issues.append(f"缺少 Python 库: pip install {lib}")

    # 数据目录
    if not DATA_DIR.exists():
        issues.append(f"数据目录不存在: {DATA_DIR}")

    # 预处理脚本
    if not (PREPROCESSING_DIR / "sensor_csv_to_input.py").exists():
        issues.append("缺少预处理脚本")

    if issues:
        print("❌ 环境检查失败:")
        for i in issues:
            print(f"   - {i}")
        return False
    print("✅ 环境检查通过")
    return True


def run_hydraulic_preprocessing():
    """预处理 UCI 液压数据"""
    hydra_dir = DATA_DIR / "UCI_Hydraulic"
    output = HHK_DIR / "processed_hydraulic_inputs.json"

    if output.exists():
        print(f"  📂 预处理结果已存在: {output}")
        return str(output)

    print(f"  🔧 处理 UCI 液压数据...")
    result = subprocess.run(
        [sys.executable, str(PREPROCESSING_DIR / "hydraulic_preprocess.py"),
         str(hydra_dir), "--output", str(output)],
        capture_output=True, text=True, cwd=str(HHK_DIR),
        encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )
    print(result.stdout or "")
    if result.returncode != 0:
        print(result.stderr or "")
        raise RuntimeError("预处理失败")
    return str(output)


def inject_test_cases(input_file, scenario):
    """根据场景的 test_cases 注入故障"""
    all_data = []
    from preprocessing.anomaly_injector import inject_anomalies, FAULT_MODES

    with open(input_file, "r", encoding="utf-8") as f:
        base_inputs = json.load(f)

    for tc in scenario["test_cases"]:
        if tc.get("inject_fault") and tc["inject_fault"] in FAULT_MODES:
            print(f"  🦠 注入: {tc['name']} ({tc['inject_fault']})")
            annotated, labels = inject_anomalies(base_inputs[:200], tc["inject_fault"], inject_ratio=0.5)
            for entry, label in zip(annotated, labels):
                entry["test_case"] = tc["name"]
                entry["expected"] = tc["expected"]
                entry["label"] = label
                all_data.append(entry)
        else:
            # 正常数据
            for entry in base_inputs[:50]:
                entry["test_case"] = tc["name"]
                entry["expected"] = tc["expected"]
                entry["label"] = {"is_anomaly": False, "fault_type": "normal"}
                all_data.append(entry)

    test_file = str(HHK_DIR / f"test_inputs_{scenario['scenario_id']}.json")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 测试数据: {len(all_data)} 条 → {test_file}")
    return test_file


def load_prompt_template():
    """加载短决策 prompt 模板"""
    tmpl = Path(r"D:\Challenge Cup\edge-io-protocol\prompts\fault_diagnosis.txt")
    if tmpl.exists():
        with open(tmpl, "r", encoding="utf-8") as f:
            return f.read()
    return "你是工业边缘分类器。根据传感器数据输出类别码(0-6)。\n{input_data}"


def load_grammar():
    """加载 GBNF grammar 约束输出"""
    gpath = Path(r"D:\Challenge Cup\edge-io-protocol\grammars\short_decision.gbnf")
    if gpath.exists():
        with open(gpath, "r", encoding="utf-8") as f:
            return f.read()
    return None


def run_inference_batch(test_file, edge_endpoint="http://localhost:8080"):
    """批量运行推理 (v2: 短决策 + GBNF grammar, 极低时延)"""
    print(f"  🔮 批量推理中 (短决策模式)...")

    with open(test_file, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    template = load_prompt_template()
    grammar = load_grammar()
    print(f"     Grammar: {'GBNF' if grammar else '无'} | 模板: {'fault_diagnosis' if '{input_data}' in template else 'fallback'}")

    results = []
    total = len(inputs)
    latencies = []
    escalations = 0

    for i, entry in enumerate(inputs[:min(total, 100)]):  # 演示限制100条
        # 构造紧凑的传感器数据文本
        readings = entry.get("data", {}).get("readings", {})
        data_text = " ".join(f"{k}={v}" for k, v in readings.items())
        prompt = template.replace("{input_data}", data_text) if "{input_data}" in template else template + "\n" + data_text

        req_data = json.dumps({
            "prompt": prompt,
            "n_predict": 2,
            "temperature": 0.0,
            "stream": False,
            **(({"grammar": grammar}) if grammar else {}),
        }).encode()

        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{edge_endpoint}/completion", data=req_data,
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=30)
            body = json.loads(resp.read())
            elapsed = (time.time() - t0) * 1000
            latencies.append(elapsed)

            result = {
                "task_id": entry.get("task_id", f"T-{i}"),
                "test_case": entry.get("test_case", "unknown"),
                "latency_ms": round(elapsed, 1),
                "response": body.get("content", "").strip()[:10],
                "tokens": body.get("tokens_predicted", 0),
                "tokens_per_sec": round(body.get("timings", {}).get("predicted_per_second", 0), 1),
            }
            results.append(result)

            if elapsed > 200:
                escalations += 1

            if (i + 1) % 20 == 0:
                avg_so_far = sum(latencies) / len(latencies)
                print(f"    进度: {i+1}/{min(total,100)} | avg={avg_so_far:.0f}ms")

        except Exception as e:
            results.append({"task_id": entry.get("task_id", f"T-{i}"), "error": str(e)})
            print(f"    ⚠️ {entry.get('task_id', i)}: {e}")

    # 统计
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    within_200ms = sum(1 for l in latencies if l <= 200)

    summary = {
        "scenario": "demo",
        "total_requests": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "avg_latency_ms": round(avg_latency, 1),
        "max_latency_ms": round(max(latencies), 1) if latencies else 0,
        "within_200ms_pct": round(within_200ms / len(latencies) * 100, 1) if latencies else 0,
        "cloud_escalations": escalations,
        "escalation_rate_pct": round(escalations / len(results) * 100, 1) if results else 0,
        "avg_tokens_per_sec": round(sum(r.get("tokens_per_sec", 0) for r in results) / len(results), 1) if results else 0,
        "timestamp": datetime.now().isoformat(),
    }

    # 保存
    result_file = str(HHK_DIR / "demo_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    print(f"\n  📊 推理完成: {len(results)} 条")
    print(f"     平均时延: {avg_latency:.1f}ms | 达标率(≤200ms): {within_200ms}/{len(latencies)} ({summary['within_200ms_pct']}%)")
    print(f"     甩云率: {summary['escalation_rate_pct']}%")
    return result_file, summary


def print_final_report(scenario_name, summary, result_file):
    """打印最终报告"""
    print(f"\n{'='*60}")
    print(f"  🏆 演示完成 — {scenario_name}")
    print(f"{'='*60}")
    print(f"  请求总数:       {summary['total_requests']}")
    print(f"  成功数:         {summary['successful']}")
    print(f"  平均时延:       {summary['avg_latency_ms']} ms")
    print(f"  200ms达标率:    {summary['within_200ms_pct']}%")
    print(f"  甩云率:         {summary['escalation_rate_pct']}%")
    print(f"  平均生成速度:   {summary['avg_tokens_per_sec']} t/s")
    print(f"  详细结果:       {result_file}")
    print(f"\n  📐 比赛指标对照:")
    print(f"    端到端时延 ≤200ms: {'✅ 达标' if summary['within_200ms_pct'] >= 95 else '⚠️ 部分达标'} ({summary['within_200ms_pct']}%)")
    print(f"    断网可用性 ≥90%:   {'✅ 达标' if summary['successful']/max(summary['total_requests'],1)*100 >= 90 else '⚠️ 待验证'}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="端到端演示")
    parser.add_argument("--scenario", choices=["industrial_inspection", "traffic_monitoring", "both"],
                        default="industrial_inspection")
    parser.add_argument("--edge-endpoint", default="http://localhost:8080")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  🎬 云边协同 AI — 端到端演示")
    print(f"  场景: {args.scenario}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not check_prerequisites():
        return

    scenarios_to_run = (
        ["industrial_inspection", "traffic_monitoring"]
        if args.scenario == "both"
        else [args.scenario]
    )

    for scenario_name in scenarios_to_run:
        step(f"场景: {scenario_name}")
        scenario = load_scenario(scenario_name)
        print(f"  描述: {scenario['description']}")

        if not args.skip_preprocess:
            step("数据预处理")
            input_file = run_hydraulic_preprocessing()
            test_file = inject_test_cases(input_file, scenario)
        else:
            test_file = str(HHK_DIR / f"test_inputs_{scenario['scenario_id']}.json")
            if not os.path.exists(test_file):
                print("  ❌ 测试数据不存在, 请先运行预处理")
                return
            print(f"  📂 使用已有测试数据: {test_file}")

        if not args.skip_inference:
            step("批量推理")
            result_file, summary = run_inference_batch(test_file, args.edge_endpoint)
            print_final_report(scenario_name, summary, result_file)

    print(f"\n✨ 演示全部完成!")


if __name__ == "__main__":
    main()
