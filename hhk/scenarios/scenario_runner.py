"""
通用场景执行器 — 加载场景配置 → 准备数据 → 注入测试用例 → 批量推理 → 收集指标

与 demo_controller.py 的区别:
  - demo_controller 硬编码了 hydraulic 预处理 + 特定逻辑
  - scenario_runner 是通用引擎, 场景配置驱动, 支持任意数据源+预处理组合

用法:
  python scenario_runner.py industrial_inspection
  python scenario_runner.py traffic_monitoring
  python scenario_runner.py industrial_inspection --data processed_motor_inputs.json
  python scenario_runner.py industrial_inspection --skip-inference --output-dir results/
"""
import sys, os, json, time, subprocess, argparse, urllib.request, hashlib, uuid
from pathlib import Path
from datetime import datetime

# 路径
HHK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HHK_DIR))  # 允许 from preprocessing.xxx import
SCENARIOS_DIR = HHK_DIR / "scenarios"
PREPROCESSING_DIR = HHK_DIR / "preprocessing"
EVALUATION_DIR = HHK_DIR / "evaluation"
DEFAULT_OUTPUT_DIR = HHK_DIR / "results"

# llama-server 默认地址
DEFAULT_EDGE_ENDPOINT = "http://localhost:8080"


# ══════════════════════════════════════════════════════
#  场景加载
# ══════════════════════════════════════════════════════

def load_scenario(name):
    """加载场景 JSON 配置"""
    path = SCENARIOS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"场景配置不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════
#  数据准备: 自动匹配预处理管线
# ══════════════════════════════════════════════════════

PREPROCESSOR_REGISTRY = {
    "hydraulic": {
        "script": "hydraulic_preprocess.py",
        "description": "UCI 液压系统 13传感器 → 标准 I/O",
        "data_dir_hint": r"D:\Challenge Cup\Data file\UCI_Hydraulic",
    },
    "motor": {
        "script": "motor_preprocess.py",
        "description": "Nature 三相电机 9传感器 → 标准 I/O",
        "data_dir_hint": r"D:\Challenge Cup\Data file\Nature_3phase_motor\27216219",
    },
    "milling": {
        "script": "sensor_csv_to_input.py",
        "description": "AI4I 铣削数据 CSV → 标准 I/O (需手动指定参数)",
        "data_dir_hint": r"D:\Challenge Cup\Data file\AI4I_2020_milling",
    },
    "generic": {
        "script": "sensor_csv_to_input.py",
        "description": "通用 CSV/TSV → 标准 I/O",
        "data_dir_hint": None,
    },
}


def detect_preprocessor_type(data_source):
    """根据 data_source 推断应用哪种预处理管线"""
    dtype = data_source.get("device_type", "")

    if any(kw in dtype for kw in ["液压", "hydraulic", "泵", "pump"]):
        return "hydraulic"
    if any(kw in dtype for kw in ["电机", "motor", "马达"]):
        return "motor"
    if any(kw in dtype for kw in ["铣", "milling", "刀具"]):
        return "milling"

    # 回退: 检查 data_files 中的文件名
    files = data_source.get("data_files", [])
    if files:
        fname = str(files[0]).lower()
        if "hydraulic" in fname:
            return "hydraulic"
        if "motor" in fname:
            return "motor"
        if "milling" in fname or "ai4i" in fname:
            return "milling"

    return "generic"


def run_preprocessing(preprocessor_type, data_source, output_path):
    """运行对应的预处理脚本, 返回输出文件路径"""
    # 确保 output_path 是绝对路径 (基于 HHK_DIR)
    if output_path and not os.path.isabs(output_path):
        output_path = str(HHK_DIR / output_path)

    if output_path and os.path.exists(output_path):
        print(f"  📂 预处理结果已存在, 跳过: {output_path}")
        return output_path

    reg = PREPROCESSOR_REGISTRY.get(preprocessor_type, PREPROCESSOR_REGISTRY["generic"])
    script = PREPROCESSING_DIR / reg["script"]

    if not script.exists():
        raise FileNotFoundError(f"预处理脚本不存在: {script}")

    print(f"  🔧 {reg['description']}")

    if preprocessor_type == "generic":
        # generic: 直接用 sensor_csv_to_input.py 的 batch 模式
        data_dir = data_source.get("data_dir", reg.get("data_dir_hint", ""))
        if not data_dir or not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        cmd = [
            sys.executable, str(script),
            str(data_dir),
            "--type", "motor",
            "--batch-mode",
            "--output", output_path or f"processed_{data_source['source_id']}_inputs.json",
        ]
    elif preprocessor_type == "motor":
        data_dir = data_source.get("data_dir", reg.get("data_dir_hint", ""))
        cmd = [
            sys.executable, str(script),
            "--data-dir", data_dir,
            "--output", output_path or "processed_motor_inputs.json",
            "--node", data_source.get("source_id", "edge-motor-01"),
        ]
    else:
        # hydraulic
        data_dir = data_source.get("data_dir", reg.get("data_dir_hint", ""))
        cmd = [
            sys.executable, str(script),
            data_dir,
            "--output", output_path or "processed_hydraulic_inputs.json",
            "--node", data_source.get("source_id", "edge-hydraulic-01"),
        ]

    print(f"  ▶️  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HHK_DIR),
                            encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""
    print(stdout_text[-500:] if len(stdout_text) > 500 else stdout_text)
    if result.returncode != 0:
        print(stderr_text[-500:] if len(stderr_text) > 500 else stderr_text)
        raise RuntimeError(f"预处理失败 (exit={result.returncode})")

    # 从 stdout 解析输出路径
    for line in result.stdout.split("\n"):
        if "→" in line or "输出:" in line or "📄" in line:
            pass
    # 直接返回我们指定的路径
    return output_path or cmd[cmd.index("--output") + 1]


# ══════════════════════════════════════════════════════
#  测试用例注入
# ══════════════════════════════════════════════════════

def inject_test_cases(input_file, scenario):
    """根据场景 test_cases 向预处理后的数据注入故障。

    Returns:
        (test_file_path, test_metadata)
    """
    from preprocessing.anomaly_injector import inject_anomalies, FAULT_MODES

    with open(input_file, "r", encoding="utf-8") as f:
        base_inputs = json.load(f)

    all_data = []
    test_summary = []

    for tc in scenario["test_cases"]:
        fault_mode = tc.get("inject_fault")
        use_count = min(200, len(base_inputs))

        if fault_mode and fault_mode in FAULT_MODES:
            print(f"  🦠 {tc['name']}: 注入 {fault_mode} (比例 0.5)")
            annotated, labels = inject_anomalies(
                base_inputs[:use_count], fault_mode, inject_ratio=0.5
            )
            anomaly_hits = sum(1 for l in labels if l.get("is_anomaly"))
            for entry, label in zip(annotated, labels):
                entry["test_case"] = tc["name"]
                entry["expected"] = tc.get("expected", {})
                entry["label"] = label
                all_data.append(entry)
            test_summary.append({
                "name": tc["name"], "fault": fault_mode,
                "count": len(annotated), "anomalies": anomaly_hits,
            })
        else:
            # 正常数据, 不做注入
            print(f"  ✅ {tc['name']}: 正常数据 (无注入)")
            for entry in base_inputs[:use_count]:
                entry["test_case"] = tc["name"]
                entry["expected"] = tc.get("expected", {})
                entry["label"] = {"is_anomaly": False, "fault_type": "normal"}
                all_data.append(entry)
            test_summary.append({
                "name": tc["name"], "fault": "none",
                "count": use_count, "anomalies": 0,
            })

    # 保存
    test_file = str(HHK_DIR / f"test_inputs_{scenario['scenario_id']}.json")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"  📦 测试数据: {len(all_data)} 条 → {os.path.basename(test_file)}")

    metadata = {
        "scenario": scenario["scenario_id"],
        "source": input_file,
        "total_test_entries": len(all_data),
        "test_cases": test_summary,
    }
    return test_file, metadata


# ══════════════════════════════════════════════════════
#  批量推理
# ══════════════════════════════════════════════════════

# ── 短决策解析 (v3: class,risk 短码) ──

CATEGORY_NAMES = {
    0: "normal", 1: "bearing_wear", 2: "bearing_failure",
    3: "current_imbalance", 4: "overheat", 5: "flow_blockage", 6: "sensor_drift",
}

# fault_type (anomaly_injector) → category code
FAULT_TO_CLASS = {
    "normal": 0, "bearing_wear": 1, "bearing_failure": 2,
    "current_imbalance": 3, "overheat": 4, "flow_blockage": 5, "sensor_drift": 6,
}

RISK_NAMES = {0: "low", 1: "medium", 2: "high", 3: "critical"}


def load_grammar():
    """加载 GBNF grammar"""
    gpath = Path(r"D:\Challenge Cup\edge-io-protocol\grammars\short_decision.gbnf")
    if gpath.exists():
        with open(gpath, "r", encoding="utf-8") as f:
            return f.read()
    return None


def parse_short_decision(raw_text):
    """解析单数字类别码 (e.g. '1') 或 'class,risk' (e.g. '1,2')"""
    import re
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.replace("```", "").strip()

    # 尝试 class,risk 格式
    m = re.search(r'(\d)\s*,\s*(\d)', text)
    if m:
        cls = int(m.group(1))
        risk = int(m.group(2))
        if cls <= 6:
            return cls, min(risk, 3), None

    # 尝试单数字
    m = re.search(r'\b([0-6])\b', text)
    if m:
        return int(m.group(1)), 1, None  # 默认中风险

    # JSON 兼容
    try:
        obj = json.loads(text)
        if "c" in obj:
            return int(obj["c"]), 1, None
    except json.JSONDecodeError:
        pass

    return 0, 0, "FORMAT_ERROR"


def stratified_sample(records, samples_per_case=50, seed=42):
    """从测试数据中分层抽样, 每类等量"""
    import random as rng_mod
    groups = {}
    for item in records:
        tc = item.get("test_case", "unknown")
        groups.setdefault(tc, []).append(item)

    rng = rng_mod.Random(seed)
    selected = []
    counts = {}
    for tc, items in sorted(groups.items()):
        rng.shuffle(items)
        n = min(samples_per_case, len(items))
        selected.extend(items[:n])
        counts[tc] = n

    rng.shuffle(selected)
    print(f"  📊 分层抽样: {counts} → 共 {len(selected)} 条")
    return selected


def compute_classification_metrics(results):
    """计算分类准确率、召回率、混淆矩阵、误报率、漏检率。

    Returns dict with all metrics.
    """
    from collections import defaultdict

    # 混淆矩阵: confusion[gt_class][pred_class] = count
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in results:
        if "error" in r or r.get("parse_error"):
            continue
        gt_fault = r.get("ground_truth", {}).get("fault_type", "normal")
        gt_class = FAULT_TO_CLASS.get(gt_fault, -1)
        pred_class = r.get("decision", {}).get("category", -1)

        if gt_class < 0 or pred_class < 0:
            continue

        total += 1
        confusion[gt_class][pred_class] += 1
        per_class[gt_class]["total"] += 1
        if gt_class == pred_class:
            correct += 1
            per_class[gt_class]["correct"] += 1

    # 总体指标
    accuracy = round(correct / total * 100, 1) if total > 0 else 0

    # 每类召回率
    recalls = {}
    for cls_id, stats in sorted(per_class.items()):
        recalls[f"{cls_id}({CATEGORY_NAMES.get(cls_id, '?')})"] = \
            round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0

    # 正常误报率: gt=normal, pred≠normal
    normal_total = per_class.get(0, {}).get("total", 0)
    false_alarms = sum(confusion[0].get(c, 0) for c in confusion[0] if c != 0)
    false_alarm_rate = round(false_alarms / normal_total * 100, 1) if normal_total > 0 else 0

    # 故障漏检率: gt≠normal, pred=normal
    fault_total = total - normal_total
    missed = sum(confusion[c].get(0, 0) for c in confusion if c != 0)
    missed_detection_rate = round(missed / fault_total * 100, 1) if fault_total > 0 else 0

    # 简化混淆矩阵 (class_name × class_name)
    cm = {}
    for gt_cls in sorted(confusion.keys()):
        gt_name = CATEGORY_NAMES.get(gt_cls, f"cls{gt_cls}")
        cm[gt_name] = {}
        for pred_cls in sorted(confusion[gt_cls].keys()):
            pred_name = CATEGORY_NAMES.get(pred_cls, f"cls{pred_cls}")
            cm[gt_name][pred_name] = confusion[gt_cls][pred_cls]

    return {
        "total_evaluated": total,
        "accuracy_pct": accuracy,
        "per_class_recall_pct": recalls,
        "false_alarm_rate_pct": false_alarm_rate,
        "missed_detection_rate_pct": missed_detection_rate,
        "confusion_matrix": cm,
    }


def build_prompt(entry, task_type, scenario_name):
    """构建 prompt — 使用 edge-io-protocol 短决策模板"""
    prompt_dir = Path(r"D:\Challenge Cup\edge-io-protocol\prompts")
    tmpl_path = prompt_dir / "fault_diagnosis.txt"
    template = None
    if tmpl_path.exists():
        with open(tmpl_path, "r", encoding="utf-8") as f:
            template = f.read()

    data_text = json.dumps(entry.get("data", entry), ensure_ascii=False, indent=2)
    if template and "{input_data}" in template:
        return template.replace("{input_data}", data_text)
    elif template:
        return template + "\n" + data_text
    else:
        return f"你是工业设备快速分类器。根据传感器数据输出短决策JSON。\n{data_text}"


def run_inference_batch(test_file, scenario, endpoint=DEFAULT_EDGE_ENDPOINT,
                        max_samples=500, n_predict=8, stratified=True, samples_per_case=50):
    """批量调用 llama-server 推理 (v3 短码 + grammar + 分层抽样)。

    Returns:
        (result_file, summary_dict)
    """
    with open(test_file, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    edge_cfg = scenario.get("edge_config", {})

    # 分层抽样
    if stratified:
        inputs = stratified_sample(inputs, samples_per_case=samples_per_case)

    total = min(len(inputs), max_samples)
    if stratified:
        total = len(inputs)  # stratified 时用全部抽样结果

    # 加载 grammar (一次)
    grammar = load_grammar()
    grammar_status = "GBNF" if grammar else "无"

    results = []
    latencies = []
    format_errors = 0
    success_count = 0

    print(f"  🔮 推理中 ({total} 条, n_predict={n_predict}, grammar={grammar_status})...")

    for i, entry in enumerate(inputs[:total]):
        task_type = entry.get("task_type", "fault_diagnosis")
        prompt = build_prompt(entry, task_type, scenario["name"])

        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "stream": False,
        }
        if grammar:
            payload["grammar"] = grammar

        req_data = json.dumps(payload).encode()

        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{endpoint}/completion", data=req_data,
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=60)
            body = json.loads(resp.read())
            elapsed = (time.time() - t0) * 1000
            latencies.append(elapsed)
            success_count += 1

            response_text = body.get("content", "")
            tps = round(body.get("timings", {}).get("predicted_per_second", 0), 1)
            tokens = body.get("tokens_predicted", 0)
            prompt_ms = round(body.get("timings", {}).get("prompt_ms", 0), 1)

            # 解析短决策 (v3: class,risk)
            category, risk_level, parse_error = parse_short_decision(response_text)

            if parse_error:
                format_errors += 1
                category, risk_level = 0, 0

            result = {
                "task_id": entry.get("task_id", f"T-{i}"),
                "test_case": entry.get("test_case", "unknown"),
                "expected": entry.get("expected", {}),
                "ground_truth": entry.get("label", {}),
                "latency_ms": round(elapsed, 1),
                "prompt_ms": prompt_ms,
                "response": response_text.strip()[:50],
                "decision": {
                    "category": category,
                    "category_name": CATEGORY_NAMES.get(category, f"?_{category}"),
                    "risk": risk_level,
                    "risk_name": RISK_NAMES.get(risk_level, "?"),
                },
                "tokens": tokens,
                "tokens_per_sec": tps,
                "parse_error": parse_error,
            }
            results.append(result)

        except Exception as e:
            results.append({
                "task_id": entry.get("task_id", f"T-{i}"),
                "test_case": entry.get("test_case", "unknown"),
                "error": str(e),
                "latency_ms": round((time.time() - t0) * 1000, 1),
            })

        if (i + 1) % 50 == 0:
            avg_ms = f"{sum(latencies)/len(latencies):.0f}" if latencies else "?"
            print(f"    [{i+1}/{total}] avg={avg_ms}ms | ok={success_count} | "
                  f"fmt_err={format_errors}")

    # 汇总
    if not latencies:
        print("\n  ❌ 所有请求均失败, 请检查 llama-server 是否在运行")
        summary = {"scenario": scenario["scenario_id"], "total_requests": len(results),
                    "successful": 0, "failed": len(results),
                    "error": "无法连接 llama-server, 请确认已启动"}
        return None, summary

    avg_latency = sum(latencies) / len(latencies)
    within_200ms = sum(1 for l in latencies if l <= 200)
    tps_values = [r.get("tokens_per_sec", 0) for r in results
                  if "error" not in r and r.get("tokens_per_sec", 0) > 0]
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0

    # 分类指标
    cls_metrics = compute_classification_metrics(results)

    summary = {
        "scenario": scenario["scenario_id"],
        "scenario_name": scenario["name"],
        "model": edge_cfg.get("model", "unknown"),
        "ngl": edge_cfg.get("ngl", 0),
        "total_requests": len(results),
        "successful": success_count,
        "failed": len(results) - success_count,
        "success_rate_pct": round(success_count / len(results) * 100, 1) if results else 0,
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(sorted(latencies)[len(latencies)//2], 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies)*0.95)], 1) if latencies else 0,
        "max_latency_ms": round(max(latencies), 1) if latencies else 0,
        "within_200ms_count": within_200ms,
        "within_200ms_pct": round(within_200ms / len(latencies) * 100, 1) if latencies else 0,
        "format_error_count": format_errors,
        "format_error_pct": round(format_errors / len(results) * 100, 1) if results else 0,
        "avg_tokens_per_sec": round(avg_tps, 1),
        "avg_tokens": round(sum(r.get("tokens", 0) for r in results
                                 if "error" not in r) / max(success_count, 1), 1),
        "grammar_used": grammar_status,
        "classification": cls_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # 比赛指标对照
    metric_targets = scenario.get("metrics_targets", {})
    verdicts = []
    target_latency = metric_targets.get("end_to_end_latency_ms", 200)
    verdicts.append({
        "metric": "端到端平均时延",
        "value": f"{summary['avg_latency_ms']} ms",
        "target": f"≤{target_latency}ms",
        "passed": summary["avg_latency_ms"] <= target_latency,
    })
    verdicts.append({
        "metric": "200ms达标率",
        "value": f"{summary['within_200ms_pct']}%",
        "target": "≥95%",
        "passed": summary["within_200ms_pct"] >= 95,
    })
    verdicts.append({
        "metric": "成功率",
        "value": f"{summary['success_rate_pct']}%",
        "target": "≥90%",
        "passed": summary["success_rate_pct"] >= 90,
    })
    cls = summary.get("classification", {})
    verdicts.append({
        "metric": "分类准确率",
        "value": f"{cls.get('accuracy_pct', 0)}%",
        "target": "≥80%",
        "passed": cls.get("accuracy_pct", 0) >= 80,
    })
    verdicts.append({
        "metric": "故障漏检率",
        "value": f"{cls.get('missed_detection_rate_pct', 0)}%",
        "target": "≤10%",
        "passed": cls.get("missed_detection_rate_pct", 100) <= 10,
    })
    summary["verdicts"] = verdicts

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = str(HHK_DIR / "results" / f"run_{scenario['scenario_id']}_{ts}.json")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    print(f"\n  📊 推理完成: {len(results)} 条 | 平均时延: {avg_latency:.0f}ms | "
          f"达标率: {summary['within_200ms_pct']}% | 格式异常: {summary['format_error_count']}")
    return result_file, summary


# ══════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════

def run_scenario(scenario_name, preprocessed_data=None, endpoint=DEFAULT_EDGE_ENDPOINT,
                 skip_preprocess=False, skip_inference=False, max_samples=500):
    """完整的场景执行管线。

    Args:
        scenario_name: 场景名 (不带 .json)
        preprocessed_data: 可选, 跳过预处理直接使用已有数据文件
        endpoint: llama-server 地址
        skip_preprocess: 跳过数据预处理
        skip_inference: 跳过推理
        max_samples: 推理最大样本数
    """
    print("=" * 60)
    print(f"  🎬 场景执行: {scenario_name}")
    print(f"  ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 加载场景
    scenario = load_scenario(scenario_name)
    print(f"\n  📋 {scenario['name']}")
    print(f"  描述: {scenario['description']}")
    print(f"  边端模型: {scenario['edge_config'].get('model', 'N/A')}")
    print(f"  测试用例: {len(scenario['test_cases'])} 个")
    print(f"  指标目标: {json.dumps(scenario.get('metrics_targets', {}), ensure_ascii=False)}")

    metadata = {
        "scenario": scenario_name,
        "scenario_id": scenario["scenario_id"],
        "started_at": datetime.now().isoformat(),
    }

    # 2. 数据准备
    print(f"\n{'─'*40}")
    print(f"  📥 数据准备")

    if preprocessed_data:
        # 用户指定了已处理数据, 跳过预处理但保留测试注入
        input_file = preprocessed_data
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"指定的数据文件不存在: {input_file}")
        print(f"  📂 使用指定数据: {input_file} ({os.path.getsize(input_file)/1024/1024:.1f} MB)")
    elif not skip_preprocess:
        data_sources = scenario.get("data_sources", [])
        if not data_sources:
            raise ValueError("场景未定义 data_sources, 且未指定 --data")

        ds = data_sources[0]
        ptype = detect_preprocessor_type(ds)
        print(f"  🔍 检测数据源类型: {ptype} ({ds.get('device_type', 'unknown')})")

        output_name = f"processed_{scenario['scenario_id']}_inputs.json"
        input_file = run_preprocessing(ptype, ds, output_name)
        metadata["preprocessor_type"] = ptype
        metadata["preprocessor_output"] = input_file
    else:
        # skip_preprocess 且无 --data: 尝试用已有测试数据
        test_file = str(HHK_DIR / f"test_inputs_{scenario['scenario_id']}.json")
        if os.path.exists(test_file):
            print(f"  ⏩ 跳过数据准备, 使用已有测试数据: {test_file}")
            metadata["test_file"] = test_file
            input_file = None  # 跳过注入
        else:
            raise FileNotFoundError(
                f"测试数据不存在: {test_file}\n"
                f"  请先运行预处理生成数据, 或使用 --data 指定已有数据文件"
            )

    # 注入测试用例 (如果有新数据)
    if input_file:
        print(f"\n{'─'*40}")
        print(f"  🦠 注入测试用例")
        test_file, test_meta = inject_test_cases(input_file, scenario)
        metadata["test_file"] = test_file
        metadata["test_metadata"] = test_meta

    # 3. 推理
    if not skip_inference:
        print(f"\n{'─'*40}")
        print(f"  🔮 批量推理")
        result_file, summary = run_inference_batch(
            test_file, scenario, endpoint, max_samples=max_samples
        )
        metadata["result_file"] = result_file
        metadata["summary"] = summary
    else:
        print(f"  ⏩ 跳过推理")
        result_file = None
        summary = {}

    # 4. 输出结果
    print(f"\n{'='*60}")
    print(f"  🏆 场景执行完成 — {scenario_name}")
    print(f"{'='*60}")

    if not skip_inference and summary:
        if summary.get("error"):
            print(f"  ❌ {summary['error']}")
        else:
            cls = summary.get("classification", {})
            print(f"  请求总数:       {summary.get('total_requests', '?')}")
            print(f"  成功率:         {summary.get('success_rate_pct', '?')}%")
            print(f"  格式异常率:     {summary.get('format_error_pct', '?')}%")
            print(f"  平均时延:       {summary.get('avg_latency_ms', '?')} ms")
            print(f"  平均 Token:     {summary.get('avg_tokens', '?')}")
            print(f"  200ms达标率:    {summary.get('within_200ms_pct', '?')}%")
            print(f"  平均生成速度:   {summary.get('avg_tokens_per_sec', '?')} t/s")
            print(f"  Grammar:        {summary.get('grammar_used', '?')}")
            print(f"\n  📐 分类指标:")
            print(f"  总体准确率:     {cls.get('accuracy_pct', '?')}%")
            print(f"  正常误报率:     {cls.get('false_alarm_rate_pct', '?')}%")
            print(f"  故障漏检率:     {cls.get('missed_detection_rate_pct', '?')}%")
            print(f"  每类召回率:     {cls.get('per_class_recall_pct', {})}")
            print(f"\n  📐 比赛指标:")
            for v in summary.get("verdicts", []):
                icon = "PASS" if v["passed"] else "WARN"
                print(f"    [{icon}] {v['metric']}: {v['value']} (要求: {v['target']})")
        if result_file:
            print(f"\n  📄 详细结果: {result_file}")

    # 5. 保存运行元数据
    meta_file = result_file.replace(".json", "_meta.json") if result_file else \
                str(HHK_DIR / "results" / f"meta_{scenario['scenario_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    if result_file:
        os.makedirs(os.path.dirname(meta_file), exist_ok=True)
        metadata["completed_at"] = datetime.now().isoformat()
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="通用场景执行器")
    parser.add_argument("scenario",
                        choices=["industrial_inspection", "traffic_monitoring", "both"],
                        help="场景名")
    parser.add_argument("--data", default=None,
                        help="指定已预处理的数据文件 (跳过预处理步骤)")
    parser.add_argument("--endpoint", default=DEFAULT_EDGE_ENDPOINT,
                        help=f"llama-server 地址 (默认 {DEFAULT_EDGE_ENDPOINT})")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="跳过数据预处理")
    parser.add_argument("--skip-inference", action="store_true",
                        help="跳过批量推理 (仅生成测试数据)")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="推理最大样本数 (默认 500)")
    parser.add_argument("--output-dir", default=str(HHK_DIR / "results"),
                        help="结果输出目录")
    args = parser.parse_args()

    scenarios_to_run = (
        ["industrial_inspection", "traffic_monitoring"]
        if args.scenario == "both"
        else [args.scenario]
    )

    all_meta = []
    for sname in scenarios_to_run:
        try:
            meta = run_scenario(
                sname,
                preprocessed_data=args.data,
                endpoint=args.endpoint,
                skip_preprocess=args.skip_preprocess,
                skip_inference=args.skip_inference,
                max_samples=args.max_samples,
            )
            all_meta.append(meta)
        except Exception as e:
            print(f"\n❌ 场景 {sname} 执行失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✨ 全部场景执行完毕 ({len(all_meta)}/{len(scenarios_to_run)} 成功)")


if __name__ == "__main__":
    main()
