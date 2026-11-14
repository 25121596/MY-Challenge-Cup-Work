"""
边端统一推理入口 — 给 CSV 就出结果

自动完成:
  1. 检测数据格式 (列名匹配 → 液压/电机/铣削/通用)
  2. 选对应预处理 → 标准 I/O
  3. 检查 llama-server (不在就启动)
  4. 短决策推理 (GBNF, ≤200ms)
  5. 输出分类结果

用法:
  python edge_run.py --data "data/UCI_Hydraulic"
  python edge_run.py --data "data/Nature_3phase_motor"
  python edge_run.py --data "data/AI4I_2020_milling/ai4i2020.csv"
  python edge_run.py --data any_sensor.csv --type motor
  python edge_run.py --data data/UCI_Hydraulic --endpoint http://127.0.0.1:8090
  python edge_run.py --data data/UCI_Hydraulic --max-samples 50
"""
import sys, os, json, time, subprocess, csv, argparse, urllib.request, math, hashlib
from pathlib import Path

# ══════════════════════════════════════════════════════
#  路径配置
# ══════════════════════════════════════════════════════
HHK_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = HHK_DIR / "preprocessing"
PROJECT_ROOT = HHK_DIR  # D:\llama.cpp (所有组件都在这个目录下)
BUILD_DIR = HHK_DIR / "build" / "bin" / "Release"
LLAMA_SERVER_EXE = BUILD_DIR / "llama-server.exe"
MODELS_DIR = HHK_DIR / "models"
MODEL_PATH = MODELS_DIR / "Qwen_Qwen3-1.7B-IQ4_XS.gguf"
EDGE_IO_DIR = HHK_DIR / "edge-io-protocol"

sys.path.insert(0, str(HHK_DIR))
sys.path.insert(0, str(EDGE_IO_DIR))
sys.path.insert(0, str(HHK_DIR / "data"))
from decision_contract import ai4i_unscored_review_fields, compact_cloud_payload, utc_now
from feedback_loop import append_observation, build_observation

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_CTX = 1024
DEFAULT_NGL = 0  # CPU 模式


# ══════════════════════════════════════════════════════
#  1. 数据格式自动检测
# ══════════════════════════════════════════════════════

# 列名特征 → 数据类型
SIGNATURES = {
    "hydraulic": ["PS1", "PS2", "FS1", "TS1", "VS1", "EPS1"],
    "motor": ["x", "y", "Z", "I1", "I2", "I3", "V1", "V2", "V3"],
    "milling": ["Air temperature [K]", "Torque [Nm]", "Tool wear [min]",
                "Rotational speed [rpm]", "Process temperature [K]"],
}


def detect_data_type(data_path):
    """根据文件/目录内容自动检测数据类型"""
    data_path = Path(data_path)

    # 如果是目录, 找第一个 CSV/TXT 文件
    if data_path.is_dir():
        files = list(data_path.glob("*.csv")) + list(data_path.glob("*.txt"))
        # 特殊: Nature 电机目录有 "FILE 1.csv"
        motor_files = [f for f in files if f.name.startswith("FILE ") and f.suffix == ".csv"]
        if motor_files and (data_path / "LABEL DATASET.csv").exists():
            return "motor"
        # 特殊: UCI 液压目录有 PS1.txt
        if (data_path / "PS1.txt").exists() or (data_path / "FS1.txt").exists():
            return "hydraulic"
        if not files:
            return "generic"
        target_file = files[0]
    else:
        target_file = data_path

    # 读第一行 (列名)
    try:
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
    except Exception:
        return "generic"

    cols = [c.strip().strip("\ufeff") for c in first_line.replace("\t", ",").split(",")]
    cols_lower = [c.lower() for c in cols]

    # 匹配签名
    for dtype, signatures in SIGNATURES.items():
        hits = sum(1 for s in signatures if s.lower() in cols_lower or s in cols)
        if hits >= 3:
            return dtype

    return "generic"


# ══════════════════════════════════════════════════════
#  2. 预处理路由
# ══════════════════════════════════════════════════════

def run_preprocessing(data_path, data_type, output_path):
    """调用对应预处理脚本, 返回标准 I/O JSON 路径"""
    data_path = str(Path(data_path).resolve())
    output_path = str(Path(output_path).resolve())

    # 泛化模式: 不依赖已知传感器类型, 直接对全部数值列做统计推断
    if data_type in ("generic", "generic_anomaly"):
        target = data_path if os.path.isfile(data_path) else data_path
        cmd = [sys.executable, str(PREPROCESSING_DIR / "sensor_csv_to_input.py"),
               target, "--type", "generic", "--output", output_path,
               "--max-rows", "5000", "--batch", "60",
               "--task", "generic_anomaly"]
    elif data_type == "hydraulic":
        cmd = [sys.executable, str(PREPROCESSING_DIR / "hydraulic_preprocess.py"),
               data_path, "--output", output_path]
    elif data_type == "motor":
        cmd = [sys.executable, str(PREPROCESSING_DIR / "motor_preprocess.py"),
               "--data-dir", data_path, "--output", output_path]
    elif data_type == "milling":
        target = data_path if os.path.isfile(data_path) else data_path
        cmd = [sys.executable, str(PREPROCESSING_DIR / "sensor_csv_to_input.py"),
               target, "--type", "milling", "--output", output_path,
               "--max-rows", "2000", "--batch", "60"]
    else:
        target = data_path if os.path.isfile(data_path) else data_path
        cmd = [sys.executable, str(PREPROCESSING_DIR / "sensor_csv_to_input.py"),
               target, "--type", "generic", "--output", output_path,
               "--max-rows", "5000", "--batch", "60",
               "--task", "generic_anomaly"]

    print(f"  ▶ {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace",
                            cwd=str(HHK_DIR),
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if result.returncode != 0:
        print(f"  ❌ 预处理失败:\n{(result.stderr or '')[-300:]}")
        return None

    if os.path.exists(output_path) and os.path.getsize(output_path) > 10:
        return output_path
    print(f"  ❌ 预处理输出为空")
    return None


# ══════════════════════════════════════════════════════
#  3. 服务器管理
# ══════════════════════════════════════════════════════

def ensure_server(endpoint, ngl=DEFAULT_NGL, ctx=DEFAULT_CTX):
    """确保 llama-server 在运行, 返回是否需要清理"""
    try:
        urllib.request.urlopen(f"{endpoint}/health", timeout=3)
        print(f"  ✅ llama-server 已在运行 ({endpoint})")
        return None  # 不需要清理
    except Exception:
        pass

    print(f"  🚀 启动 llama-server (ngl={ngl}, ctx={ctx})...")
    if not LLAMA_SERVER_EXE.exists():
        print(f"  ❌ 找不到: {LLAMA_SERVER_EXE}")
        return False
    if not MODEL_PATH.exists():
        print(f"  ❌ 找不到模型: {MODEL_PATH}")
        return False

    proc = subprocess.Popen(
        [str(LLAMA_SERVER_EXE), "-m", str(MODEL_PATH),
         "--host", "127.0.0.1", "--port", endpoint.split(":")[-1],
         "-ngl", str(ngl), "-c", str(ctx)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # 等待就绪
    for i in range(60):
        try:
            urllib.request.urlopen(f"{endpoint}/health", timeout=2)
            print(f"  ✅ 服务器就绪 ({i*0.5:.1f}s)")
            return proc
        except Exception:
            time.sleep(0.5)

    proc.kill()
    print(f"  ❌ 服务器启动超时")
    return False


# ══════════════════════════════════════════════════════
#  4. 批量推理 (短决策)
# ══════════════════════════════════════════════════════

def load_prompt_and_grammar():
    """加载 prompt 模板和 GBNF grammar (泛化模式默认)"""
    # 泛化 prompt (不假设传感器类型)
    generic_prompt_path = EDGE_IO_DIR / "prompts" / "generic_anomaly.txt"
    generic_grammar_path = EDGE_IO_DIR / "grammars" / "generic_decision.gbnf"

    template = None
    if generic_prompt_path.exists():
        with open(generic_prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

    grammar = None
    if generic_grammar_path.exists():
        with open(generic_grammar_path, "r", encoding="utf-8") as f:
            grammar = f.read()

    return template, grammar


def submit_cloud_payload(payload, endpoint):
    """Upload only the compact, program-generated escalation contract."""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"http_status": response.status, "body": json.loads(response.read())}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_inference(input_file, endpoint, max_samples=100, cloud_escalation_endpoint=None,
                  feedback_log=None):
    """批量短决策推理 — 复用 edge_inference.infer() 产出标准 output.json"""
    with open(input_file, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    # 复用标准推理框架 (edge-io-protocol/edge_inference.py)
    import edge_inference
    edge_inference.LLAMA_SERVER = endpoint

    # An explicit AI4I route must never silently fall back to Qwen when its
    # schema or local artifact is broken.  Such a record is fail-closed and
    # sent to review without a synthetic probability.
    ai4i_requested = any(_declares_ai4i_route(entry) for entry in inputs[:max_samples])
    moe = None
    moe_init_error = None
    if ai4i_requested:
        try:
            from ai4i_moe_inference import Ai4iMoeInference
            moe = Ai4iMoeInference()
            print("  AI4I route: 8 features -> local tabular MoE")
        except Exception as exc:
            moe_init_error = exc
            print(f"  Local MoE unavailable; AI4I records will fail closed: {type(exc).__name__}")

    total = min(len(inputs), max_samples)
    results = []       # 标准 output.json 列表
    latencies = []

    print(f"  🔮 推理中 ({total} 条, 标准输出 output.json)...")

    for i, entry in enumerate(inputs[:total]):
        declared_ai4i = _declares_ai4i_route(entry)
        started = time.perf_counter_ns()
        try:
            if declared_ai4i:
                if moe is None:
                    raise RuntimeError("AI4I local model initialization failed") from moe_init_error
                r = {"output": moe.infer(entry), "raw_model_response": "", "route": "ai4i_tabular_moe"}
            else:
                r = edge_inference.infer(entry, use_grammar=True)
            out = r["output"]                       # 标准 output.json
            out["raw_model_response"] = r.get("raw_model_response", "")
            out["route"] = r.get("route", "qwen_text")
            if out.get("review_state") == "required" and "cloud_payload" not in out:
                out["cloud_payload"] = compact_cloud_payload(
                    entry, out
                )
            if cloud_escalation_endpoint and out.get("cloud_payload"):
                out["cloud_ack"] = submit_cloud_payload(
                    out["cloud_payload"], cloud_escalation_endpoint
                )
            if feedback_log:
                append_observation(feedback_log, build_observation(
                    entry, out, raw_model_response=out.get("raw_model_response", ""),
                    timing=r.get("timing"), cloud_ack=out.get("cloud_ack")))
            results.append(out)
            latencies.append(out.get("latency_ms", 0))
        except Exception as exc:
            if not declared_ai4i:
                results.append({"task_id": entry.get("task_id", f"T-{i}"), "error": str(exc)})
                continue
            reason = _ai4i_failure_reason(exc)
            fields = ai4i_unscored_review_fields(entry, reason)
            output = {
                "task_id": entry.get("task_id", f"T-{i}"),
                "node_id": entry.get("node_id", "edge-unknown"),
                "observed_at": entry.get("observed_at", entry.get("timestamp")),
                "received_at": utc_now(),
                "decided_at": utc_now(),
                "latency_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
                "decision": {
                    "action": fields["action"],
                    "level": "critical" if fields["action"] == "protect" else "warning",
                    "summary": "AI4I result unavailable; fail-closed cloud review required",
                },
                "route": "ai4i_tabular_moe_fail_closed",
                "execution_error": {"reason": reason, "exception_type": type(exc).__name__},
            }
            output.update(fields)
            output["cloud_payload"] = compact_cloud_payload(entry, output)
            if cloud_escalation_endpoint:
                output["cloud_ack"] = submit_cloud_payload(
                    output["cloud_payload"], cloud_escalation_endpoint
                )
            if feedback_log:
                append_observation(feedback_log, build_observation(
                    entry, output, cloud_ack=output.get("cloud_ack")))
            results.append(output)
            latencies.append(output["latency_ms"])

        if (i + 1) % 20 == 0:
            avg = sum(latencies) / len(latencies) if latencies else 0
            print(f"    [{i+1}/{total}] avg={avg:.0f}ms")

    return results, latencies


# ══════════════════════════════════════════════════════
#  5. 预处理跳过检测 (已是标准 JSON 就直接推理)
# ══════════════════════════════════════════════════════

def _is_standard_input_json(filepath):
    """检查是否已是标准 input.json 格式 (无需预处理)。

    video_adapter.py / sensor_csv_to_input.py 的输出都符合此格式:
      [{"task_id": ..., "data": {"readings": {...}}}, ...]
    """
    path = Path(filepath)
    if path.suffix.lower() != ".json":
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
            has_task = "task_id" in entry
            data = entry.get("data", {})
            has_readings = "readings" in data
            has_bound_features = "features" in data and "feature_schema_id" in data
            return has_task and (has_readings or has_bound_features)
    except (json.JSONDecodeError, Exception):
        pass
    return False


def _declares_ai4i_route(record):
    """Return true only for records explicitly claiming the AI4I route."""
    data = record.get("data", {})
    return (data.get("model_route") == "ai4i_tabular_moe_v1" or
            data.get("feature_schema_id") == "ai4i_8d_v1")


def _ai4i_failure_reason(exc):
    """Map untrusted exception text into the closed protocol vocabulary."""
    message = str(exc).lower()
    schema_markers = (
        "feature_schema", "model_route", "preprocessing_version", "features-only",
        "exactly eight", "features must", "one-hot", "ground_truth",
    )
    return "feature_schema_mismatch" if any(marker in message for marker in schema_markers) else "model_exception"


def _records_are_ai4i(records, max_samples):
    """Strictly recognize the frozen AI4I eight-feature online contract."""
    sample = records[:max_samples]
    if not sample:
        return False
    for record in sample:
        data = record.get("data", {})
        features = data.get("features")
        if data.get("device_id") != "milling-machine-01" or not isinstance(features, list) or len(features) != 8:
            return False
        if data.get("feature_schema_id") != "ai4i_8d_v1" or data.get("model_route") != "ai4i_tabular_moe_v1":
            return False
        schema_path = EDGE_IO_DIR / "feature_schemas" / "ai4i_8d_v1.json"
        expected_version = "sha256:" + hashlib.sha256(schema_path.read_bytes()).hexdigest()
        if data.get("preprocessing_version") != expected_version or "ground_truth" in data:
            return False
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in features):
            return False
        if sum(features[:3]) != 1 or not all(value in (0, 1) for value in features[:3]):
            return False
    return True


def _is_ai4i_moe_input(filepath, max_samples):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            records = json.load(f)[:max_samples]
        return bool(records) and all(_declares_ai4i_route(record) for record in records)
    except (OSError, json.JSONDecodeError, TypeError):
        return False


# ══════════════════════════════════════════════════════
#  6. 主流程
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="边端统一推理入口")
    parser.add_argument("--data", required=True, help="数据文件或目录路径 (CSV/TSV/标准JSON)")
    parser.add_argument("--type", default=None,
                        choices=["hydraulic", "motor", "milling", "generic", "generic_anomaly"],
                        help="手动指定数据类型 (默认自动检测，未知数据自动走泛化模式)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="llama-server 地址")
    parser.add_argument("--max-samples", type=int, default=100, help="最大推理样本数")
    parser.add_argument("--ngl", type=int, default=DEFAULT_NGL, help="GPU 层数 (0=CPU)")
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX, help="上下文长度")
    parser.add_argument("--output", default=None, help="结果输出路径")
    parser.add_argument("--results-output", default=None,
                        help="path for the JSON execution report (defaults to edge_run_results.json)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="跳过预处理，数据已是标准 input.json")
    parser.add_argument("--latency-target-ms", type=float, default=200.0,
                        help="completed local decision latency target")
    parser.add_argument("--cloud-escalation-endpoint", default=None,
                        help="optional POST endpoint, e.g. http://127.0.0.1:8080/api/v1/escalations")
    parser.add_argument("--feedback-log", default=None,
                        help="append version-bound edge/cloud shadow observations to JSONL")
    args = parser.parse_args()

    print("=" * 60)
    print("  ⚡ 边端统一推理入口")
    print(f"  数据: {args.data}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    input_file = args.data

    # ── 已是标准 JSON → 跳过预处理, 直接推理 ──
    if args.skip_preprocess or _is_standard_input_json(args.data):
        if not _is_standard_input_json(args.data):
            print("  ⚠️ --skip-preprocess 但文件不像标准 input.json, 继续尝试...")
        else:
            print(f"\n  📋 检测到标准 input.json 格式, 跳过预处理")
        with open(args.data, "r", encoding="utf-8") as f:
            count = len(json.load(f))
        print(f"  ✅ {count} 条标准输入, 直接推理")
    else:
        # 1. 检测数据类型
        data_type = args.type or detect_data_type(args.data)
        print(f"\n  📋 数据类型: {data_type}")

        # 2. 预处理
        print(f"\n{'─'*40}")
        print(f"  📥 预处理")
        output_json = args.output or str(HHK_DIR / f"edge_run_{data_type}_inputs.json")
        input_file = run_preprocessing(args.data, data_type, output_json)
        if not input_file:
            print("  ❌ 预处理失败, 退出")
            return

        with open(input_file, "r", encoding="utf-8") as f:
            count = len(json.load(f))
        print(f"  ✅ {count} 条标准输入")

    # 3. 确保服务器运行
    print(f"\n{'─'*40}")
    print(f"  🖥️ 推理服务")
    ai4i_direct = _is_ai4i_moe_input(input_file, args.max_samples)
    if ai4i_direct:
        server_proc = None
        print("  AI4I 8-feature contract confirmed: local MoE, 0 generated tokens; llama-server bypassed")
    else:
        server_proc = ensure_server(args.endpoint, args.ngl, args.ctx)
        if server_proc is False:
            print("  ❌ 无法启动服务器, 退出")
            return

    # 4. 推理
    print(f"\n{'─'*40}")
    print(f"  🔮 短决策推理")
    try:
        results, latencies = run_inference(
            input_file, args.endpoint, args.max_samples,
            cloud_escalation_endpoint=args.cloud_escalation_endpoint,
            feedback_log=args.feedback_log,
        )
    finally:
        # 如果我们启动了服务器, 结束后关闭
        if server_proc and server_proc is not None:
            server_proc.terminate()

    # 5. 输出报告
    print(f"\n{'='*60}")
    print(f"  📊 结果")
    print(f"{'='*60}")

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        within_target = sum(1 for l in latencies if l <= args.latency_target_ms)
        escalated = sum(1 for r in results if r.get("review_state") == "required")

        print(f"  请求数:     {len(results)}")
        print(f"  成功数:     {sum(1 for r in results if 'error' not in r)}")
        print(f"  平均时延:   {avg_lat:.3f} ms")
        print(f"  P95 时延:   {p95:.3f} ms")
        print(f"  {args.latency_target_ms:g}ms达标:  {within_target}/{len(latencies)} ({within_target/len(latencies)*100:.0f}%)")
        print(f"  需要复核:   {escalated}/{len(results)} ({escalated/max(len(results),1)*100:.0f}%)")

        # 决策分布 (从标准 output 的 decision.action 提取)
        from collections import Counter
        acts = Counter(r.get("decision", {}).get("action", "错误")
                       for r in results if "error" not in r)
        print(f"\n  决策分布 (action):")
        for name, cnt in acts.most_common():
            print(f"    {name:<12}: {cnt}")

    # 保存结果
    result_file = args.results_output or str(HHK_DIR / "edge_run_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"measurement_scope": "per-record local inference output latency; excludes HTTP upload, cloud queueing, cloud inference, and cloud response",
                   "latency_avg_ms": round(avg_lat, 3) if latencies else 0,
                   "total": len(results), "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  📄 详细结果: {result_file}")
    print(f"\n✨ 完成!")


if __name__ == "__main__":
    main()
