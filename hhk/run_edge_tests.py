# -*- coding: utf-8 -*-
"""
边端全链路自检 (一键) —— 不需要模型/llama-server 即可验证

覆盖 (除"模型推理"外的全部边端环节):
  [1] 模块导入      edge_inference / stream_adapter / video_adapter / edge_run
  [2] 静态资源      schemas / prompts / grammars / task_types (存在且 JSON 合法)
  [3] 推理纯函数    parse_short_decision / make_output / build_prompt / load_grammar
  [4] 入口纯函数    detect_data_type / _is_standard_input_json
  [5] 流/视频适配   raw_to_standard_input / batch_features / build_standard_input
  [6] 预处理管线    sensor_csv_to_input.py 实跑 (临时CSV → 标准JSON)
  [7] 子测试        test_conflict_detection.py / test_offline_resilience.py
  [8] 联机冒烟(可选) 若本地 llama-server 在线, 跑一次真实 infer; 否则 SKIP

用法:
  python run_edge_tests.py
  python run_edge_tests.py --endpoint http://127.0.0.1:8080

说明: 模型与 llama-server 位于另外的文件夹, 本脚本不依赖它们;
      [8] 仅在检测到 /health 时才真正推理, 用于确认联机链路。
"""
import sys, os, json, time, argparse, tempfile, shutil, subprocess, urllib.request
from pathlib import Path

HHK_DIR = Path(__file__).resolve().parent
EDGE_IO_DIR = HHK_DIR / "edge-io-protocol"
PREPROCESSING_DIR = HHK_DIR / "preprocessing"
TESTS_DIR = HHK_DIR / "tests"
sys.path.insert(0, str(HHK_DIR))
sys.path.insert(0, str(EDGE_IO_DIR))

# ── 受保护的模块导入 (失败不崩溃, 记为 None) ──
MODS, IMPORT_ERR = {}, {}
for name in ("edge_inference", "stream_adapter", "video_adapter", "edge_run"):
    try:
        MODS[name] = __import__(name)
    except Exception as e:  # noqa
        MODS[name] = None
        IMPORT_ERR[name] = f"{type(e).__name__}: {e}"

EI = MODS["edge_inference"]
SA = MODS["stream_adapter"]
VA = MODS["video_adapter"]
ER = MODS["edge_run"]


# ══════════════════════════════════════════════════════
#  检查框架
# ══════════════════════════════════════════════════════

class Checker:
    def __init__(self):
        self.results = []

    def check(self, name, fn):
        try:
            ok = bool(fn())
            print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        except Exception as e:  # noqa
            ok = False
            print(f"    [FAIL] {name}  ({type(e).__name__}: {e})")
        self.results.append((name, ok))
        return ok

    def note(self, text):
        print(f"    [NOTE] {text}")

    def passed(self):
        return all(ok for _, ok in self.results)

    def summary(self):
        n = len(self.results)
        p = sum(1 for _, ok in self.results if ok)
        return p, n


def run_py(args, timeout=180):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, *args], cwd=str(HHK_DIR), env=env,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _hdr(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ══════════════════════════════════════════════════════
#  [1] 模块导入
# ══════════════════════════════════════════════════════

def group_imports(ck):
    _hdr("[1] 模块导入")
    for name in ("edge_inference", "stream_adapter", "video_adapter", "edge_run"):
        ck.check(f"import {name}", (lambda n=name: MODS[n] is not None))
    for n, err in IMPORT_ERR.items():
        ck.note(f"{n} 导入失败: {err}")
    if VA is not None and not getattr(VA, "HAS_CV2", False):
        ck.note("video_adapter: 未安装 opencv (cv2), 特征提取函数仍可测, 实采视频需 pip install opencv-python")


# ══════════════════════════════════════════════════════
#  [2] 静态资源
# ══════════════════════════════════════════════════════

def group_static(ck):
    _hdr("[2] 静态资源 (schemas / prompts / grammars)")

    def json_ok(rel):
        p = EDGE_IO_DIR / rel
        with open(p, "r", encoding="utf-8") as f:
            json.load(f)
        return True

    def exists(rel):
        return (EDGE_IO_DIR / rel).exists()

    # task_types.json 会被 edge_inference 运行时 json.load, 必须严格合法
    ck.check("JSON 合法(运行时加载): task_types.json", lambda: json_ok("task_types.json"))

    # 下列 schema 是人读规范文档 (JSONC, 含 // 注释), 运行时不解析 → 只校验存在
    for rel in ("schemas/input.json", "schemas/output.json", "schemas/cloud_escalation.json"):
        ck.check(f"存在(JSONC规范文档): {rel}", (lambda r=rel: exists(r)))

    for rel in ("grammars/short_decision.gbnf", "grammars/generic_decision.gbnf",
                "prompts/fault_diagnosis.txt", "prompts/generic_anomaly.txt",
                "prompts/video_anomaly.txt"):
        ck.check(f"存在: {rel}", (lambda r=rel: exists(r)))


# ══════════════════════════════════════════════════════
#  [3] 推理纯函数
# ══════════════════════════════════════════════════════

def group_inference(ck):
    _hdr("[3] 推理纯函数 (无网络)")
    if EI is None:
        ck.check("edge_inference 可用", lambda: False)
        return

    def t_parse_single():
        c, err = EI.parse_short_decision("2")
        return c == 2 and err is None

    def t_parse_legacy_json_rejected():
        c, err = EI.parse_short_decision('{"c":1,"s":2,"r":1}')
        return c is None and err == "FORMAT_ERROR"

    def t_parse_json():
        # 真实链路由 GBNF 约束为裸数字；JSON 仅接受严格的 {"c":N}
        # 兼容对象，避免静默丢弃旧式 score/routing 字段。
        c, err = EI.parse_short_decision('{"c":2}')
        return c == 2 and err is None

    def t_parse_garbage():
        c, err = EI.parse_short_decision("hello world")
        return c is None and err == "FORMAT_ERROR"

    def t_parse_generic_clamp():
        # generic_mode: 超范围(>3)应被夹到 3 并标记 UNKNOWN_CLASS
        c, err = EI.parse_short_decision("5", generic_mode=True)
        return c == 3 and err == "UNKNOWN_CLASS"

    def t_make_output():
        o = EI.make_output("T-1", "edge-zoneX-01", 2, 12.3)
        return (o["decision"]["provisional_category"]["code"] == 2
                and o["decision"]["action"] == "continue_with_review"
                and o["consistency"]["zone"] == "edge-zoneX"
                and len(o["consistency"]["decision_hash"]) == 16
                and o["failure_probability"] is None
                and o["local_prediction"] is None
                and o["threshold_review_margin"] is None
                and "confidence" not in o)

    def t_make_output_generic():
        o = EI.make_output("T-2", "edge-01", 3, 30.0, generic_mode=True)
        return (o["decision"]["provisional_category"]["name"] == "severe_anomaly"
                and o["review_state"] == "required")

    def t_build_prompt():
        p = EI.build_prompt("fault_diagnosis",
                            {"data": {"readings": {"vibration_mm_s": 5.1, "temperature_c": 88}}})
        return "5.1" in p and "vibration_mm_s" in p

    def t_grammar_fault():
        g = EI.load_grammar("fault_diagnosis")
        return g is not None and "[0-6]" in g

    def t_grammar_generic():
        g = EI.load_grammar("generic_anomaly")
        return g is not None and "[0-3]" in g

    ck.check("parse: 单数字 '2'", t_parse_single)
    ck.check("parse: legacy JSON score/routing fields are rejected", t_parse_legacy_json_rejected)
    ck.check("parse: strict JSON compatibility restores only its class", t_parse_json)
    ck.check("parse: 乱码 → FORMAT_ERROR", t_parse_garbage)
    ck.check("parse: generic 越界 → 夹取+UNKNOWN", t_parse_generic_clamp)
    ck.check("make_output: provisional Qwen class + mandatory review", t_make_output)
    ck.check("make_output: generic provisional class + mandatory review", t_make_output_generic)
    ck.check("build_prompt: readings 注入模板", t_build_prompt)
    ck.check("load_grammar: fault → [0-6]", t_grammar_fault)
    ck.check("load_grammar: generic → [0-3]", t_grammar_generic)


# ══════════════════════════════════════════════════════
#  [4] 入口纯函数
# ══════════════════════════════════════════════════════

def group_entry(ck, tmp):
    _hdr("[4] 入口纯函数 (detect / skip-preprocess)")
    if ER is None:
        ck.check("edge_run 可用", lambda: False)
        return

    hydraulic_csv = tmp / "hydraulic_like.csv"
    hydraulic_csv.write_text("PS1,PS2,PS3,FS1,TS1,VS1\n1,2,3,4,5,6\n", encoding="utf-8")

    generic_csv = tmp / "generic_like.csv"
    generic_csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    std_json = tmp / "std_input.json"
    std_json.write_text(json.dumps(
        [{"task_id": "T-1", "node_id": "edge-01", "data": {"readings": {"x": 1.0}}}]),
        encoding="utf-8")

    ck.check("detect_data_type: 液压", lambda: ER.detect_data_type(str(hydraulic_csv)) == "hydraulic")
    ck.check("detect_data_type: 通用", lambda: ER.detect_data_type(str(generic_csv)) == "generic")
    ck.check("_is_standard_input_json: 标准JSON→True", lambda: ER._is_standard_input_json(str(std_json)) is True)
    ck.check("_is_standard_input_json: CSV→False", lambda: ER._is_standard_input_json(str(generic_csv)) is False)


# ══════════════════════════════════════════════════════
#  [5] 流 / 视频适配
# ══════════════════════════════════════════════════════

def group_stream_video(ck):
    _hdr("[5] 流/视频适配 (纯函数, 无需 cv2)")

    if SA is not None:
        def t_raw():
            s = SA.raw_to_standard_input({"vibration": 5.1, "temperature": 88})
            return (s["task_id"].startswith("T-STREAM")
                    and s["data"]["readings"] == {"vibration": 5.1, "temperature": 88})
        ck.check("stream: raw_to_standard_input 包装", t_raw)
    else:
        ck.check("stream_adapter 可用", lambda: False)

    if VA is not None:
        def t_batch():
            r = VA.batch_features([{"motion_score": 0.1, "brightness_mean": 120.0},
                                   {"motion_score": 0.2, "brightness_mean": 122.0}])
            return abs(r["motion_score"] - 0.15) < 1e-6 and abs(r["brightness_mean"] - 121.0) < 1e-6

        def t_build():
            entry = VA.build_standard_input({"motion_score": 0.15}, {"batch_index": 0, "source": "cam"})
            return (entry["task_type"] == "generic_anomaly"
                    and entry["data"]["generic_mode"] is True
                    and "readings" in entry["data"])
        ck.check("video: batch_features 批平均", t_batch)
        ck.check("video: build_standard_input 标准格式", t_build)
    else:
        ck.check("video_adapter 可用", lambda: False)


# ══════════════════════════════════════════════════════
#  [6] 预处理管线 (子进程实跑)
# ══════════════════════════════════════════════════════

def group_preprocess(ck, tmp):
    _hdr("[6] 预处理管线 (sensor_csv_to_input.py 实跑)")

    csv_path = tmp / "smoke_sensors.csv"
    rows = ["sensorA,sensorB,sensorC"]
    for i in range(12):
        rows.append(f"{1.0 + i * 0.03:.3f},{2.0 + (i % 4) * 0.05:.3f},{3.0 + (i % 3) * 0.07:.3f}")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    out_json = tmp / "smoke_std.json"

    def t_run():
        r = run_py([str(PREPROCESSING_DIR / "sensor_csv_to_input.py"), str(csv_path),
                    "--type", "generic", "--output", str(out_json),
                    "--max-rows", "5000", "--batch", "60", "--task", "generic_anomaly"],
                   timeout=60)
        if r.returncode != 0:
            print("      stderr:", (r.stderr or "")[-200:].replace("\n", " "))
            return False
        return out_json.exists()

    ck.check("sensor_csv_to_input.py 生成标准JSON", t_run)
    if ER is not None and out_json.exists():
        ck.check("产物被 _is_standard_input_json 识别", lambda: ER._is_standard_input_json(str(out_json)) is True)


# ══════════════════════════════════════════════════════
#  [7] 子测试 (子进程)
# ══════════════════════════════════════════════════════

def group_subtests(ck, samples):
    _hdr("[7] 子测试脚本 (退出码校验)")

    def run_sub(rel, extra=None):
        r = run_py([str(TESTS_DIR / rel), *(extra or [])], timeout=120)
        if r.returncode != 0:
            tail = (r.stdout or "").splitlines()[-6:]
            for line in tail:
                print("      |", line)
        return r.returncode == 0

    ck.check("tests/test_conflict_detection.py", lambda: run_sub("test_conflict_detection.py"))
    ck.check("tests/test_offline_resilience.py",
             lambda: run_sub("test_offline_resilience.py", ["--samples", str(samples)]))


# ══════════════════════════════════════════════════════
#  [8] 联机冒烟 (可选)
# ══════════════════════════════════════════════════════

def group_online(ck, endpoint):
    _hdr("[8] 联机冒烟 (可选, 需 llama-server)")
    try:
        urllib.request.urlopen(f"{endpoint}/health", timeout=2)
        alive = True
    except Exception:
        alive = False

    if not alive:
        print(f"    [SKIP] {endpoint} 无 llama-server (模型/server 在另外文件夹)")
        print(f"           启动后重跑可验证真实推理链路")
        return

    def t_infer():
        EI.LLAMA_SERVER = endpoint
        sample = {
            "task_id": "T-SMOKE", "node_id": "edge-zoneA-01",
            "task_type": "fault_diagnosis", "priority": "high",
            "data": {"device_id": "m1", "device_type": "motor",
                     "readings": {"vibration_mm_s": 5.1, "temperature_c": 88}},
        }
        res = EI.infer(sample, use_grammar=True)
        o = res["output"]
        return "decision" in o and "latency_ms" in o and o.get("review_state") == "required"
    ck.check("真实 infer 产出标准 output", t_infer)


# ══════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="边端全链路自检")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080", help="联机冒烟用的 llama-server 地址")
    ap.add_argument("--samples", type=int, default=12, help="断网子测试样本数")
    args = ap.parse_args()

    print("=" * 62)
    print("  边端全链路自检 (除模型/server 外)")
    print(f"  目录: {HHK_DIR}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    ck = Checker()
    tmp = Path(tempfile.mkdtemp(prefix="edge_selfcheck_"))
    try:
        group_imports(ck)
        group_static(ck)
        group_inference(ck)
        group_entry(ck, tmp)
        group_stream_video(ck)
        group_preprocess(ck, tmp)
        group_subtests(ck, args.samples)
        group_online(ck, args.endpoint)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    p, n = ck.summary()
    print("\n" + "=" * 62)
    print("  自检总结")
    print("=" * 62)
    print(f"  通过 {p}/{n} 项")
    failed = [name for name, ok in ck.results if not ok]
    if failed:
        print("  失败项:")
        for name in failed:
            print(f"    - {name}")
    print(f"\n  结果: {'边端链路正常 (全部通过)' if ck.passed() else '存在失败项, 见上'}")
    sys.exit(0 if ck.passed() else 1)


if __name__ == "__main__":
    main()
