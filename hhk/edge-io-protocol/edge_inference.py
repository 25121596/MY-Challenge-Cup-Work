"""
边端推理框架 v2.0 — 短决策输出 + GBNF grammar
输出: {"c":<code>,"s":<score>,"r":<0|1>}  3字段, ~10 tokens
"""
import json, hashlib, time, os, urllib.request

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
LLAMA_SERVER = "http://localhost:8080"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
GRAMMAR_DIR = os.path.join(BASE_DIR, "grammars")

with open(os.path.join(BASE_DIR, "task_types.json"), "r", encoding="utf-8") as f:
    TASK_CONFIG = json.load(f)

# 故障类别码 → 名称
CATEGORY_NAMES = {
    0: "normal",
    1: "bearing_wear",
    2: "bearing_failure",
    3: "current_imbalance",
    4: "overheat",
    5: "flow_blockage",
    6: "sensor_drift",
}

# 故障类别码 → 业务决策 (action + level + 一句话摘要)
# 对齐 schemas/output.json 的 decision.action / decision.level
CATEGORY_ACTION = {
    0: {"action": "normal",   "level": "info",     "summary": "设备运行正常, 无需处理"},
    1: {"action": "warning",  "level": "warning",  "summary": "疑似轴承磨损, 建议加强监测"},
    2: {"action": "shutdown", "level": "critical", "summary": "轴承严重失效, 建议立即停机检修"},
    3: {"action": "alert",    "level": "warning",  "summary": "三相电流不平衡, 需检查电气连接"},
    4: {"action": "alert",    "level": "critical", "summary": "设备过热, 需检查冷却系统"},
    5: {"action": "maintain", "level": "warning",  "summary": "管路堵塞嫌疑, 建议维护"},
    6: {"action": "warning",  "level": "info",     "summary": "传感器疑似漂移, 建议校准"},
}

# 甩云原因枚举
ESCALATE_REASON = {
    "LOW_CONFIDENCE": "置信度不足",
    "FORMAT_ERROR": "输出格式异常",
    "UNKNOWN_CLASS": "未知类别",
    "HIGH_RISK": "高风险故障需云端确认",
    "TIMEOUT": "推理超时",
}


def load_prompt(task_type):
    """加载对应任务的 prompt 模板"""
    cfg = TASK_CONFIG["task_types"].get(task_type)
    if not cfg or not cfg.get("prompt_template"):
        return None
    path = os.path.join(BASE_DIR, cfg["prompt_template"])
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_grammar(task_type="fault_diagnosis"):
    """加载 GBNF grammar 文件"""
    # 将 task_type 映射到 grammar 文件
    grammar_map = {
        "fault_diagnosis": "short_decision.gbnf",
        "quality_inspection": "short_decision.gbnf",
        "safety_check": "short_decision.gbnf",
        "status_monitoring": "short_decision.gbnf",
    }
    gname = grammar_map.get(task_type, "short_decision.gbnf")
    gpath = os.path.join(GRAMMAR_DIR, gname)
    if os.path.exists(gpath):
        with open(gpath, "r", encoding="utf-8") as f:
            return f.read()
    return None


def build_prompt(task_type, input_data):
    """把输入数据填入 prompt 模板 (排除大数组字段避免超上下文)"""
    template = load_prompt(task_type)
    if not template:
        # 无模板时用 readings 的紧凑表示
        readings = input_data.get("data", {}).get("readings", {})
        return " ".join(f"{k}={v}" for k, v in readings.items())

    # 构建 prompt 数据: 排除 features 大数组, 只保留 readings 等可读字段
    data_for_prompt = dict(input_data.get("data", input_data))
    data_for_prompt.pop("features", None)  # 移除大特征向量
    data_for_prompt.pop("feature_dim", None)

    # 优先用 readings 的紧凑格式 (更短, 推理更快)
    readings = data_for_prompt.get("readings", {})
    if readings:
        data_text = " ".join(f"{k}={v}" for k, v in readings.items())
    else:
        data_text = json.dumps(data_for_prompt, ensure_ascii=False)

    if "{input_data}" in template:
        return template.replace("{input_data}", data_text)
    elif "{inspection_data}" in template:
        return template.replace("{inspection_data}", data_text)
    elif "{scene_description}" in template:
        scene = input_data.get("data", {}).get("description", data_text)
        return template.replace("{scene_description}", scene)
    elif "{historical_data}" in template:
        return template.replace("{historical_data}", data_text)
    else:
        return template + "\n" + data_text


def call_model(prompt, n_predict=32, temperature=0.0, grammar=None):
    """调用 llama-server 推理 (short decision 只需 ~10 tokens)"""
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stream": False,
    }
    if grammar:
        payload["grammar"] = grammar

    req_data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{LLAMA_SERVER}/completion", data=req_data,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read())
    return result.get("content", ""), {
        "tokens": result.get("tokens_predicted", 0),
        "tokens_per_second": round(result.get("timings", {}).get("predicted_per_second", 0), 1),
        "prompt_ms": round(result.get("timings", {}).get("prompt_ms", 0), 1),
    }


def parse_short_decision(raw_text):
    """解析短决策输出。

    兼容三种格式:
      1. GBNF 单数字: "1" → category=1
      2. class,risk: "1,2" → category=1, risk=2
      3. JSON: {"c":1,"s":80,"r":0} → 完整解析

    Returns:
        (category, confidence, request_cloud, parse_error_reason_or_None)
    """
    import re
    text = raw_text.strip()

    # 去除可能的 markdown fence
    if text.startswith("```"):
        text = text.replace("```", "").strip()

    # ── 格式 1: class,risk (e.g. "1,2") ──
    m = re.search(r'(\d)\s*,\s*(\d)', text)
    if m:
        cls = int(m.group(1))
        risk = int(m.group(2))
        if cls <= 6:
            # risk 0-3 映射到 confidence: 越高越不确定
            confidence = max(0, 100 - risk * 25)
            request_cloud = 1 if risk >= 2 else 0
            return cls, confidence, request_cloud, None

    # ── 格式 2: 单数字 (GBNF grammar 输出) ──
    m = re.search(r'\b([0-6])\b', text)
    if m:
        cls = int(m.group(1))
        # 单数字模式: 默认高置信度, 不甩云
        return cls, 85, 0, None

    # ── 格式 3: JSON {"c":N,"s":N,"r":0|1} ──
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        json_text = text[brace_start:brace_end + 1]
        try:
            obj = json.loads(json_text)
            c = obj.get("c")
            s = obj.get("s", 80)
            r = obj.get("r", 0)
            if c is not None:
                return int(c), int(s), int(r), None
        except json.JSONDecodeError:
            m2 = re.search(r'"c"\s*:\s*(\d+)\s*,\s*"s"\s*:\s*(\d+)\s*,\s*"r"\s*:\s*([01])', json_text)
            if m2:
                return int(m2.group(1)), int(m2.group(2)), int(m2.group(3)), None

    return None, 0, 1, "FORMAT_ERROR"


def make_output(task_id, node_id, category, confidence, request_cloud,
                latency_ms, parse_error=None):
    """组装标准输出 JSON (v2 短决策版, 对齐 schemas/output.json)"""
    category_name = CATEGORY_NAMES.get(category, f"unknown_{category}")
    biz = CATEGORY_ACTION.get(category, {"action": "warning", "level": "warning",
                                          "summary": category_name})

    # 甩云判定
    if parse_error:
        escalate = True
        escalate_reason = parse_error
    elif request_cloud:
        escalate = True
        escalate_reason = "LOW_CONFIDENCE" if confidence < 60 else "HIGH_RISK"
    elif category is None:
        escalate = True
        escalate_reason = "UNKNOWN_CLASS"
    elif category >= 2 and confidence < 70:
        # 严重故障 (bearing_failure=2, overheat=4 等) 且置信度低 → 甩云确认
        escalate = True
        escalate_reason = "HIGH_RISK"
    else:
        escalate = False
        escalate_reason = None

    output = {
        "task_id": task_id,
        "node_id": node_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}",
        "latency_ms": round(latency_ms, 1),
        "decision": {
            # 业务面 (对齐 schemas/output.json)
            "action": biz.get("action", "warning"),
            "level": biz.get("level", "warning"),
            "summary": biz.get("summary", category_name),
            # 故障分类面 (短决策原始结果)
            "category": category,
            "category_name": category_name,
            "request_cloud": bool(request_cloud),
        },
        # 顶层 confidence (0-1 浮点, 对齐 schema)
        "confidence": round(confidence / 100.0, 2) if confidence else 0.0,
        "escalate_to_cloud": escalate,
    }

    if escalate:
        output["escalate_reason"] = ESCALATE_REASON.get(escalate_reason, escalate_reason or "UNKNOWN")

    # 一致性 hash
    output["consistency"] = {
        "decision_hash": hashlib.sha256(
            json.dumps({"c": category, "r": request_cloud}, sort_keys=True).encode()
        ).hexdigest()[:16],
        "zone": node_id.rsplit("-", 1)[0] if "-" in node_id else "default",
    }

    return output


def infer(task_input, use_grammar=True):
    """完整的一次边端推理 (v2 短决策版)。

    task_input: 符合 schemas/input.json 格式的 dict
    返回: 符合 schemas/output.json 格式的 dict
    """
    task_type = task_input.get("task_type", "fault_diagnosis")
    task_id = task_input.get("task_id", "auto")
    node_id = task_input.get("node_id", "edge-01")

    # 1. 构建 prompt
    prompt = build_prompt(task_type, task_input)

    # 2. 加载 grammar
    grammar = load_grammar(task_type) if use_grammar else None

    # 3. 推理 (GBNF grammar 约束输出为单数字, n_predict=2 即可)
    t0 = time.time()
    raw, timing = call_model(prompt, n_predict=2, temperature=0.0, grammar=grammar)
    elapsed = (time.time() - t0) * 1000

    # 4. 解析短决策
    category, confidence, request_cloud, parse_error = parse_short_decision(raw)

    # 5. 组装标准输出
    output = make_output(task_id, node_id, category, confidence, request_cloud,
                         elapsed, parse_error)

    return {
        "output": output,
        "raw_model_response": raw.strip(),
        "timing": timing,
        "prompt_length": len(prompt),
        "grammar_used": grammar is not None,
        "parse_error": parse_error,
    }


# ══════════════════════════════════════════════════════
#  示例
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    sample_input = {
        "task_id": "T-001",
        "node_id": "edge-workshop3",
        "timestamp": "2026-07-15T14:30:00",
        "task_type": "fault_diagnosis",
        "priority": "high",
        "data": {
            "device_id": "motor-a01",
            "device_type": "三相异步电机",
            "readings": {
                "vibration_mm_s": 5.1,
                "temperature_c": 85,
                "current_a": 12.3,
                "current_b": 13.1,
                "current_c": 12.5
            },
            "context": "连续运行3个月未维护，额定功率7.5kW"
        }
    }

    print("=" * 60)
    print("  边端推理 v2 — 短决策 + GBNF Grammar")
    print("=" * 60)

    result = infer(sample_input, use_grammar=True)

    print(f"\n📥 输入: {json.dumps(sample_input['data']['readings'], ensure_ascii=False)}")
    print(f"\n📤 原始输出: {result['raw_model_response']}")
    print(f"\n📤 解析结果: c={result['output']['decision']['category']} "
          f"({result['output']['decision']['category_name']}) "
          f"s={result['output']['decision']['confidence']} "
          f"r={result['output']['decision']['request_cloud']}")
    print(f"\n⚡ 性能: {result['timing']['tokens']} tokens @ "
          f"{result['timing']['tokens_per_second']} t/s | "
          f"延迟 {result['output']['latency_ms']}ms "
          f"(prompt处理 {result['timing']['prompt_ms']}ms)")
    print(f"🔒 Grammar: {'启用' if result['grammar_used'] else '未用'}")
    if result["parse_error"]:
        print(f"⚠️ 解析问题: {result['parse_error']}")
    print(f"🌩️ 甩云: {'是 ← ' + result['output'].get('escalate_reason','') if result['output']['escalate_to_cloud'] else '否'}")
