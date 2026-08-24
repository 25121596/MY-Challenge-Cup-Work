"""
边端推理框架 v3 — 单类别观察 + GBNF grammar
输出: 一个未经校准的类别码；它始终需要复核，不能表示概率或安全等级。
"""
import json, hashlib, time, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from decision_contract import compact_cloud_payload, qwen_fallback_fields, utc_now
from ai4i_qwen_encoding import encode_ai4i_qwen_review, is_ai4i_online_task

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
LLAMA_SERVER = "http://localhost:8080"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")
GRAMMAR_DIR = os.path.join(BASE_DIR, "grammars")

with open(os.path.join(BASE_DIR, "task_types.json"), "r", encoding="utf-8") as f:
    TASK_CONFIG = json.load(f)

# 故障类别码 → 名称 (已知工业场景 7 类)
CATEGORY_NAMES = {
    0: "normal",
    1: "bearing_wear",
    2: "bearing_failure",
    3: "current_imbalance",
    4: "overheat",
    5: "flow_blockage",
    6: "sensor_drift",
}

# ── 泛化模式: 4 级通用异常等级 (不假设传感器类型/故障模式) ──
GENERIC_SEVERITY_NAMES = {
    0: "normal",
    1: "mild_anomaly",
    2: "moderate_anomaly",
    3: "severe_anomaly",
}

def load_prompt(task_type):
    """加载对应任务的 prompt 模板"""
    cfg = TASK_CONFIG["task_types"].get(task_type)
    if not cfg or not cfg.get("prompt_template"):
        # 泛化/视频模式 fallback
        fallback_map = {
            "generic_anomaly": "prompts/generic_anomaly.txt",
            "video_anomaly": "prompts/video_anomaly.txt",
        }
        fallback = fallback_map.get(task_type, "prompts/generic_anomaly.txt")
        generic_path = os.path.join(BASE_DIR, fallback)
        if os.path.exists(generic_path):
            with open(generic_path, "r", encoding="utf-8") as f:
                return f.read()
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
        "ai4i_qwen_review": "ai4i_review.gbnf",
        "fault_diagnosis": "short_decision.gbnf",
        "quality_inspection": "short_decision.gbnf",
        "safety_check": "short_decision.gbnf",
        "status_monitoring": "short_decision.gbnf",
        "generic_anomaly": "generic_decision.gbnf",
        "video_anomaly": "generic_decision.gbnf",
    }
    gname = grammar_map.get(task_type, "generic_decision.gbnf")  # 默认泛化 grammar
    gpath = os.path.join(GRAMMAR_DIR, gname)
    if os.path.exists(gpath):
        with open(gpath, "r", encoding="utf-8") as f:
            return f.read()
    return None


def build_prompt(task_type, input_data):
    """把输入数据填入 prompt 模板 (排除大数组字段避免超上下文)"""
    if task_type == "ai4i_qwen_review":
        return encode_ai4i_qwen_review(input_data)

    template = load_prompt(task_type)
    if not template:
        # 无模板时用 readings 的紧凑表示
        readings = input_data.get("data", {}).get("readings", {})
        return " ".join(f"{k}={v}" for k, v in readings.items())

    # 构建 prompt 数据: 排除 features 大数组, 只保留 readings 等可读字段
    data_for_prompt = dict(input_data.get("data", input_data))
    data_for_prompt.pop("features", None)  # 移除大特征向量
    data_for_prompt.pop("feature_dim", None)

    # ── 泛化模式: use context for statistical baseline and deviation analysis ──
    context = data_for_prompt.pop("context", "")
    generic_mode = data_for_prompt.pop("generic_mode", False)

    # 优先用 readings 的紧凑格式 (更短, 推理更快)
    readings = data_for_prompt.get("readings", {})
    if readings:
        data_text = " ".join(f"{k}={v}" for k, v in readings.items())
        # 泛化模式: 附上统计基线上下文
        if generic_mode and context:
            data_text = f"{context}\n\n当前读数: {data_text}"
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


def parse_short_decision(raw_text, generic_mode=False, ai4i_review=False):
    """Parse an uncalibrated Qwen class observation.

    The parser deliberately returns no score.  A generated class token is
    never a probability, confidence value, safety level, or cloud-routing
    authorization.

    Returns ``(category, parse_error_reason_or_None)``.
    """
    import re
    text = raw_text.strip()
    max_cat = 1 if ai4i_review else (3 if generic_mode else 6)

    # 去除可能的 markdown fence
    if text.startswith("```"):
        text = text.replace("```", "").strip()

    # Compatibility input accepts exactly {"c": <class>}.  Do not silently
    # discard legacy score/routing fields: accepting them would make it too
    # easy to reintroduce a pseudo-confidence into the live protocol.
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        json_text = text[brace_start:brace_end + 1]
        try:
            obj = json.loads(json_text)
            if set(obj) != {"c"}:
                return None, "FORMAT_ERROR"
            c = obj.get("c")
            if c is not None:
                c = int(c)
                if c > max_cat:
                    return max_cat, "UNKNOWN_CLASS"
                return c, None
        except json.JSONDecodeError:
            return None, "FORMAT_ERROR"

    # Production grammar output is exactly one class digit.
    m = re.fullmatch(r"\s*(\d)\s*", text)
    if m:
        cls = int(m.group(1))
        if cls > max_cat:
            return max_cat, "UNKNOWN_CLASS"
        return cls, None

    return None, "FORMAT_ERROR"


def make_output(task_id, node_id, category, latency_ms, parse_error=None,
                generic_mode=False, task_input=None, ai4i_review=False):
    """组装标准输出 JSON (v2 短决策版, 对齐 schemas/output.json)

    generic_mode=True: 使用 4 级通用异常等级
    generic_mode=False: 使用 7 类已知故障分类
    """
    cat_names = ({0: "no_failure_pattern", 1: "possible_failure_pattern"}
                 if ai4i_review else (GENERIC_SEVERITY_NAMES if generic_mode else CATEGORY_NAMES))

    category_name = cat_names.get(category, f"unknown_{category}")

    output = {
        "task_id": task_id,
        "node_id": node_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}",
        "latency_ms": round(latency_ms, 1),
        "decision": {
            "action": "continue_with_review",
            "level": "warning",
            "summary": "Uncalibrated Qwen fallback requires cloud review",
            "provisional_category": {
                "code": category,
                "name": category_name,
            },
        },
        "generic_mode": generic_mode,
    }

    # 一致性 hash
    output["consistency"] = {
        "decision_hash": hashlib.sha256(
            json.dumps({"c": category, "parse_error": parse_error}, sort_keys=True).encode()
        ).hexdigest()[:16],
        "zone": node_id.rsplit("-", 1)[0] if "-" in node_id else "default",
    }

    # Qwen has no retained-set calibration in this path.  It must always be
    # reviewed; only a hard safety trigger can promote the action to P0.
    output.update(qwen_fallback_fields(task_input or {}))
    output["decision"]["action"] = output["action"]
    if parse_error:
        output["review_reason"].append("qwen_format_error")
    if output["action"] == "protect":
        output["decision"].update({"action": "protect", "level": "critical",
                                   "summary": "Hard safety trigger; protect immediately and review"})
    return output


def infer(task_input, use_grammar=True, use_ai4i_qwen_review=False):
    """完整的一次边端推理 (v2 短决策版)。

    task_input: 符合 schemas/input.json 格式的 dict
    返回: 符合 schemas/output.json 格式的 dict

    自动检测泛化模式: data.generic_mode==True → 4级异常等级
    """
    task_type = task_input.get("task_type", "fault_diagnosis")
    task_id = task_input.get("task_id", "auto")
    node_id = task_input.get("node_id", "edge-01")
    generic_mode = task_input.get("data", {}).get("generic_mode", False)
    ai4i_review = use_ai4i_qwen_review
    if ai4i_review and not is_ai4i_online_task(task_input):
        raise ValueError("AI4I Qwen review requires the frozen AI4I online contract")

    # 泛化模式: 统一使用 generic_anomaly 任务类型
    if ai4i_review:
        task_type = "ai4i_qwen_review"
    elif generic_mode:
        task_type = "generic_anomaly"

    # 1. 构建 prompt
    prompt = build_prompt(task_type, task_input)

    # 2. 加载 grammar
    grammar = load_grammar(task_type) if use_grammar else None

    # 3. 推理 (GBNF grammar 约束输出为单数字, n_predict=2 即可)
    received_at = utc_now()
    t0 = time.time()
    # The grammar accepts exactly one class digit.  Requesting a second token
    # only adds decode latency and is not part of the output contract.
    raw, timing = call_model(prompt, n_predict=1, temperature=0.0, grammar=grammar)
    elapsed = (time.time() - t0) * 1000

    # 4. 解析短决策
    category, parse_error = parse_short_decision(raw, generic_mode=generic_mode, ai4i_review=ai4i_review)

    # 5. 组装标准输出
    output = make_output(task_id, node_id, category, elapsed, parse_error,
                         generic_mode=generic_mode, task_input=task_input, ai4i_review=ai4i_review)
    output.pop("timestamp", None)
    output["observed_at"] = task_input.get("observed_at", task_input.get("timestamp"))
    output["received_at"] = received_at
    output["decided_at"] = utc_now()
    output["cloud_payload"] = compact_cloud_payload(task_input, output)

    return {
        "output": output,
        "raw_model_response": raw.strip(),
        "timing": timing,
        "prompt_length": len(prompt),
        "grammar_used": grammar is not None,
        "parse_error": parse_error,
        "generic_mode": generic_mode,
        "ai4i_qwen_review": ai4i_review,
    }


def infer_ai4i_qwen_review(task_input, use_grammar=True):
    """Run the optional Qwen review expert on a frozen AI4I online record."""
    return infer(task_input, use_grammar=use_grammar, use_ai4i_qwen_review=True)


# ══════════════════════════════════════════════════════
#  示例
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    sample_input = {
        "task_id": "T-001",
        "node_id": "edge-workshop3",
        "timestamp": "2026-07-15T14:30:00",
        "task_type": "fault_diagnosis",
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
    provisional = result["output"]["decision"]["provisional_category"]
    print(f"\n📤 解析结果: c={provisional['code']} ({provisional['name']})")
    print(f"\n⚡ 性能: {result['timing']['tokens']} tokens @ "
          f"{result['timing']['tokens_per_second']} t/s | "
          f"延迟 {result['output']['latency_ms']}ms "
          f"(prompt处理 {result['timing']['prompt_ms']}ms)")
    print(f"🔒 Grammar: {'启用' if result['grammar_used'] else '未用'}")
    if result["parse_error"]:
        print(f"⚠️ 解析问题: {result['parse_error']}")
    print(f"🌩️ 复核: {result['output']['review_state']} ({','.join(result['output']['review_reason'])})")
