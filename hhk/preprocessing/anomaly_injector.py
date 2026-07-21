"""
异常注入器 — 向正常数据中注入已知故障模式

用于: 生成带标签的测试数据, 验证边端模型能否正确检测异常

用法:
  python anomaly_injector.py <标准input.json> --mode bearing_wear|current_imbalance|overheat [--output result.json]
"""
import sys, os, json, random, math, argparse, copy
from pathlib import Path

# ── 故障模式定义 ──
# 每种故障会在哪些 readings 上产生什么变化
FAULT_MODES = {
    "bearing_wear": {
        "name": "轴承磨损",
        "description": "渐进性磨损, 振动增大, 温度上升",
        "modifications": [
            {"field": "vibration_mm_s", "operation": "multiply", "value": (2.0, 4.0), "onset_delay_pct": 0.3},
            {"field": "temperature_c", "operation": "add", "value": (15, 30), "onset_delay_pct": 0.4},
        ],
        "expected_action": "alert",
        "expected_level": "warning",
    },
    "bearing_failure": {
        "name": "轴承严重失效",
        "description": "轴承接近完全失效, 振动剧烈, 温度飙升",
        "modifications": [
            {"field": "vibration_mm_s", "operation": "multiply", "value": (5.0, 10.0), "onset_delay_pct": 0.2},
            {"field": "temperature_c", "operation": "add", "value": (40, 60), "onset_delay_pct": 0.2},
        ],
        "expected_action": "shutdown",
        "expected_level": "critical",
    },
    "current_imbalance": {
        "name": "三相电流不平衡",
        "description": "一相电流偏离, 其余两相正常",
        "modifications": [
            {"field": "current_a", "operation": "multiply", "value": (1.3, 1.6), "onset_delay_pct": 1.0},  # 立即生效
            {"field": "current_b", "operation": "multiply", "value": (0.85, 0.95), "onset_delay_pct": 1.0},
        ],
        "expected_action": "alert",
        "expected_level": "warning",
    },
    "overheat": {
        "name": "过热",
        "description": "冷却系统故障导致温度持续上升",
        "modifications": [
            {"field": "temperature_c", "operation": "add", "value": (25, 50), "onset_delay_pct": 0.5},
            {"field": "flow_l_min", "operation": "multiply", "value": (0.3, 0.6), "onset_delay_pct": 0.6},
        ],
        "expected_action": "alert",
        "expected_level": "critical",
    },
    "flow_blockage": {
        "name": "管路堵塞",
        "description": "管路部分堵塞导致流量下降, 压力上升",
        "modifications": [
            {"field": "flow_l_min", "operation": "multiply", "value": (0.3, 0.5), "onset_delay_pct": 0.15},
            {"field": "ps1_bar", "operation": "multiply", "value": (1.3, 1.6), "onset_delay_pct": 0.15},
        ],
        "expected_action": "maintain",
        "expected_level": "warning",
    },
    "sensor_drift": {
        "name": "传感器漂移",
        "description": "传感器读数缓慢偏离真实值(软故障, 难以检测)",
        "modifications": [
            {"field": "vibration_mm_s", "operation": "drift", "drift_rate": (0.003, 0.008), "onset_delay_pct": 0.1},
        ],
        "expected_action": "warning",
        "expected_level": "info",
    },
}


def apply_modification(value, operation, params, position_pct):
    """对单个值施加修改"""
    if "onset_delay_pct" in params and position_pct < params["onset_delay_pct"]:
        return value  # 还没到故障发作点

    if operation == "multiply":
        factor = random.uniform(*params["value"]) if isinstance(params["value"], tuple) else params["value"]
        return value * factor
    elif operation == "add":
        delta = random.uniform(*params["value"]) if isinstance(params["value"], tuple) else params["value"]
        return value + delta
    elif operation == "drift":
        drift = random.uniform(*params["drift_rate"])
        return value + drift * position_pct * 100
    return value


def inject_anomalies(inputs, fault_mode, inject_ratio=0.3):
    """
    向标准输入数据中注入异常。

    Args:
        inputs: 标准 input.json 列表
        fault_mode: 故障模式名
        inject_ratio: 注入比例 (0.0-1.0), 多少比例的数据变成异常

    Returns:
        (annotated_inputs[], labels[])
    """
    if fault_mode not in FAULT_MODES:
        print(f"❌ 未知故障模式: {fault_mode}")
        print(f"   可选: {list(FAULT_MODES.keys())}")
        return [], []

    fault = FAULT_MODES[fault_mode]
    total = len(inputs)
    anomaly_count = int(total * inject_ratio)
    anomaly_indices = set(random.sample(range(total), max(1, anomaly_count)))

    annotated = []
    labels = []

    for i, entry in enumerate(inputs):
        new_entry = copy.deepcopy(entry)
        is_anomaly = i in anomaly_indices

        if is_anomaly:
            position_pct = (i % 100) / 100.0  # 故障在批次中的位置
            for mod in fault["modifications"]:
                field = mod["field"]
                if field in new_entry["data"]["readings"]:
                    new_entry["data"]["readings"][field] = apply_modification(
                        new_entry["data"]["readings"][field],
                        mod["operation"],
                        mod,
                        position_pct
                    )

            # 添加标签
            label = {
                "is_anomaly": True,
                "fault_type": fault_mode,
                "fault_name": fault["name"],
                "expected_action": fault["expected_action"],
                "expected_level": fault["expected_level"],
            }
        else:
            label = {"is_anomaly": False, "fault_type": "normal"}

        annotated.append(new_entry)
        labels.append(label)

    return annotated, labels


def main():
    parser = argparse.ArgumentParser(description="异常注入器")
    parser.add_argument("input", help="标准 input.json 文件")
    parser.add_argument("--mode", choices=list(FAULT_MODES.keys()), required=True, help="故障模式")
    parser.add_argument("--ratio", type=float, default=0.3, help="注入比例 (默认0.3)")
    parser.add_argument("--output", default=None, help="输出文件")
    parser.add_argument("--label-output", default=None, help="标签文件")
    parser.add_argument("--list", action="store_true", help="列出所有故障模式")

    args = parser.parse_args()

    if args.list:
        print("\n可用故障模式:\n")
        for name, fault in FAULT_MODES.items():
            print(f"  {name:<25} {fault['name']:<15} {fault['description']}")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    print(f"📂 加载 {len(inputs)} 条标准输入")
    print(f"🦠 注入故障: {FAULT_MODES[args.mode]['name']} (比例 {args.ratio})")

    annotated, labels = inject_anomalies(inputs, args.mode, inject_ratio=args.ratio)

    anomaly_count = sum(1 for l in labels if l["is_anomaly"])
    print(f"   异常条数: {anomaly_count}/{len(annotated)}")

    out_file = args.output or f"injected_{args.mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)
    print(f"✅ 数据 → {out_file}")

    label_file = args.label_output or f"labels_{args.mode}.json"
    with open(label_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"✅ 标签 → {label_file}")


if __name__ == "__main__":
    main()
