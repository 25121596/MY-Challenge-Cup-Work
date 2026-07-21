"""
通用传感器数据 → 标准 I/O input.json 转换器

将各种格式的工业传感器数据(CSV/TSV/TXT)转换为 edge-io-protocol 标准输入格式。

用法:
  python sensor_csv_to_input.py <数据文件> --type hydraulic|motor|generic [--output out.json]
  python sensor_csv_to_input.py batch <目录> --type hydraulic             # 批量转换
"""
import sys, os, json, csv, argparse, time, hashlib
from pathlib import Path

# ── 传感器通道映射 ──
# 不同数据集的列名映射到标准 reading 字段
CHANNEL_MAPS = {
    "hydraulic": {
        "ps1_bar": ["PS1", "ps1", "pressure_1", "Pressure1"],
        "ps2_bar": ["PS2", "ps2", "pressure_2", "Pressure2"],
        "ps3_bar": ["PS3", "ps3", "pressure_3", "Pressure3"],
        "flow_l_min": ["FS1", "fs1", "flow_1", "Flow1"],
        "temperature_c": ["TS1", "ts1", "temp_1", "Temperature1"],
        "vibration_mm_s": ["VS1", "vs1", "vib_1", "Vibration1"],
    },
    "motor": {
        "vibration_mm_s": ["vibration", "vib", "Vibration", "acceleration"],
        "temperature_c": ["temperature", "temp", "Temperature", "motor_temp"],
        "current_a": ["current", "Current", "motor_current", "phase_current"],
        "voltage_v": ["voltage", "Voltage", "motor_voltage"],
        "speed_rpm": ["speed", "rpm", "RPM", "motor_speed"],
    },
    "milling": {
        "vibration_mm_s": ["vibration", "vib"],
        "temperature_c": ["temperature", "temp", "Air temperature [K]", "Process temperature [K]"],
        "tool_wear_um": ["tool_wear", "wear", "Tool wear [min]"],
        "cutting_force_n": ["force", "cutting_force", "Torque [Nm]"],
        "speed_rpm": ["speed", "rpm", "Rotational speed [rpm]"],
    },
}


def detect_delimiter(filepath):
    """自动检测分隔符"""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    if ";" in first_line:
        return ";"
    return None


def map_headers(headers, channel_map):
    """将数据集列名映射到标准 reading 字段名"""
    mapping = {}
    for standard_name, aliases in channel_map.items():
        for alias in aliases:
            for h in headers:
                if alias.lower() == h.lower().strip():
                    mapping[h] = standard_name
                    break
            if any(h in mapping.values() for h in [standard_name]):
                break
    return mapping


def read_sensor_file(filepath, max_rows=None):
    """读取传感器文件，返回 headers 和 rows"""
    delimiter = detect_delimiter(filepath)
    if delimiter is None:
        print(f"⚠️ 无法检测分隔符: {filepath}, 尝试空格")
        delimiter = " "

    rows = []
    headers = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i == 0:
                headers = [h.strip() for h in row if h.strip()]
                continue
            if max_rows and i > max_rows:
                break
            # 逐值转换: 数字转 float, 非数字置为 None (保留行不跳过)
            values = []
            has_numeric = False
            for v in row:
                v = v.strip()
                if not v:
                    continue
                try:
                    values.append(float(v))
                    has_numeric = True
                except ValueError:
                    values.append(None)
            if has_numeric:
                rows.append(values)
    return headers, rows


def to_standard_input(headers, rows, channel_map, device_type, device_id, node_id, task_type, batch_size=60):
    """将原始数据转为标准 input.json 格式的列表"""
    header_map = map_headers(headers, channel_map)

    if not header_map:
        print(f"⚠️ 无法匹配传感器通道! 数据集列: {headers}")
        print(f"   期望的别名: {[v for aliases in channel_map.values() for v in aliases]}")
        return []

    inputs = []
    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        if len(batch) < 5:
            break

        # 取这个批次的平均值作为"当前读数"
        readings = {}
        for col_idx, (col_name, std_name) in enumerate(
            [(h, header_map.get(h)) for h in headers if h in header_map]
        ):
            if col_idx >= len(batch[0]):
                continue
            values = [r[col_idx] for r in batch if col_idx < len(r) and r[col_idx] is not None]
            if values:
                readings[std_name] = round(sum(values) / len(values), 4)

        if not readings:
            continue

        input_entry = {
            "task_id": f"T-{int(time.time()*1000)}-{batch_start:04d}",
            "node_id": node_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_type": task_type,
            "priority": "high",
            "data": {
                "device_id": device_id,
                "device_type": device_type,
                "readings": readings,
                "context": f"批次 {batch_start}-{batch_start+batch_size}, 共{len(batch)}条记录",
                "raw_stats": {
                    "batch_size": len(batch),
                    "sample_period_ms": 10,  # 假设10ms采样
                }
            }
        }
        inputs.append(input_entry)
    return inputs


# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="传感器数据 → 标准输入 JSON")
    parser.add_argument("input", help="输入文件或目录(batch模式)")
    parser.add_argument("--type", choices=["hydraulic", "motor", "milling", "generic"], default="generic")
    parser.add_argument("--device", default="sensor-unit-01", help="设备ID")
    parser.add_argument("--node", default="edge-workshop3", help="边端节点ID")
    parser.add_argument("--task", default="fault_diagnosis", help="任务类型")
    parser.add_argument("--output", default=None, help="输出JSON文件")
    parser.add_argument("--max-rows", type=int, default=5000, help="最大读取行数")
    parser.add_argument("--batch", type=int, default=60, help="每批行数")
    parser.add_argument("--batch-mode", action="store_true", help="批量处理目录")

    args = parser.parse_args()

    channel_map = CHANNEL_MAPS.get(args.type, {})

    if args.batch_mode or os.path.isdir(args.input):
        # 批量模式
        import glob as g
        files = sorted(g.glob(os.path.join(args.input, "*")))
        all_inputs = []
        for f in files:
            if os.path.isfile(f) and not f.endswith(".json"):
                headers, rows = read_sensor_file(f, max_rows=args.max_rows)
                inputs = to_standard_input(headers, rows, channel_map,
                                           args.type, args.device, args.node, args.task, args.batch)
                all_inputs.extend(inputs)
                print(f"  {os.path.basename(f)}: {len(inputs)} 条标准输入")
        output_path = args.output or f"processed_{args.type}_inputs.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_inputs, f, ensure_ascii=False, indent=2)
        print(f"✅ 批量完成: {len(all_inputs)} 条 → {output_path}")

    else:
        # 单文件模式
        headers, rows = read_sensor_file(args.input, max_rows=args.max_rows)
        print(f"📂 {args.input}: {len(headers)} 列, {len(rows)} 行")
        print(f"   列名: {headers[:10]}...")

        inputs = to_standard_input(headers, rows, channel_map,
                                   args.type, args.device, args.node, args.task, args.batch)

        output_path = args.output or f"{Path(args.input).stem}_standard.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(inputs, f, ensure_ascii=False, indent=2)
        print(f"✅ {len(inputs)} 条标准输入 → {output_path}")


if __name__ == "__main__":
    main()
