"""
特征 CSV → 标准 I/O input.json 桥接器

将 edge_preprocess/ 产出的特征 CSV (feat_0..feat_N) 转换为
edge-io-protocol 标准输入格式, 使 LLM 推理管线可以消费。

策略:
  - 从每个通道的 17 维特征中提取关键统计量 (mean/rms/max/kurtosis) 作为 readings
  - 完整特征向量保留在 data.features 字段, 供云端深度分析
  - meta_json 中的标签信息保留为 ground_truth

用法:
  python features_to_input.py processed/uci_hydraulic_features.csv --schema processed/uci_hydraulic_schema.json
  python features_to_input.py processed/nature_motor_features.csv --schema processed/nature_motor_schema.json
  python features_to_input.py processed/ai4i_milling_features.csv --schema processed/ai4i_milling_schema.json
"""
import sys, os, json, csv, argparse, time
from pathlib import Path

# 每个通道 17 维特征中, 提取哪些作为可读 readings
KEY_FEATURE_INDICES = {
    "mean": 0,
    "rms": 5,
    "max": 3,
    "kurtosis": 6,
}

# 通道名 → 标准传感器字段名映射
CHANNEL_TO_SENSOR = {
    # UCI 液压
    "PS1": "pressure_1_bar", "PS2": "pressure_2_bar",
    "EPS1": "motor_power_w", "FS1": "flow_l_min",
    "TS1": "temperature_c", "VS1": "vibration_mm_s",
    # Nature 电机
    "x": "vibration_x", "y": "vibration_y", "Z": "vibration_z",
    "I1": "current_a", "I2": "current_b", "I3": "current_c",
    "V1": "voltage_a", "V2": "voltage_b", "V3": "voltage_c",
}

# 数据源 → 设备类型
SOURCE_DEVICE_MAP = {
    "uci_hydraulic": ("hydraulic-system-01", "液压系统", "fault_diagnosis"),
    "nature_motor": ("three-phase-motor-01", "三相异步电机", "fault_diagnosis"),
    "ai4i_milling": ("milling-machine-01", "铣削机床", "status_monitoring"),
}


def load_schema(schema_path):
    """加载特征 schema, 获取通道名和特征名"""
    if not schema_path or not os.path.exists(schema_path):
        return None, None
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    channels = schema.get("channels", [])
    feature_names = schema.get("feature_names", [])
    return channels, feature_names


def extract_readings(features, channels, features_per_channel=17):
    """从特征向量中提取关键统计量作为可读 readings"""
    readings = {}

    if not channels:
        # 无 schema 信息, 直接用 feat_N 命名
        for i, v in enumerate(features[:20]):  # 限制前20个
            readings[f"feat_{i}"] = round(v, 4)
        return readings

    for ch_idx, ch_name in enumerate(channels):
        base = ch_idx * features_per_channel
        sensor_name = CHANNEL_TO_SENSOR.get(ch_name, ch_name.lower())

        for feat_name, offset in KEY_FEATURE_INDICES.items():
            idx = base + offset
            if idx < len(features):
                key = f"{sensor_name}_{feat_name}" if len(channels) > 1 else feat_name
                readings[key] = round(features[idx], 4)

    return readings


def convert_csv_to_inputs(csv_path, schema_path=None, output_path=None,
                          node_id="edge-01", max_rows=None):
    """主转换函数"""
    channels, feature_names = load_schema(schema_path)
    features_per_channel = 17 if channels else 1

    # 推断数据源类型
    source_name = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row:
            source_name = first_row.get("source", "unknown")

    device_id, device_type, task_type = SOURCE_DEVICE_MAP.get(
        source_name, ("device-01", source_name or "unknown", "fault_diagnosis")
    )

    print(f"  📂 源: {csv_path}")
    print(f"  🔧 数据源: {source_name} → {device_type}")
    print(f"  📐 通道: {channels or '无schema'}")

    inputs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break

            # 提取特征向量
            features = []
            fi = 0
            while f"feat_{fi}" in row:
                try:
                    features.append(float(row[f"feat_{fi}"]))
                except (ValueError, TypeError):
                    features.append(0.0)
                fi += 1

            if not features:
                continue

            # 提取可读 readings
            readings = extract_readings(features, channels, features_per_channel)

            # 解析 meta
            meta = {}
            meta_raw = row.get("meta_json", "{}")
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                pass

            # 构建标准输入
            entry = {
                "task_id": f"T-FEAT-{source_name or 'X'}-{i:05d}",
                "node_id": node_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "task_type": task_type,
                "priority": "high" if task_type == "fault_diagnosis" else "medium",
                "data": {
                    "device_id": device_id,
                    "device_type": device_type,
                    "readings": readings,
                    "context": f"特征提取窗口 sample={row.get('sample_id', i)}, "
                               f"fs={row.get('fs', '?')}Hz, "
                               f"{len(features)}维特征向量",
                    "features": features,  # 完整特征向量 (云端用)
                    "feature_dim": len(features),
                },
            }

            # 保留标签信息
            if meta:
                entry["data"]["ground_truth"] = meta

            inputs.append(entry)

    # 输出
    if not output_path:
        base = Path(csv_path).stem.replace("_features", "")
        output_path = str(Path(csv_path).parent / f"{base}_standard_inputs.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✅ {len(inputs)} 条标准输入 → {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="特征CSV → 标准I/O桥接器")
    parser.add_argument("csv", help="特征 CSV 文件路径")
    parser.add_argument("--schema", default=None, help="特征 schema JSON 路径")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--node", default="edge-01", help="边端节点ID")
    parser.add_argument("--max-rows", type=int, default=None, help="最大转换行数")
    args = parser.parse_args()

    print("=" * 60)
    print("  🔗 特征CSV → 标准I/O 桥接")
    print("=" * 60)

    convert_csv_to_inputs(args.csv, args.schema, args.output, args.node, args.max_rows)
