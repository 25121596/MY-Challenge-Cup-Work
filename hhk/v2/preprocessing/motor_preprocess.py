r"""
Nature 三相电机数据集 — 专用预处理

原始数据: data/Nature_3phase_motor/27216219/
  FILE 1.csv ~ FILE 10.csv  — 10个工况文件, 各100万行
    x, y, Z           — 三轴振动 (加速度)
    I1, I2, I3        — 三相电流
    V1, V2, V3        — 三相电压
  LABEL DATASET.csv   — 标签 (单行, ~10000个值, 每个覆盖1000行)

输出: processed_motor_inputs.json (标准 I/O 格式)

用法:
  python motor_preprocess.py
  python motor_preprocess.py --data-dir "D:/.../27216219" --output motor_inputs.json
"""
import sys, os, json, csv, argparse, time as tm
from pathlib import Path
from sensor_csv_to_input import read_sensor_file


# 数据集中关键列在标准 I/O 中的映射
SENSOR_MAP = {
    "x": "vibration_x_mm_s",
    "y": "vibration_y_mm_s",
    "Z": "vibration_z_mm_s",
    "I1": "current_a",
    "I2": "current_b",
    "I3": "current_c",
    "V1": "voltage_a",
    "V2": "voltage_b",
    "V3": "voltage_c",
}


def load_labels(label_path):
    """加载标签文件。单行CSV, 每个值对应一个数据段的故障类别。"""
    with open(label_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().strip()
    labels = [int(x.strip()) for x in content.split(",") if x.strip().lstrip("-").isdigit()]
    print(f"  📋 标签: {len(labels)} 个 (类别: {sorted(set(labels))})")
    return labels


def process_motor(data_dir, output_path, node_id="edge-motor-01",
                  batch_size=60, downsample=100):
    """
    处理三相电机全部数据文件。

    Args:
        data_dir: 含 FILE N.csv 和 LABEL DATASET.csv 的目录
        output_path: 输出 JSON 路径
        node_id: 边端节点标识
        batch_size: 每个标准输入包含的采样点数
        downsample: 降采样因子 (原始100Hz → 目标1Hz = 100)
    """
    # 1. 加载标签
    label_path = os.path.join(data_dir, "LABEL DATASET.csv")
    if not os.path.exists(label_path):
        print("❌ 未找到 LABEL DATASET.csv")
        return
    labels = load_labels(label_path)

    # 2. 定位数据文件
    file_patterns = sorted(
        [f for f in os.listdir(data_dir)
         if f.startswith("FILE ") and f.endswith(".csv") and "LABEL" not in f],
        key=lambda x: int(x.replace("FILE ", "").replace(".csv", ""))
    )

    if not file_patterns:
        print("❌ 未找到 FILE N.csv 文件")
        return

    print(f"  📂 {len(file_patterns)} 个数据文件")
    print(f"  ⚙️  降采样: 100Hz → 1Hz (stride={downsample})")

    # 3. 统计每个文件的行数, 用于标签对齐
    file_rows = []
    total_rows = 0
    for fname in file_patterns:
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            row_count = sum(1 for _ in f) - 1  # 减掉 header
        file_rows.append(row_count)
        total_rows += row_count
        print(f"    {fname}: {row_count:,} 行")

    # 每个标签覆盖的行数
    rows_per_label = max(1, total_rows // len(labels))
    print(f"  🏷️  标签粒度: {rows_per_label:,} 行/标签 (总{total_rows:,}行 ÷ {len(labels)}标签)")

    # 4. 逐文件处理, 降采样 + 分批 + 贴标签
    inputs = []
    global_offset = 0  # 跨文件的全局行偏移

    # 注: LABEL DATASET.csv 中是顺序段ID(0~8999), 非故障分类标签
    # 各段对应的真实故障类型需从 .mat 文件中提取, CSV 版本暂按段ID标记
    # 段ID可用于: 时序切分、异常检测(偏离段内基线)、数据划分

    for fi, fname in enumerate(file_patterns):
        fpath = os.path.join(data_dir, fname)
        headers, rows = read_sensor_file(fpath)

        if not headers:
            print(f"  ⚠️ {fname}: 无法解析")
            continue

        # 映射列名 (SENSOR_MAP: CSV列名 → 标准字段名)
        col_indices = {}
        for csv_col, std_field in SENSOR_MAP.items():
            for hi, h in enumerate(headers):
                if h.strip() == csv_col:
                    col_indices[std_field] = hi
                    break

        file_len = len(rows)
        print(f"  🔧 {fname}: {file_len:,} 行 → {file_len // downsample:,} 采样点 (stride={downsample})")

        # 降采样 + 分批
        chunk_values = {k: [] for k in col_indices}
        chunk_labels = []
        chunk_row_count = 0

        for i in range(0, file_len, downsample):
            row = rows[i]
            global_row = global_offset + i

            # 累加降采样窗口的值
            for std_name, ci in col_indices.items():
                if ci < len(row):
                    chunk_values[std_name].append(row[ci])

            # 确定当前标签
            label_idx = min(global_row // rows_per_label, len(labels) - 1)
            label_val = labels[label_idx]

            chunk_row_count += 1

            # 批次满了就输出
            if chunk_row_count >= batch_size:
                readings = {}
                for std_name, vals in chunk_values.items():
                    if vals:
                        readings[std_name] = round(sum(vals) / len(vals), 4)

                if readings:
                    label_name = f"segment_{label_val}"
                    entry = {
                        "task_id": f"T-MOTOR-{int(tm.time()*1000)}-{len(inputs):05d}",
                        "node_id": node_id,
                        "timestamp": tm.strftime("%Y-%m-%dT%H:%M:%S"),
                        "task_type": "status_monitoring",
                        "priority": "low",
                        "data": {
                            "device_id": "three-phase-motor-01",
                            "device_type": "三相异步电机",
                            "readings": readings,
                            "context": (
                                f"电机监测, 文件{fi+1}/10, 全局行{global_row:,}, "
                                f"段ID={label_val}"
                            ),
                            "segment_id": label_val,
                        }
                    }
                    inputs.append(entry)

                # 重置批次
                chunk_values = {k: [] for k in col_indices}
                chunk_labels = []
                chunk_row_count = 0

        global_offset += file_len

    # 5. 处理最后一个不完整批次
    if chunk_row_count > 0:
        readings = {}
        for std_name, vals in chunk_values.items():
            if vals:
                readings[std_name] = round(sum(vals) / len(vals), 4)
        if readings:
            label_val = labels[-1]
            label_name = f"segment_{label_val}"
            entry = {
                "task_id": f"T-MOTOR-{int(tm.time()*1000)}-{len(inputs):05d}",
                "node_id": node_id,
                "timestamp": tm.strftime("%Y-%m-%dT%H:%M:%S"),
                "task_type": "status_monitoring",
                "priority": "low",
                "data": {
                    "device_id": "three-phase-motor-01",
                    "device_type": "三相异步电机",
                    "readings": readings,
                    "context": f"电机监测 (尾部批次), 段ID={label_val}",
                    "segment_id": label_val,
                }
            }
            inputs.append(entry)

    # 6. 统计
    # 统计段分布
    print(f"\n  ✅ 完成: {len(inputs):,} 条标准输入")
    print(f"     覆盖 {len(set(inp['data']['segment_id'] for inp in inputs))} 个数据段")
    dist = {}
    for inp in inputs:
        seg = inp["data"]["segment_id"]
        dist[seg] = dist.get(seg, 0) + 1
    # 只显示段数不超过20个的情况, 否则打印范围
    if len(dist) <= 20:
        print(f"     段分布: ", end="")
        for k, v in sorted(dist.items()):
            print(f"  seg_{k}={v}", end="")
        print()
    else:
        print(f"     段ID范围: {min(dist.keys())} ~ {max(dist.keys())}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)
    print(f"  📄 输出: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三相电机数据集预处理")
    parser.add_argument("--data-dir",
                        default=r"D:\Challenge Cup\Data file\Nature_3phase_motor\27216219")
    parser.add_argument("--output", default="processed_motor_inputs.json")
    parser.add_argument("--node", default="edge-motor-01")
    parser.add_argument("--batch-size", type=int, default=60,
                        help="每批次采样点数 (默认60 = 60秒 @1Hz)")
    parser.add_argument("--downsample", type=int, default=100,
                        help="降采样因子 (100Hz→1Hz = 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ⚡ 三相电机数据预处理")
    print(f"  数据目录: {args.data_dir}")
    print("=" * 60)

    process_motor(args.data_dir, args.output, args.node,
                  batch_size=args.batch_size, downsample=args.downsample)
