r"""
UCI液压系统数据集 — 专用预处理

原始数据: data/UCI_Hydraulic/
  FS1.txt, FS2.txt — 流量传感器 (Hz)
  PS1-PS5.txt       — 压力传感器 (bar)
  TS1-TS4.txt       — 温度传感器 (°C)
  VS1.txt           — 振动传感器 (mm/s)
  EPS1.txt          — 电机功率 (W)
  CE.txt, CP.txt    — 冷却效率, 冷却功率
  SE.txt            — 系统效率
  profile.txt       — 工况标签 (目标变量)

输出: processed_hydraulic_inputs.json (标准 I/O 格式)
"""
import sys, os, json, glob, argparse
from pathlib import Path
from sensor_csv_to_input import read_sensor_file, to_standard_input, CHANNEL_MAPS


def process_hydraulic(data_dir, output_path, node_id="edge-hydraulic-01", batch_size=60):
    """处理 UCI 液压系统全部传感器文件, 合并为统一输入流"""

    # 定义所有传感器文件及其含义
    sensor_files = {
        "FS1.txt": ("flow_l_min", 1),    # 流量传感器1 (1Hz采样)
        "FS2.txt": ("flow_l_min", 1),
        "PS1.txt": ("ps1_bar", 100),     # 压力传感器1 (100Hz)
        "PS2.txt": ("ps2_bar", 100),
        "PS3.txt": ("ps3_bar", 100),
        "PS4.txt": ("ps4_bar", 100),
        "PS5.txt": ("ps5_bar", 100),
        "TS1.txt": ("temperature_c", 1),
        "TS2.txt": ("temperature_c", 1),
        "TS3.txt": ("temperature_c", 1),
        "TS4.txt": ("temperature_c", 1),
        "VS1.txt": ("vibration_mm_s", 1),
        "EPS1.txt": ("motor_power_w", 1),
    }

    all_readings = {}  # timestamp_index → {field: value}

    for filename, (field, sample_rate) in sensor_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue

        headers, rows = read_sensor_file(filepath)

        print(f"  {filename}: {len(rows)} 行 @ {sample_rate}Hz → 字段 '{field}'")

        # 统一到1Hz (每sample_rate个值取平均)
        stride = max(1, sample_rate)
        for idx in range(0, len(rows), stride):
            t = idx // stride
            chunk = rows[idx:idx+stride]
            if not chunk:
                continue
            # 取第一个值(如果只有1列) 或 所有列平均
            vals = []
            for r in chunk:
                vals.extend(r[:1])  # 取第一列
            if vals:
                if t not in all_readings:
                    all_readings[t] = {}
                all_readings[t][field] = round(sum(vals) / len(vals), 4)

    if not all_readings:
        print("❌ 未找到数据文件, 请确认目录路径")
        return

    # 转换为标准 input.json 格式
    # 把合并后的 readings 分批打包
    import time as tm
    sorted_times = sorted(all_readings.keys())
    inputs = []
    batch_count = 0

    for i in range(0, len(sorted_times), batch_size):
        batch_times = sorted_times[i:i+batch_size]
        # 取这个批次的平均读数
        readings = {}
        for field_set in [all_readings[t] for t in batch_times]:
            for k, v in field_set.items():
                readings.setdefault(k, []).append(v)

        avg_readings = {k: round(sum(v)/len(v), 4) for k, v in readings.items()}

        entry = {
            "task_id": f"T-HYD-{int(tm.time()*1000)}-{i:05d}",
            "node_id": node_id,
            "timestamp": tm.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_type": "fault_diagnosis",
            "priority": "high",
            "data": {
                "device_id": "hydraulic-system-01",
                "device_type": "液压系统",
                "readings": avg_readings,
                "context": f"液压系统监测, 批次{i}-{i+batch_size}, {len(batch_times)}时间点",
            }
        }
        inputs.append(entry)
        batch_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 完成: {batch_count} 条标准输入 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCI液压系统预处理")
    parser.add_argument("data_dir", nargs="?", default=r"D:\Challenge Cup\Data file\UCI_Hydraulic")
    parser.add_argument("--output", default="processed_hydraulic_inputs.json")
    parser.add_argument("--node", default="edge-hydraulic-01")
    args = parser.parse_args()
    process_hydraulic(args.data_dir, args.output, args.node)
