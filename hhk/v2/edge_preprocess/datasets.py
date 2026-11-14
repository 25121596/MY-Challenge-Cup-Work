"""
三个数据集的适配器 — 各自实现 extract(), 输出统一记录列表
"""
import os
import time
import json
from pathlib import Path
from edge_preprocess.features import extract_multichannel_features, extract_channel_features, has_numpy
from edge_preprocess.stream import BatchWindowReader, read_sensor_columns


# ══════════════════════════════════════════════════════
#  UCI 液压系统
# ══════════════════════════════════════════════════════

# 选取 6 个代表通道 (3 类采样率)
UCI_CHANNELS = [
    "PS1",   # 压力 100Hz
    "PS2",   # 压力 100Hz
    "EPS1",  # 功率 100Hz
    "FS1",   # 流量 10Hz
    "TS1",   # 温度 1Hz
    "VS1",   # 振动 1Hz
]

# 采样率映射 (Hz)
UCI_SAMPLE_RATES = {
    "PS1": 100, "PS2": 100, "EPS1": 100,
    "FS1": 10, "TS1": 1, "VS1": 1,
}


def extract_uci_hydraulic(data_dir, max_cycles=500):
    """UCI 液压系统: 按 60s 工作循环提取特征。

    每个文件 = 一个传感器的时间序列 (TSV, 无header, 单列数值)。
    60s 工作循环 = 60 × sample_rate 个采样点。

    Args:
        data_dir: UCI_Hydraulic 目录路径
        max_cycles: 最大循环数 (演示用 500)

    Returns:
        (records, channel_names, feature_names, summary)
    """
    t0 = time.time()
    records = []
    channel_names = []
    all_feature_names = []

    # 逐文件读取 + 逐循环切片
    for ch_name in UCI_CHANNELS:
        filepath = os.path.join(data_dir, f"{ch_name}.txt")
        if not os.path.exists(filepath):
            print(f"  ⚠ 跳过: {filepath} 不存在")
            continue

        # 读取: TSV, 每行是一个 60s 循环的所有采样点
        cycles = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Tab 分隔的数值
                samples = []
                for tok in line.split("\t"):
                    try:
                        samples.append(float(tok))
                    except ValueError:
                        continue
                if samples:
                    cycles.append(samples)

        fs = UCI_SAMPLE_RATES.get(ch_name, 1)
        expected_samples = 60 * fs  # 理论每循环采样点数

        print(f"  {ch_name}: {len(cycles)} 循环 @{fs}Hz "
              f"(每循环 {len(cycles[0]) if cycles else 0} 采样点, 期望 {expected_samples})")

        channel_names.append(ch_name)

        # 如果是第一个通道, 初始化 records 列表结构
        if not records:
            n_cycles = min(len(cycles), max_cycles)
            records = [{} for _ in range(n_cycles)]

        # 逐循环提取特征
        for ci, chunk in enumerate(cycles[:max_cycles]):
            if ci >= len(records):
                break

            feats, fnames = extract_channel_features(chunk, fs)
            ch_prefix = f"{ch_name}_"

            if "features" not in records[ci]:
                records[ci]["features"] = []
                records[ci]["source"] = "uci_hydraulic"
                records[ci]["sample_id"] = ci
                records[ci]["window_id"] = ci
                records[ci]["timestamp_ms"] = ci * 60_000.0  # 循环开始时刻 ms
                records[ci]["fs"] = fs
                records[ci]["channel_count"] = 0
                records[ci]["meta"] = {"cycle_seconds": 60, "cycle_index": ci}

            records[ci]["features"].extend(feats)
            records[ci]["channel_count"] += 1

        # 记录特征名 (所有通道拼接)
        all_feature_names.extend([f"{ch_name}_{n}" for n in fnames])

    # 过滤掉不完整的循环
    expected_dim = len(UCI_CHANNELS) * 17
    valid = [r for r in records if len(r.get("features", [])) == expected_dim]
    dropped = len(records) - len(valid)

    elapsed = (time.time() - t0) * 1000
    latency_per_cycle = elapsed / max(len(valid), 1)

    summary = {
        "dataset": "uci_hydraulic",
        "total_windows": len(valid),
        "dropped_incomplete": dropped,
        "channels": len(channel_names),
        "feature_dim": expected_dim,
        "avg_extraction_ms": round(latency_per_cycle, 2),
        "total_time_ms": round(elapsed, 1),
    }

    print(f"  ✅ UCI: {len(valid)} 循环 ({dropped} 不完整), "
          f"每循环 {latency_per_cycle:.2f}ms")
    return valid, channel_names, all_feature_names, summary


# ══════════════════════════════════════════════════════
#  Nature 三相电机
# ══════════════════════════════════════════════════════

NATURE_CHANNELS = ["x", "y", "Z", "I1", "I2", "I3", "V1", "V2", "V3"]
NATURE_FS = 50_000  # 50 kHz


def extract_nature_motor(data_dir, window_size=2560, max_windows=200):
    """Nature 三相电机: 流式滑动窗口提取特征。

    窗口 2560 点 @50kHz ≈ 51.2ms。
    逐行读取, 内存上限 9×2560×8 ≈ 184KB。

    Args:
        data_dir: 含 FILE 1.csv ~ FILE 10.csv 的目录
        window_size: 窗口采样点数
        max_windows: 最大窗口数 (演示用 200)

    Returns:
        (records, channel_names, feature_names, summary)
    """
    t0 = time.time()
    import csv

    data_file = os.path.join(data_dir, "FILE 1.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    # 初始化: 9 个通道的 buffer
    buffers = {ch: [] for ch in NATURE_CHANNELS}
    records = []
    col_indices = {}

    with open(data_file, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            for i, h in enumerate(header):
                h_clean = h.strip()
                if h_clean in NATURE_CHANNELS:
                    col_indices[h_clean] = i

        if len(col_indices) < len(NATURE_CHANNELS):
            # 列名可能不完全匹配, 尝试按位置映射
            found = set(col_indices.keys())
            for ch in NATURE_CHANNELS:
                if ch not in found:
                    for i, h in enumerate(header):
                        if h.strip().upper() == ch.upper():
                            col_indices[ch] = i
                            break

        window_id = 0
        row_count = 0

        for row in reader:
            row_count += 1
            if max_windows and window_id >= max_windows:
                break

            # 逐通道收集数据
            for ch in NATURE_CHANNELS:
                ci = col_indices.get(ch)
                if ci is None or ci >= len(row):
                    continue
                try:
                    val = float(row[ci].strip())
                except (ValueError, IndexError):
                    val = 0.0
                buffers[ch].append(val)

            # 检查是否所有通道都凑齐一个窗口
            if all(len(buffers[ch]) >= window_size for ch in NATURE_CHANNELS):
                # 提取特征
                channels_data = [buffers[ch][:window_size]
                                 for ch in NATURE_CHANNELS]
                feats, fnames, n_ch, dim = extract_multichannel_features(
                    channels_data, fs=NATURE_FS
                )

                records.append({
                    "source": "nature_motor",
                    "sample_id": window_id,
                    "window_id": window_id,
                    "timestamp_ms": window_id * (window_size / NATURE_FS * 1000),
                    "fs": NATURE_FS,
                    "channel_count": n_ch,
                    "features": feats,
                    "meta": {
                        "window_size": window_size,
                        "window_duration_ms": window_size / NATURE_FS * 1000,
                        "file_row_start": row_count - window_size + 1,
                    },
                })

                # 清空 buffer (非重叠)
                for ch in NATURE_CHANNELS:
                    buffers[ch] = buffers[ch][window_size:]

                window_id += 1

    elapsed = (time.time() - t0) * 1000
    latency_per_window = elapsed / max(len(records), 1)

    summary = {
        "dataset": "nature_motor",
        "total_windows": len(records),
        "channels": len(NATURE_CHANNELS),
        "feature_dim": len(NATURE_CHANNELS) * 17,
        "window_size": window_size,
        "window_duration_ms": round(window_size / NATURE_FS * 1000, 1),
        "avg_extraction_ms": round(latency_per_window, 2),
        "total_time_ms": round(elapsed, 1),
    }

    channel_names = NATURE_CHANNELS
    all_feature_names = fnames  # from extract_multichannel_features

    print(f"  ✅ Nature: {len(records)} 窗口, 每窗 {latency_per_window:.2f}ms "
          f"(内存 ~{len(NATURE_CHANNELS)*window_size*8/1024:.0f}KB)")

    return records, channel_names, all_feature_names, summary


# ══════════════════════════════════════════════════════
#  AI4I 铣削
# ══════════════════════════════════════════════════════

AI4I_FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

AI4I_LABEL_COLS = [
    "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF",
]


def extract_ai4i_milling(data_dir, max_rows=None):
    """AI4I 铣削: 特征级数据统一对齐。

    原始数据已是聚合统计量。
    处理: 去标识 → Type 独热 → 数值 z-score → 标签放 meta。

    Args:
        data_dir: AI4I 数据目录
        max_rows: 最大行数 (None=全量)

    Returns:
        (records, feature_names, summary)
    """
    t0 = time.time()
    import csv

    # 定位 CSV
    csv_files = list(Path(data_dir).rglob("ai4i2020.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 ai4i2020.csv, 搜索路径: {data_dir}")
    filepath = str(csv_files[0])

    # 全量读取 (AI4I 只有 10000 行, 很轻量)
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break

    # 收集 Type 的所有可能值做独热编码
    type_values = sorted(set(r["Type"] for r in rows if r.get("Type")))
    type_map = {t: i for i, t in enumerate(type_values)}

    # z-score 用全量估计 (模拟"云端已下发归一化参数")
    all_vals = {col: [] for col in AI4I_FEATURE_COLS}
    for r in rows:
        for col in AI4I_FEATURE_COLS:
            try:
                all_vals[col].append(float(r[col]))
            except (ValueError, KeyError):
                all_vals[col].append(0.0)

    mean_std = {}
    for col in AI4I_FEATURE_COLS:
        vals = all_vals[col]
        n = len(vals)
        mu = sum(vals) / n if n > 0 else 0.0
        sigma = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5 if n > 0 else 1.0
        mean_std[col] = (mu, sigma)

    # 逐行处理
    records = []
    for si, row in enumerate(rows):
        feats = []

        # Type 独热编码
        type_vec = [0.0] * len(type_values)
        t = row.get("Type", "")
        if t in type_map:
            type_vec[type_map[t]] = 1.0
        feats.extend(type_vec)

        # 数值列 z-score
        for col in AI4I_FEATURE_COLS:
            try:
                val = float(row[col])
            except (ValueError, KeyError):
                val = 0.0
            mu, sigma = mean_std[col]
            if sigma > 1e-12:
                val = (val - mu) / sigma
            feats.append(round(val, 6))

        # 标签 → meta (不进特征向量)
        meta = {"type": t}
        for lc in AI4I_LABEL_COLS:
            try:
                meta[lc.lower().replace(" ", "_")] = int(row.get(lc, 0))
            except (ValueError, KeyError):
                meta[lc.lower().replace(" ", "_")] = 0

        records.append({
            "source": "ai4i_milling",
            "sample_id": si,
            "window_id": 0,
            "timestamp_ms": 0.0,
            "fs": 0.0,
            "channel_count": 1,
            "features": feats,
            "meta": meta,
        })

    feature_names = [f"type_{t}" for t in type_values] + \
                    [f"zscore_{c}" for c in AI4I_FEATURE_COLS]

    elapsed = (time.time() - t0) * 1000
    summary = {
        "dataset": "ai4i_milling",
        "total_windows": len(records),
        "channels": 1,
        "feature_dim": len(feature_names),
        "type_categories": len(type_values),
        "total_time_ms": round(elapsed, 1),
    }

    print(f"  ✅ AI4I: {len(records)} 条, {len(feature_names)} 维 "
          f"({len(type_values)} 独热 + {len(AI4I_FEATURE_COLS)} 归一化)")

    return records, ["features"], feature_names, summary
