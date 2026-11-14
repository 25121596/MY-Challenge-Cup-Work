"""
统一 I/O 契约 — 所有数据源输出同一套记录结构

CSV 列: source, sample_id, window_id, timestamp_ms, fs,
         channel_count, feat_0..feat_{N-1}, meta_json

features 字段为定长浮点向量, 云端无需关心底层物理含义。
"""
import csv
import json
import os
import time


def write_features_csv(records, output_path, feature_names=None):
    """将特征记录列表写入统一 CSV。

    Args:
        records: list of dict, 每条记录包含:
            - source: str
            - sample_id: int
            - window_id: int
            - timestamp_ms: float
            - fs: float (采样率)
            - channel_count: int
            - features: list[float] (定长)
            - meta: dict (标签/状态等, 序列化为 JSON)
    """
    if not records:
        print("  ⚠ 无记录, 跳过写入")
        return

    n_features = len(records[0]["features"])
    feat_cols = [f"feat_{i}" for i in range(n_features)]

    fieldnames = ["source", "sample_id", "window_id", "timestamp_ms",
                  "fs", "channel_count"] + feat_cols + ["meta_json"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            row = {
                "source": rec["source"],
                "sample_id": rec["sample_id"],
                "window_id": rec["window_id"],
                "timestamp_ms": rec["timestamp_ms"],
                "fs": rec["fs"],
                "channel_count": rec["channel_count"],
                "meta_json": json.dumps(rec.get("meta", {}), ensure_ascii=False),
            }
            for i, val in enumerate(rec["features"]):
                row[f"feat_{i}"] = round(val, 6)
            writer.writerow(row)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  📄 {output_path}: {len(records)} 条 × {n_features} 维 ({size_mb:.1f} MB)")


def write_feature_schema(dataset_name, channel_names, feature_names, output_path):
    """写出特征的 JSON Schema, 供云端解析 features 字段。"""
    schema = {
        "dataset": dataset_name,
        "channel_count": len(channel_names),
        "channels": channel_names,
        "features_per_channel": len(feature_names) // len(channel_names)
        if channel_names else 0,
        "total_dimension": len(feature_names),
        "feature_names": feature_names,
        "note": "features = [ch0_mean, ch0_std, ..., ch0_spectral_rolloff, ch1_mean, ...]",
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"  📋 Schema: {output_path}")


def write_summary(summaries, output_path):
    """写出预处理摘要 JSON"""
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_windows": sum(s.get("total_windows", 0) for s in summaries),
        "datasets": summaries,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  📋 摘要: {output_path}")
