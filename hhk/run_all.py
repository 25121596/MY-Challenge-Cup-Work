#!/usr/bin/env python
"""
边缘侧多源异构数据实时预处理 — 编排脚本

一键运行三个数据集的流式特征提取, 产出统一 CSV + Schema + 摘要。

用法:
  python run_all.py
  python run_all.py --output processed/

产物:
  processed/uci_hydraulic_features.csv     500 循环 × 102 维
  processed/nature_motor_features.csv      200 窗口 × 153 维
  processed/ai4i_milling_features.csv      10000 条 × 8 维
  processed/feature_schema.json            统一特征契约
  processed/preprocessing_summary.json     预处理摘要
"""
import sys, os, json, time, argparse
from pathlib import Path

# 确保可以 import edge_preprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edge_preprocess.datasets import (
    extract_uci_hydraulic, extract_nature_motor, extract_ai4i_milling,
    UCI_CHANNELS, NATURE_CHANNELS, NATURE_FS,
)
from edge_preprocess.features import ALL_FEATURE_NAMES
from edge_preprocess.schema import (
    write_features_csv, write_feature_schema, write_summary,
)


# ══════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIRS = {
    "uci": str(SCRIPT_DIR / "data" / "UCI_Hydraulic"),
    "nature": str(SCRIPT_DIR / "data" / "Nature_3phase_motor" / "27216219"),
    "ai4i": str(SCRIPT_DIR / "data" / "AI4I_2020_milling"),
}


def main():
    parser = argparse.ArgumentParser(description="边缘侧实时预处理 — 编排脚本")
    parser.add_argument("--output", default="processed",
                        help="输出目录 (默认 processed/)")
    parser.add_argument("--uci-max-cycles", type=int, default=500,
                        help="UCI 最大循环数 (默认 500)")
    parser.add_argument("--nature-max-windows", type=int, default=200,
                        help="Nature 最大窗口数 (默认 200)")
    parser.add_argument("--ai4i-max-rows", type=int, default=None,
                        help="AI4I 最大行数 (默认全量 10000)")
    parser.add_argument("--nature-window", type=int, default=2560,
                        help="Nature 窗口大小 (默认 2560 ≈ 51ms @50kHz)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  边侧多源异构数据实时预处理")
    print(f"  输出目录: {out_dir.resolve()}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_summaries = []
    total_t0 = time.time()
    products = []

    # ── 1. UCI 液压系统 ──
    print("\n" + "─" * 40)
    print("  [1/3] UCI 液压系统")
    print("─" * 40)

    uci_dir = DATA_DIRS["uci"]
    if os.path.isdir(uci_dir):
        records, ch_names, feat_names_uci, summary = \
            extract_uci_hydraulic(uci_dir, max_cycles=args.uci_max_cycles)

        csv_path = str(out_dir / "uci_hydraulic_features.csv")
        write_features_csv(records, csv_path, feat_names_uci)
        write_feature_schema("uci_hydraulic", ch_names, feat_names_uci,
                             str(out_dir / "uci_hydraulic_schema.json"))
        all_summaries.append(summary)
        products.append({
            "dataset": "uci_hydraulic",
            "csv": csv_path,
            "records": len(records),
            "dim": len(feat_names_uci),
        })
    else:
        print(f"  ⚠ UCI 数据目录不存在: {uci_dir}")

    # ── 2. Nature 三相电机 ──
    print("\n" + "─" * 40)
    print("  [2/3] Nature 三相电机")
    print("─" * 40)

    nature_dir = DATA_DIRS["nature"]
    if os.path.isdir(nature_dir):
        records, ch_names, feat_names_nat, summary = \
            extract_nature_motor(nature_dir,
                                 window_size=args.nature_window,
                                 max_windows=args.nature_max_windows)

        csv_path = str(out_dir / "nature_motor_features.csv")
        write_features_csv(records, csv_path, feat_names_nat)
        write_feature_schema("nature_motor", ch_names, feat_names_nat,
                             str(out_dir / "nature_motor_schema.json"))
        all_summaries.append(summary)
        products.append({
            "dataset": "nature_motor",
            "csv": csv_path,
            "records": len(records),
            "dim": len(feat_names_nat),
        })
    else:
        print(f"  ⚠ Nature 数据目录不存在: {nature_dir}")

    # ── 3. AI4I 铣削 ──
    print("\n" + "─" * 40)
    print("  [3/3] AI4I 铣削")
    print("─" * 40)

    ai4i_dir = DATA_DIRS["ai4i"]
    if os.path.isdir(ai4i_dir):
        records, ch_names, feat_names_ai4i, summary = \
            extract_ai4i_milling(ai4i_dir, max_rows=args.ai4i_max_rows)

        csv_path = str(out_dir / "ai4i_milling_features.csv")
        write_features_csv(records, csv_path, feat_names_ai4i)
        write_feature_schema("ai4i_milling", ch_names, feat_names_ai4i,
                             str(out_dir / "ai4i_milling_schema.json"))
        all_summaries.append(summary)
        products.append({
            "dataset": "ai4i_milling",
            "csv": csv_path,
            "records": len(records),
            "dim": len(feat_names_ai4i),
        })
    else:
        print(f"  ⚠ AI4I 数据目录不存在: {ai4i_dir}")

    # ── 汇总 ──
    total_elapsed = (time.time() - total_t0) * 1000
    print("\n" + "=" * 60)
    print("  📊 预处理完成")
    print("=" * 60)

    for p in products:
        print(f"  {p['dataset']:<20}: {p['records']:>6} 条 × {p['dim']:>4} 维 → {p['csv']}")

    # 写入统一 schema
    unified_schema = {
        "format": "CSV with columns: source, sample_id, window_id, timestamp_ms, fs, channel_count, feat_0..feat_N, meta_json",
        "datasets": products,
        "note": "云端通过 source 字段区分数据源, features 按 channel×17 顺序排列",
    }
    schema_path = str(out_dir / "feature_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(unified_schema, f, ensure_ascii=False, indent=2)

    # 写入摘要
    summary_path = str(out_dir / "preprocessing_summary.json")
    write_summary(all_summaries, summary_path)

    print(f"\n  总耗时: {total_elapsed/1000:.2f}s")
    print(f"  产物目录: {out_dir.resolve()}")
    print(f"\n✨ 全部完成!")


if __name__ == "__main__":
    main()
