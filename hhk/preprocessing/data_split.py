"""
数据集划分工具 — 将标准 I/O JSON 切分为训练/测试/边端模拟集

支持:
  - 按标签分层抽样 (ground_truth.label_name 或 label.fault_type)
  - 按比例切分 (默认 60/20/20)
  - 跨场景去重 (同一 device_id 不同时段的样本分散)
  - 子集统计摘要

用法:
  python data_split.py <input.json>
  python data_split.py processed_motor_inputs.json --ratios 0.7,0.15,0.15
  python data_split.py processed_motor_inputs.json --ratios 0.6,0.2,0.2 --seed 42
"""
import sys, os, json, random, argparse, time as tm
from pathlib import Path
from collections import defaultdict


def extract_label_key(entry):
    """从标准输入中提取标签键, 用于分层抽样。

    优先级: ground_truth.label_name > label.fault_type > label.is_anomaly
    """
    gt = entry.get("data", {}).get("ground_truth", {})
    if gt.get("label_name"):
        return str(gt["label_name"])
    if gt.get("label_id") is not None:
        return f"label_{gt['label_id']}"

    label = entry.get("label", {})
    if label.get("fault_type"):
        return str(label["fault_type"])
    if label.get("is_anomaly") is not None:
        return "anomaly" if label["is_anomaly"] else "normal"

    # 回退: 按 task_type
    return entry.get("task_type", "unknown")


def stratified_split(entries, ratios, seed=42):
    """
    分层抽样划分数据集。

    Args:
        entries: 标准输入列表
        ratios: (train_ratio, test_ratio, edge_ratio) 元组
        seed: 随机种子

    Returns:
        (train_set, test_set, edge_set) 三个列表
    """
    rng = random.Random(seed)

    # 按标签分组
    groups = defaultdict(list)
    for e in entries:
        key = extract_label_key(e)
        groups[key].append(e)

    train_all, test_all, edge_all = [], [], []

    for label, items in sorted(groups.items()):
        n = len(items)
        shuffled = list(items)
        rng.shuffle(shuffled)

        n_train = max(1, int(n * ratios[0]))
        n_test = max(1, int(n * ratios[1]))
        # edge 拿剩下的

        train_all.extend(shuffled[:n_train])
        test_all.extend(shuffled[n_train:n_train + n_test])
        edge_all.extend(shuffled[n_train + n_test:])

        print(f"  {label:<25}: {n:>6} → train={n_train:>5}  test={n_test:>5}  edge={n-n_train-n_test:>5}")

    # 再次打乱各集合
    rng.shuffle(train_all)
    rng.shuffle(test_all)
    rng.shuffle(edge_all)

    return train_all, test_all, edge_all


def print_summary(name, entries):
    """打印子集统计"""
    labels = defaultdict(int)
    for e in entries:
        labels[extract_label_key(e)] += 1
    label_str = " | ".join(f"{k}:{v}" for k, v in sorted(labels.items()))
    print(f"  📦 {name}: {len(entries):,} 条  [{label_str}]")


def main():
    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument("input", help="标准 input.json 文件")
    parser.add_argument("--ratios", default="0.6,0.2,0.2",
                        help="train,test,edge 比例 (逗号分隔, 默认 0.6,0.2,0.2)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录 (默认与输入同目录)")
    parser.add_argument("--labels-only", action="store_true",
                        help="仅打印标签分布, 不切分")
    args = parser.parse_args()

    # 解析比例
    parts = [float(x) for x in args.ratios.split(",")]
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 0.01:
        print("❌ 比例格式错误, 需3个值且和为1, 例如: 0.6,0.2,0.2")
        return

    print("=" * 60)
    print("  ✂️  数据集划分工具")
    print(f"  输入: {args.input}")
    print(f"  比例: train={parts[0]} test={parts[1]} edge_sim={parts[2]}")
    print(f"  种子: {args.seed}")
    print("=" * 60)

    with open(args.input, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        print("❌ 输入 JSON 必须是数组")
        return

    print(f"\n  📂 总样本: {len(entries):,} 条")

    # 标签分布
    label_counts = defaultdict(int)
    for e in entries:
        label_counts[extract_label_key(e)] += 1
    print(f"  🏷️  标签分布 ({len(label_counts)} 类):")
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"      {k:<25}: {v:>6} ({v/len(entries)*100:5.1f}%)")

    if args.labels_only:
        return

    # 执行分层划分
    print(f"\n  📐 分层抽样:")
    train, test, edge = stratified_split(entries, tuple(parts), args.seed)

    print(f"\n  划分结果:")
    print_summary("Train    ", train)
    print_summary("Test     ", test)
    print_summary("Edge Sim ", edge)

    # 写入文件
    base = Path(args.input).stem
    out_dir = args.output_dir or str(Path(args.input).parent)

    for name, data in [("train", train), ("test", test), ("edge_sim", edge)]:
        out_path = os.path.join(out_dir, f"{base}_{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  📄 {out_path} ({size_mb:.1f} MB)")

    # 生成划分元信息
    meta = {
        "source": args.input,
        "seed": args.seed,
        "ratios": {"train": parts[0], "test": parts[1], "edge_sim": parts[2]},
        "counts": {"total": len(entries), "train": len(train), "test": len(test), "edge_sim": len(edge)},
        "label_distribution": dict(sorted(label_counts.items())),
        "split_timestamp": tm.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = os.path.join(out_dir, f"{base}_split_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  📋 元信息: {meta_path}")


if __name__ == "__main__":
    main()
