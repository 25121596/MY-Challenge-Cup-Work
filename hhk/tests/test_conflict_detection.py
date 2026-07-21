# -*- coding: utf-8 -*-
"""
多节点冲突检测 + 本地仲裁验证 (纯边端, 不需要云端/服务)

直接复用 edge_inference.make_output() 产出的 consistency.decision_hash 与 zone,
验证同一区域多个边端节点的决策一致性检测逻辑 (不发起任何网络请求):

  场景A 共识: 同 zone 所有节点同类别 -> decision_hash 相同 -> 判定无冲突
  场景B 冲突: 同 zone 节点类别分歧   -> decision_hash 不同 -> 判定有冲突
  场景C 仲裁: 对冲突 zone 做多数决, 验证可解决率 (对应"冲突解决率>=90%")

并在一个模拟机群上计算冲突率, 对照比赛指标 (冲突率 <= 5%)。

用法:
  python tests/test_conflict_detection.py

说明: 冲突"率"取决于输入数据本身; 本脚本重点验证"检测/仲裁逻辑正确",
      冲突率/解决率作为可度量指标一并打印, 证明该指标可被采集。
复用现有组件: edge_inference.make_output (不修改任何现有文件)
"""
import sys
from pathlib import Path
from collections import defaultdict, Counter

# ── 路径: 复用 edge-io-protocol/edge_inference.py ──
HHK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HHK_DIR / "edge-io-protocol"))

import edge_inference as ei


def node_decision(node_id, category, confidence=85, request_cloud=0):
    """用真实的 make_output 生成一个节点决策 (纯函数, 无网络)。"""
    return ei.make_output(
        task_id=f"T-{node_id}",
        node_id=node_id,
        category=category,
        confidence=confidence,
        request_cloud=request_cloud,
        latency_ms=10.0,
    )


def detect_conflicts(outputs):
    """按 zone 分组, 依据 decision_hash 判定每个 zone 是否存在冲突。

    Returns: (reports: dict[zone]->info, conflict_zone_count, total_zone_count)
    """
    by_zone = defaultdict(list)
    for o in outputs:
        by_zone[o["consistency"]["zone"]].append(o)

    reports = {}
    conflict_zones = 0
    for zone, outs in by_zone.items():
        hashes = {o["consistency"]["decision_hash"] for o in outs}
        conflict = len(hashes) > 1
        conflict_zones += 1 if conflict else 0
        reports[zone] = {
            "nodes": len(outs),
            "distinct_decisions": len(hashes),
            "categories": [o["decision"]["category"] for o in outs],
            "conflict": conflict,
        }
    return reports, conflict_zones, len(by_zone)


def majority_arbiter(outputs):
    """本地多数决仲裁。Returns: (resolved_category, resolvable)。

    resolvable = 存在唯一严格多数 (票数 > 半数)。否则需甩云裁决。
    """
    cats = [o["decision"]["category"] for o in outputs]
    counts = Counter(cats)
    top_cat, top_n = counts.most_common(1)[0]
    tie = sum(1 for c, n in counts.items() if n == top_n) > 1
    resolvable = (top_n > len(cats) / 2) and not tie
    return top_cat, resolvable


def check(name, ok):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def test_consensus():
    print("=" * 62)
    print("  [场景A] 共识 —— 同 zone 3 节点均判 normal(0)")
    print("=" * 62)
    outs = [node_decision(f"edge-zoneA-{i:02d}", 0) for i in (1, 2, 3)]
    reports, conflict_zones, _ = detect_conflicts(outs)
    r = reports["edge-zoneA"]
    print(f"  zone=edge-zoneA nodes={r['nodes']} 决策类别={r['categories']} "
          f"distinct={r['distinct_decisions']} conflict={r['conflict']}")
    ok = check("共识场景应判定为无冲突", r["conflict"] is False)
    ok &= check("decision_hash 完全一致 (distinct==1)", r["distinct_decisions"] == 1)
    return ok


def test_conflict():
    print("\n" + "=" * 62)
    print("  [场景B] 冲突 —— 同 zone 节点判 [1,1,2] 分歧")
    print("=" * 62)
    outs = [
        node_decision("edge-zoneB-01", 1),
        node_decision("edge-zoneB-02", 1),
        node_decision("edge-zoneB-03", 2),
    ]
    reports, conflict_zones, _ = detect_conflicts(outs)
    r = reports["edge-zoneB"]
    print(f"  zone=edge-zoneB nodes={r['nodes']} 决策类别={r['categories']} "
          f"distinct={r['distinct_decisions']} conflict={r['conflict']}")
    ok = check("分歧场景应检出冲突", r["conflict"] is True)
    ok &= check("distinct_decisions == 2", r["distinct_decisions"] == 2)
    return ok


def test_arbiter():
    print("\n" + "=" * 62)
    print("  [场景C] 本地仲裁 —— 多数决解决冲突")
    print("=" * 62)
    # [1,1,2]: 类别1 占多数 -> 可解决为 1
    outs = [
        node_decision("edge-zoneB-01", 1),
        node_decision("edge-zoneB-02", 1),
        node_decision("edge-zoneB-03", 2),
    ]
    resolved, resolvable = majority_arbiter(outs)
    print(f"  投票={[o['decision']['category'] for o in outs]} "
          f"-> 仲裁类别={resolved} 可本地解决={resolvable}")
    ok = check("多数决可解决 (类别1 严格多数)", resolvable is True and resolved == 1)

    # 平票场景 [1,2] -> 不可本地解决 -> 应甩云
    outs2 = [node_decision("edge-zoneC-01", 1), node_decision("edge-zoneC-02", 2)]
    _, resolvable2 = majority_arbiter(outs2)
    print(f"  投票={[o['decision']['category'] for o in outs2]} "
          f"-> 可本地解决={resolvable2} (平票应甩云)")
    ok &= check("平票场景不可本地解决 (交云端裁决)", resolvable2 is False)
    return ok


def fleet_metrics():
    """模拟一个机群: 20 个 zone, 每 zone 3 节点; 注入 1 个冲突 zone。
    计算冲突率 / 冲突解决率, 证明指标可采集并对照阈值。"""
    print("\n" + "=" * 62)
    print("  [机群度量] 20 zone x 3 节点, 注入 1 个冲突 zone")
    print("=" * 62)

    outputs = []
    n_zones = 20
    conflict_zone_ids = {7}  # 注入冲突的 zone 下标
    for z in range(n_zones):
        base_cat = z % 4  # 各 zone 正常判定
        if z in conflict_zone_ids:
            cats = [base_cat, base_cat, (base_cat + 1) % 7]  # 1 个节点分歧
        else:
            cats = [base_cat, base_cat, base_cat]
        for i, c in enumerate(cats):
            outputs.append(node_decision(f"edge-zone{z:02d}-{i:02d}", c))

    reports, conflict_zones, total_zones = detect_conflicts(outputs)
    conflict_rate = conflict_zones / total_zones * 100 if total_zones else 0

    # 对冲突 zone 做仲裁, 统计解决率
    resolved = 0
    for zone, r in reports.items():
        if r["conflict"]:
            outs = [o for o in outputs if o["consistency"]["zone"] == zone]
            _, ok = majority_arbiter(outs)
            resolved += 1 if ok else 0
    resolution_rate = resolved / conflict_zones * 100 if conflict_zones else 100.0

    print(f"  总 zone={total_zones} 冲突 zone={conflict_zones}")
    print(f"  >> 冲突率: {conflict_rate:.1f}% (目标 <=5%)  "
          f"{'PASS' if conflict_rate <= 5 else 'FAIL'}")
    print(f"  >> 冲突解决率: {resolution_rate:.1f}% (目标 >=90%)  "
          f"{'PASS' if resolution_rate >= 90 else 'FAIL'}")
    # 度量能力本身即为通过条件 (指标可计算且注入的冲突被检出/解决)
    return conflict_zones == len(conflict_zone_ids) and resolution_rate >= 90


def main():
    results = [
        ("场景A 共识检测", test_consensus()),
        ("场景B 冲突检测", test_conflict()),
        ("场景C 本地仲裁", test_arbiter()),
        ("机群冲突率/解决率度量", fleet_metrics()),
    ]

    print("\n" + "=" * 62)
    print("  总结")
    print("=" * 62)
    all_ok = True
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    print(f"\n  结果: {'全部通过' if all_ok else '存在失败项'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
