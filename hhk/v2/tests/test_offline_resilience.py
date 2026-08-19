# -*- coding: utf-8 -*-
"""
断网韧性验证 (纯边端, 不需要云端)

验证当推理端点不可达 (模拟断网) 时, 边端实时数据流管线的降级能力:
  1) 提交不阻塞  —— submit() 立即返回, 不卡住采集线程
  2) 进程不崩溃  —— 推理异常被捕获, 工作线程继续存活
  3) 数据零丢失  —— 每条输入都被处理并进入本地缓存, 供网络恢复后补传

恢复阶段 (可选): 若本地已有 llama-server 运行, 用同一批输入验证补传成功率。
  —— 注意: llama-server 是本地推理服务; 这里用"不可达端点"模拟其暂时不可用,
     以检验管线的容错与恢复, 对应比赛"断网业务保持"里的零丢失+不中断要求。

用法:
  python tests/test_offline_resilience.py
  python tests/test_offline_resilience.py --endpoint http://127.0.0.1:8080 --samples 30

复用现有组件: stream_adapter.InferenceWorker (不修改任何现有文件)
"""
import sys, time, argparse, socket, threading, urllib.request
from pathlib import Path

# ── 路径: 复用 hhk 目录下的 stream_adapter / edge_inference ──
HHK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HHK_DIR))
sys.path.insert(0, str(HHK_DIR / "edge-io-protocol"))

import stream_adapter  # 内部已 import edge_inference

class BrokenEndpoint:
    """立即关闭连接的假端点 —— 快速、稳定地模拟"推理端点不可用/链路中断"。

    比直接连不存在的端口更可靠: 后者在部分 Windows 环境下每次连接要 ~2s 才失败,
    这里 accept 后立刻 close, 客户端瞬间读到连接断开 → 毫秒级失败。
    """
    def __init__(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(50)
        self.port = self._srv.getsockname()[1]
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._running:
            try:
                self._srv.settimeout(0.5)
                conn, _ = self._srv.accept()
                conn.close()  # 立即关闭 → 客户端快速失败
            except socket.timeout:
                continue
            except OSError:
                break

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self._running = False
        try:
            self._srv.close()
        except OSError:
            pass


def make_samples(n):
    """构造 n 条标准 input.json (同一 zone 的一个节点)。"""
    samples = []
    for i in range(n):
        samples.append({
            "task_id": f"T-OFFLINE-{i:04d}",
            "node_id": "edge-zoneA-01",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task_type": "fault_diagnosis",
            "priority": "high",
            "data": {
                "device_id": "motor-a01",
                "device_type": "test-motor",
                "readings": {
                    "vibration_mm_s": round(1.0 + (i % 5) * 0.7, 2),
                    "temperature_c": 50 + (i % 30),
                    "current_a": round(12.0 + (i % 3) * 0.5, 2),
                },
                "context": "offline resilience test",
            },
        })
    return samples


def endpoint_alive(endpoint):
    try:
        urllib.request.urlopen(f"{endpoint}/health", timeout=2)
        return True
    except Exception:
        return False


def drain(worker, expected, timeout_s):
    """等待工作线程把 expected 条全部处理完并落入缓存 (或超时)。

    注意: InferenceWorker 在每轮循环开头就 +1 stats['total'], 但 success/failed
    与 results 缓存是在推理结束后才写入。因此以 len(worker.results) 作为
    "已完成并缓存" 的可靠信号, 避免统计最后一条 in-flight 数据时产生竞态。
    """
    deadline = time.time() + timeout_s
    while len(worker.results) < expected and time.time() < deadline:
        time.sleep(0.2)
    time.sleep(0.3)  # 让 stats 与缓存完全落定


def run_offline_phase(n):
    print("=" * 62)
    print("  [PHASE 1] 断网阶段 —— 推理端点不可用 (链路中断)")
    print("=" * 62)

    broken = BrokenEndpoint()
    worker = stream_adapter.InferenceWorker(broken.endpoint)
    worker.start()

    t0 = time.time()
    accepted = sum(1 for s in make_samples(n) if worker.submit(s))
    submit_ms = (time.time() - t0) * 1000

    drain(worker, n, timeout_s=max(20, n))
    stats = worker.get_stats()
    worker.stop()
    broken.close()

    print(f"  提交 {n} 条 | 接受 {accepted} 条 | 提交耗时 {submit_ms:.1f} ms")
    print(f"  已处理 {stats['total']} | 成功 {stats['success']} | 失败 {stats['failed']}")
    print(f"  本地缓存 {stats['results_cached']} 条 (可供恢复后补传)")

    checks = [
        ("提交不阻塞 (<2s)",        submit_ms < 2000),
        ("全部被接受 (无丢弃)",     accepted == n),
        ("全部被处理 (无卡死)",     stats["total"] == n),
        ("端点不可达时成功=0",      stats["success"] == 0),
        ("数据零丢失 (缓存==提交)", stats["results_cached"] == n),
    ]
    all_ok = True
    for name, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    zero_loss = stats["results_cached"] / n * 100 if n else 0
    print(f"  >> 数据零丢失率: {zero_loss:.1f}%")
    return all_ok


def run_recovery_phase(n, endpoint):
    print("\n" + "=" * 62)
    print("  [PHASE 2] 恢复阶段 —— 端点恢复后补传")
    print("=" * 62)

    if not endpoint_alive(endpoint):
        print(f"  [SKIP] {endpoint} 无 llama-server 响应")
        print(f"         启动本地 llama-server 后重跑本脚本, 可验证补传成功率")
        return None

    worker = stream_adapter.InferenceWorker(endpoint)
    worker.start()
    for s in make_samples(n):
        worker.submit(s)
    drain(worker, n, timeout_s=120)
    stats = worker.get_stats()
    worker.stop()

    retention = stats["success"] / n * 100 if n else 0
    print(f"  补传 {n} 条 | 成功 {stats['success']} | 失败 {stats['failed']} "
          f"| 平均时延 {stats['avg_latency_ms']} ms")
    print(f"  >> 业务恢复率: {retention:.1f}% (目标 >=90%)")
    return retention >= 90


def main():
    ap = argparse.ArgumentParser(description="边端断网韧性验证")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080",
                    help="恢复阶段用的真实 llama-server 地址")
    ap.add_argument("--samples", type=int, default=20, help="测试样本数")
    args = ap.parse_args()

    ok1 = run_offline_phase(args.samples)
    ok2 = run_recovery_phase(args.samples, args.endpoint)

    print("\n" + "=" * 62)
    print("  总结")
    print("=" * 62)
    print(f"  断网韧性 (零丢失+不崩溃+不阻塞): {'PASS' if ok1 else 'FAIL'}")
    if ok2 is None:
        print(f"  补传恢复 (>=90%):               SKIP (需本地 llama-server)")
    else:
        print(f"  补传恢复 (>=90%):               {'PASS' if ok2 else 'FAIL'}")

    passed = ok1 and (ok2 in (None, True))
    print(f"\n  结果: {'全部通过' if passed else '存在失败项'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
