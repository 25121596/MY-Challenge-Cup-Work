"""
边端实时数据流适配器

支持三种数据流接入方式:
  1. TCP Socket — 传感器/PLC 通过 TCP 推送 JSON 数据
  2. 模拟流 — 读取 CSV 文件逐行模拟实时数据流 (演示用)
  3. 目录监听 — 监听文件夹中新增的 CSV 文件 (落盘即推理)

核心原则:
  - 本地推理不依赖网络 (断网照跑)
  - 推理线程与接收线程解耦 (不阻塞)
  - 结果写入本地队列 (网络恢复后可补传)

用法:
  # 模拟流模式 (用已有CSV模拟实时数据)
  python stream_adapter.py --mode simulate --data "data/UCI_Hydraulic/PS1.txt"

  # TCP 监听模式 (等待传感器推送)
  python stream_adapter.py --mode tcp --port 9000

  # 目录监听模式 (新文件落盘即推理)
  python stream_adapter.py --mode watch --watch-dir ./incoming_data
"""
import sys, os, json, time, socket, threading, queue, argparse, csv
from pathlib import Path
from collections import deque

# 路径
SCRIPT_DIR = Path(__file__).resolve().parent
EDGE_IO_DIR = SCRIPT_DIR / "edge-io-protocol"
sys.path.insert(0, str(EDGE_IO_DIR))

import edge_inference

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
RESULT_QUEUE_MAX = 1000  # 本地结果缓存上限
BATCH_INTERVAL_S = 0.5   # 批量推理间隔


# ══════════════════════════════════════════════════════
#  数据流 → 标准 I/O 转换
# ══════════════════════════════════════════════════════

def raw_to_standard_input(raw_data, node_id="edge-stream-01"):
    """将原始流数据转为标准 input.json 格式。

    支持两种输入:
      1. 已经是标准格式的 JSON (含 task_id, data.readings)
      2. 纯传感器读数 JSON (如 {"vibration": 5.1, "temperature": 85})
    """
    # 已经是标准格式
    if "task_id" in raw_data and "data" in raw_data:
        return raw_data

    # 纯读数 → 包装为标准格式
    readings = raw_data.get("readings", raw_data)
    # 过滤非数字字段
    readings = {k: v for k, v in readings.items()
                if isinstance(v, (int, float))}

    return {
        "task_id": f"T-STREAM-{int(time.time()*1000)}",
        "node_id": node_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_type": raw_data.get("task_type", "fault_diagnosis"),
        "priority": raw_data.get("priority", "high"),
        "data": {
            "device_id": raw_data.get("device_id", "stream-device-01"),
            "device_type": raw_data.get("device_type", "实时传感器"),
            "readings": readings,
            "context": "实时数据流输入",
        }
    }


# ══════════════════════════════════════════════════════
#  推理工作线程 (与接收解耦)
# ══════════════════════════════════════════════════════

class InferenceWorker:
    """后台推理线程, 从队列取数据推理, 结果存入本地缓存"""

    def __init__(self, endpoint=DEFAULT_ENDPOINT):
        self.endpoint = endpoint
        self.input_queue = queue.Queue(maxsize=500)
        self.results = deque(maxlen=RESULT_QUEUE_MAX)
        self.running = False
        self._thread = None
        self.stats = {"total": 0, "success": 0, "failed": 0, "latencies": []}

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, standard_input):
        """提交一条数据到推理队列 (非阻塞)"""
        try:
            self.input_queue.put_nowait(standard_input)
            return True
        except queue.Full:
            return False

    def _loop(self):
        edge_inference.LLAMA_SERVER = self.endpoint
        while self.running:
            try:
                item = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            self.stats["total"] += 1
            try:
                result = edge_inference.infer(item, use_grammar=True)
                self.stats["success"] += 1
                self.stats["latencies"].append(result["output"]["latency_ms"])
                self.results.append({
                    "task_id": item.get("task_id"),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "output": result["output"],
                    "raw": result["raw_model_response"],
                })
            except Exception as e:
                self.stats["failed"] += 1
                self.results.append({
                    "task_id": item.get("task_id"),
                    "error": str(e),
                })

    def get_stats(self):
        lats = self.stats["latencies"]
        return {
            "total": self.stats["total"],
            "success": self.stats["success"],
            "failed": self.stats["failed"],
            "queue_size": self.input_queue.qsize(),
            "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
            "results_cached": len(self.results),
        }


# ══════════════════════════════════════════════════════
#  模式 1: TCP Socket 监听
# ══════════════════════════════════════════════════════

def run_tcp_mode(port, worker, node_id):
    """监听 TCP 端口, 接收 JSON 行数据"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(5)
    server.settimeout(1.0)

    print(f"  🌐 TCP 监听: 0.0.0.0:{port}")
    print(f"  📡 等待传感器连接... (Ctrl+C 停止)")

    while worker.running:
        try:
            conn, addr = server.accept()
            print(f"  📎 连接: {addr}")
            handle_tcp_client(conn, worker, node_id)
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break

    server.close()


def handle_tcp_client(conn, worker, node_id):
    """处理单个 TCP 客户端"""
    buffer = ""
    conn.settimeout(1.0)
    while worker.running:
        try:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data.decode("utf-8", errors="replace")

            # 按行分割 (每行一个 JSON)
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    std_input = raw_to_standard_input(raw, node_id)
                    worker.submit(std_input)
                except json.JSONDecodeError:
                    pass
        except socket.timeout:
            continue
        except Exception:
            break
    conn.close()


# ══════════════════════════════════════════════════════
#  模式 2: 模拟流 (读文件逐行发送)
# ══════════════════════════════════════════════════════

def run_simulate_mode(data_path, worker, node_id, interval=0.1):
    """读取数据文件, 逐行模拟实时数据流"""
    data_path = Path(data_path)
    print(f"  📂 模拟数据源: {data_path}")
    print(f"  ⏱️  发送间隔: {interval}s")

    if data_path.suffix in (".csv", ".txt"):
        # 读取 CSV/TXT, 逐行作为传感器读数
        with open(data_path, "r", encoding="utf-8", errors="replace") as f:
            # 自动检测分隔符
            first_line = f.readline()
            delimiter = "\t" if "\t" in first_line else ","
            f.seek(0)

            reader = csv.reader(f, delimiter=delimiter)
            first_row = next(reader, None)
            if not first_row:
                print("  ❌ 空文件")
                return

            # 判断是否有表头 (第一行是否全为数字)
            has_header = False
            try:
                [float(v.strip()) for v in first_row if v.strip()]
            except ValueError:
                has_header = True

            if has_header:
                headers = [h.strip().strip("\ufeff") for h in first_row]
            else:
                # 无表头: 生成 col_0, col_1, ... 列名, 并把第一行当数据
                headers = [f"col_{i}" for i in range(len(first_row))]

            print(f"  📐 {len(headers)} 列 | 表头={'有' if has_header else '无(自动编号)'}")
            count = 0

            # 处理第一行 (如果不是表头)
            rows_to_process = []
            if not has_header:
                rows_to_process.append(first_row)

            for row in reader:
                rows_to_process.append(row)

            for row in rows_to_process:
                if not worker.running:
                    break

                # 构建读数 dict
                readings = {}
                for i, val in enumerate(row):
                    if i < len(headers):
                        try:
                            readings[headers[i]] = float(val.strip())
                        except (ValueError, AttributeError):
                            pass

                if readings:
                    raw = {"readings": readings, "device_id": data_path.stem}
                    std_input = raw_to_standard_input(raw, node_id)
                    worker.submit(std_input)
                    count += 1

                    if count % 50 == 0:
                        stats = worker.get_stats()
                        print(f"    [{count}] 已发送 | "
                              f"推理: {stats['success']} ok, {stats['failed']} err | "
                              f"avg={stats['avg_latency_ms']}ms | "
                              f"队列={stats['queue_size']}")

                time.sleep(interval)

    print(f"\n  ✅ 模拟完成: {count} 条数据")


# ══════════════════════════════════════════════════════
#  模式 3: 目录监听
# ══════════════════════════════════════════════════════

def run_watch_mode(watch_dir, worker, node_id):
    """监听目录中新增的 JSON/CSV 文件"""
    watch_path = Path(watch_dir)
    watch_path.mkdir(parents=True, exist_ok=True)
    seen_files = set(watch_path.iterdir())

    print(f"  👁️  监听目录: {watch_path}")
    print(f"  📡 等待新文件... (Ctrl+C 停止)")

    while worker.running:
        try:
            current_files = set(watch_path.iterdir())
            new_files = current_files - seen_files

            for f in sorted(new_files):
                if f.suffix in (".json", ".csv", ".txt"):
                    print(f"  📄 新文件: {f.name}")
                    process_incoming_file(f, worker, node_id)
                    seen_files.add(f)

            time.sleep(1)
        except KeyboardInterrupt:
            break


def process_incoming_file(filepath, worker, node_id):
    """处理新落盘的文件"""
    if filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                worker.submit(raw_to_standard_input(item, node_id))
        else:
            worker.submit(raw_to_standard_input(data, node_id))
    else:
        # CSV/TXT: 逐行
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                headers = [h.strip() for h in headers]
                for row in reader:
                    readings = {}
                    for i, val in enumerate(row):
                        if i < len(headers):
                            try:
                                readings[headers[i]] = float(val.strip())
                            except (ValueError, AttributeError):
                                pass
                    if readings:
                        raw = {"readings": readings}
                        worker.submit(raw_to_standard_input(raw, node_id))


# ══════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="边端实时数据流适配器")
    parser.add_argument("--mode", choices=["tcp", "simulate", "watch"],
                        default="simulate", help="数据流模式")
    parser.add_argument("--data", default=None, help="模拟模式的数据文件")
    parser.add_argument("--port", type=int, default=9000, help="TCP 监听端口")
    parser.add_argument("--watch-dir", default="./incoming_data", help="监听目录")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="llama-server 地址")
    parser.add_argument("--interval", type=float, default=0.05, help="模拟流发送间隔(秒)")
    parser.add_argument("--node", default="edge-stream-01", help="节点ID")
    parser.add_argument("--duration", type=int, default=0, help="运行时长(秒), 0=无限")
    args = parser.parse_args()

    print("=" * 60)
    print("  📡 边端实时数据流适配器")
    print(f"  模式: {args.mode}")
    print(f"  推理服务: {args.endpoint}")
    print(f"  节点: {args.node}")
    print("=" * 60)

    # 启动推理工作线程
    worker = InferenceWorker(args.endpoint)
    worker.start()
    print(f"  ✅ 推理工作线程已启动")

    try:
        if args.mode == "tcp":
            run_tcp_mode(args.port, worker, args.node)
        elif args.mode == "simulate":
            if not args.data:
                print("  ❌ 模拟模式需要 --data 参数")
                return
            run_simulate_mode(args.data, worker, args.node, args.interval)
        elif args.mode == "watch":
            run_watch_mode(args.watch_dir, worker, args.node)

        # 等待队列消化
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n  ⏹️  停止中...")
    finally:
        worker.stop()

    # 最终统计
    stats = worker.get_stats()
    print(f"\n{'='*60}")
    print(f"  📊 运行统计")
    print(f"{'='*60}")
    print(f"  总请求:     {stats['total']}")
    print(f"  成功:       {stats['success']}")
    print(f"  失败:       {stats['failed']}")
    print(f"  平均时延:   {stats['avg_latency_ms']} ms")
    print(f"  结果缓存:   {stats['results_cached']} 条")
    print(f"  业务保持率: {stats['success']/max(stats['total'],1)*100:.1f}%")

    # 保存结果
    result_file = SCRIPT_DIR / "stream_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "results": list(worker.results)[-100:]},
                  f, ensure_ascii=False, indent=2)
    print(f"  📄 结果: {result_file}")


if __name__ == "__main__":
    main()
