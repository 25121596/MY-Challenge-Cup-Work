"""
视频/摄像头 → 数值特征 → 标准 readings → 馈入现有管线

零侵入: 产出的 JSON 与 sensor_csv_to_input.py 完全一致,
       可直接被 stream_adapter.py (TCP) / edge_run.py (文件) / edge_inference.py 消费。

特征提取策略 (每帧/每N帧):
  - motion_score     帧间差分均值 → 检测运动异常
  - brightness_mean/std  全局亮度统计
  - roi_N_mean       网格分区平均亮度 (默认 4x4=16 区域)
  - edge_density     Canny 边缘像素占比 → 纹理/复杂度
  - foreground_ratio 简单背景减除前景占比

输出格式 (与 sensor_csv_to_input.py 泛化模式一致):
  {
    "task_id": "T-VIDEO-...",
    "node_id": "edge-camera-01",
    "task_type": "generic_anomaly",
    "data": {
      "device_type": "视频监控",
      "readings": {"motion_score": 0.12, "brightness_mean": 128.5, ...},
      "generic_mode": true,
      "context": "视频特征, 10帧平均, 分辨率640x480"
    }
  }

用法:
  # 视频文件 → JSON → edge_run.py 消费
  python video_adapter.py --input video.mp4 --output features.json
  python edge_run.py --data features.json

  # 摄像头 → TCP → stream_adapter.py 消费 (终端1)
  python stream_adapter.py --mode tcp --port 9000
  # 终端2
  python video_adapter.py --camera 0 --mode tcp --port 9000

  # 直接推理模式 (不经过中间文件/TCP)
  python video_adapter.py --input video.mp4 --mode direct --endpoint http://127.0.0.1:8080

  # RTSP/RTMP 网络摄像头流 (无限流建议配 --max-frames 限制)
  python video_adapter.py --input rtsp://192.168.1.100:554/stream --mode direct --max-frames 300
"""

import sys, os, json, time, argparse, socket, math, threading
from pathlib import Path
from collections import deque

# ══════════════════════════════════════════════════════
#  可选依赖检查
# ══════════════════════════════════════════════════════
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import urllib.request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False


# ══════════════════════════════════════════════════════
#  特征提取
# ══════════════════════════════════════════════════════

def _to_float(val):
    """安全转 float"""
    if val is None:
        return 0.0
    return round(float(val), 6)


class VideoFeatureExtractor:
    """从视频帧中提取数值特征, 不依赖任何视觉模型/CNN。

    所有特征都是标量数值, 直接作为 readings 进入现有传感器管线。
    """

    def __init__(self, grid_rows=4, grid_cols=4, resize_to=(320, 240)):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.resize_to = resize_to  # 先缩放到此分辨率再提取 (加速)
        self._prev_gray = None
        self._bg_model = None       # 简单滑动平均背景
        self._bg_alpha = 0.05       # 背景更新速率
        self._frame_count = 0

    def extract(self, frame_bgr):
        """从一帧 BGR 图像中提取数值特征。

        Returns:
            dict: {feature_name: float, ...}
        """
        if frame_bgr is None:
            return self._empty_features()

        h, w = frame_bgr.shape[:2]

        # 缩放加速
        if self.resize_to and (w > self.resize_to[0] or h > self.resize_to[1]):
            frame_bgr = cv2.resize(frame_bgr, self.resize_to)
            h, w = frame_bgr.shape[:2]

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        feats = {}
        self._frame_count += 1

        # ── 1. 运动检测 (帧差) ──
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            feats["motion_score"] = _to_float(np.mean(diff))
            feats["motion_max"] = _to_float(np.max(diff))
            feats["motion_std"] = _to_float(np.std(diff))
        else:
            feats["motion_score"] = 0.0
            feats["motion_max"] = 0.0
            feats["motion_std"] = 0.0
        self._prev_gray = gray.copy()

        # ── 2. 全局亮度 ──
        feats["brightness_mean"] = _to_float(np.mean(gray))
        feats["brightness_std"] = _to_float(np.std(gray))
        feats["brightness_min"] = _to_float(np.min(gray))
        feats["brightness_max"] = _to_float(np.max(gray))

        # ── 3. 网格分区 (空间信息, 不依赖CNN) ──
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        for ri in range(self.grid_rows):
            for ci in range(self.grid_cols):
                y1, y2 = ri * cell_h, (ri + 1) * cell_h
                x1, x2 = ci * cell_w, (ci + 1) * cell_w
                cell = gray[y1:y2, x1:x2]
                feats[f"roi_r{ri}c{ci}_mean"] = _to_float(np.mean(cell))
                if ri == 0 and ci == 0:  # 只对第一个区域存 std (控制维度)
                    pass  # 网格已足够

        # ── 4. 边缘密度 (纹理/复杂度) ──
        edges = cv2.Canny(gray, 50, 150)
        feats["edge_density"] = _to_float(np.count_nonzero(edges) / edges.size)

        # ── 5. 简单前景检测 (滑动平均背景减除) ──
        gray_f = gray.astype(np.float32)
        if self._bg_model is None:
            self._bg_model = gray_f.copy()
        else:
            self._bg_model = self._bg_alpha * gray_f + (1 - self._bg_alpha) * self._bg_model

        if self._frame_count > 10:  # 等背景稳定
            fg_mask = np.abs(gray_f - self._bg_model) > 25
            feats["foreground_ratio"] = _to_float(np.count_nonzero(fg_mask) / fg_mask.size)
        else:
            feats["foreground_ratio"] = 0.0

        # ── 6. 色彩统计 (HSV) ──
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        for ci, ch_name in enumerate(["h", "s", "v"]):
            ch = hsv[:, :, ci]
            feats[f"{ch_name}_mean"] = _to_float(np.mean(ch))
            feats[f"{ch_name}_std"] = _to_float(np.std(ch))

        return feats

    def _empty_features(self):
        """返回零值特征 (视频读取失败时)"""
        feats = {
            "motion_score": 0.0, "motion_max": 0.0, "motion_std": 0.0,
            "brightness_mean": 0.0, "brightness_std": 0.0,
            "brightness_min": 0.0, "brightness_max": 0.0,
            "edge_density": 0.0, "foreground_ratio": 0.0,
            "h_mean": 0.0, "h_std": 0.0, "s_mean": 0.0, "s_std": 0.0,
            "v_mean": 0.0, "v_std": 0.0,
        }
        for ri in range(self.grid_rows):
            for ci in range(self.grid_cols):
                feats[f"roi_r{ri}c{ci}_mean"] = 0.0
        return feats

    def reset_background(self):
        """重置背景模型 (场景切换时调用)"""
        self._bg_model = None
        self._frame_count = 0


def batch_features(frame_features, batch_size=30):
    """将逐帧特征聚合为批量读数 (每 batch_size 帧取平均)。

    输入: [{"motion_score": 0.1, ...}, ...]  逐帧
    输出: {"motion_score": 0.12, ...}          批平均
    """
    if not frame_features:
        return {}
    keys = frame_features[0].keys()
    batched = {}
    for key in keys:
        vals = [f[key] for f in frame_features if key in f]
        if vals:
            batched[key] = round(sum(vals) / len(vals), 6)
    return batched


def extract_features_from_source(source, camera_id=None, frame_skip=3,
                                  batch_size=30, max_frames=None, resize_to=(320, 240)):
    """从视频源迭代提取批量特征。

    Args:
        source: 视频文件路径 (None 表示摄像头)
        camera_id: 摄像头 ID (source 为 None 时使用)
        frame_skip: 每 N 帧取一帧 (降采样)
        batch_size: 每批聚合多少帧
        max_frames: 最大处理帧数
        resize_to: 缩放分辨率 (加速)

    Yields:
        (batch_idx, readings_dict, metadata)
    """
    if not HAS_CV2:
        raise ImportError("需要 opencv-python: pip install opencv-python")

    if source:
        cap = cv2.VideoCapture(source)
        s = str(source)
        if "://" in s:
            # 网络流 (RTSP/RTMP/HTTP): 从 URL 安全提取主机名作为设备名
            host = s.split("://", 1)[1].split("/")[0].split("@")[-1].split(":")[0]
            device_name = f"stream_{host or 'remote'}"
        else:
            device_name = Path(source).stem
    else:
        cap = cv2.VideoCapture(camera_id if camera_id is not None else 0)
        device_name = f"camera_{camera_id or 0}"

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source or f'camera {camera_id}'}")

    extractor = VideoFeatureExtractor(resize_to=resize_to)
    frame_features = []
    batch_idx = 0
    frame_idx = 0
    total_frames = 0

    try:
        while True:
            if max_frames and total_frames >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # 降采样
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            feats = extractor.extract(frame)
            frame_features.append(feats)

            # 凑齐一个 batch
            if len(frame_features) >= batch_size:
                readings = batch_features(frame_features)
                fps = cap.get(cv2.CAP_PROP_FPS)
                ts = total_frames / max(fps, 1.0) if fps > 0 else total_frames / 30.0
                meta = {
                    "batch_index": batch_idx,
                    "frame_start": total_frames - len(frame_features) * frame_skip,
                    "frame_end": total_frames,
                    "timestamp_sec": round(ts, 1),
                    "fps": round(fps, 1),
                    "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                    "source": device_name,
                }
                yield batch_idx, readings, meta
                frame_features = []
                batch_idx += 1

        # flush 剩余帧
        if frame_features:
            readings = batch_features(frame_features)
            yield batch_idx, readings, {"batch_index": batch_idx, "source": device_name}

    finally:
        cap.release()


# ══════════════════════════════════════════════════════
#  输出: 标准 input.json 格式 → 馈入现有管线
# ══════════════════════════════════════════════════════

def build_standard_input(readings, metadata, node_id="edge-camera-01"):
    """将视频特征包装为标准 input.json 格式。

    与 sensor_csv_to_input.py 泛化模式输出完全一致，
    下游 edge_inference.py / edge_run.py 无需任何修改。
    """
    batch_idx = metadata.get("batch_index", 0)
    source = metadata.get("source", "video")

    # 构建上下文 (给 LLM 的统计基线说明)
    context_lines = [
        f"数据源: 视频特征 ({source})",
        f"分辨率: {metadata.get('resolution', 'unknown')}",
        f"帧率: {metadata.get('fps', '?')} fps",
        f"批次: {batch_idx}, 时间: {metadata.get('timestamp_sec', '?')}s",
        "",
        "特征说明:",
        "  motion_score: 帧间差分均值 (越高=运动越剧烈)",
        "  brightness_mean/std: 全局亮度统计 (突变=光照异常/遮挡)",
        "  roi_*_mean: 图像分区平均亮度 (局部变化检测)",
        "  edge_density: 边缘像素占比 (纹理/复杂度)",
        "  foreground_ratio: 前景占比 (运动目标检测)",
        "  h/s/v_mean/std: 色彩统计 (色调/饱和度/明度)",
    ]

    return {
        "task_id": f"T-VIDEO-{source}-{batch_idx:05d}",
        "node_id": node_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task_type": "generic_anomaly",
        "priority": "high",
        "data": {
            "device_id": source,
            "device_type": "视频监控",
            "readings": readings,
            "context": "\n".join(context_lines),
            "generic_mode": True,
            "source": "video_adapter",
        },
    }


# ══════════════════════════════════════════════════════
#  输出模式
# ══════════════════════════════════════════════════════

def run_file_mode(source, camera_id, output_path, **kwargs):
    """输出为标准 JSON 文件 → edge_run.py --data 直接消费"""
    inputs = []
    for batch_idx, readings, meta in extract_features_from_source(
        source, camera_id, **kwargs
    ):
        entry = build_standard_input(readings, meta)
        inputs.append(entry)
        if (batch_idx + 1) % 10 == 0:
            print(f"  [batch {batch_idx}] {len(readings)} 特征, "
                  f"motion={readings.get('motion_score', 0):.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)

    print(f"\n  [video_adapter] {len(inputs)} 条标准输入 -> {output_path}")
    print(f"  [下一步] python edge_run.py --data {output_path}")
    return output_path


def run_tcp_mode(source, camera_id, port, host="127.0.0.1", **kwargs):
    """TCP 推送 JSON 行 → stream_adapter.py TCP 模式直接消费

    用法 (终端1): python stream_adapter.py --mode tcp --port 9000
    用法 (终端2): python video_adapter.py --camera 0 --mode tcp --port 9000
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"  [video_adapter] 已连接 stream_adapter TCP {host}:{port}")

    try:
        for batch_idx, readings, meta in extract_features_from_source(
            source, camera_id, **kwargs
        ):
            entry = build_standard_input(readings, meta)
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            sock.sendall(line.encode("utf-8"))
            if (batch_idx + 1) % 10 == 0:
                print(f"  [batch {batch_idx}] 已推送, "
                      f"motion={readings.get('motion_score', 0):.4f}")
    finally:
        sock.close()
        print(f"  [video_adapter] 推送完成, 共 {batch_idx + 1} 批次")


def run_direct_mode(source, camera_id, endpoint, **kwargs):
    """直接调用 llama-server 推理 (不经过中间文件/TCP)

    需要 llama-server 已启动。
    """
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR / "edge-io-protocol"))
    import edge_inference

    edge_inference.LLAMA_SERVER = endpoint
    results = []

    for batch_idx, readings, meta in extract_features_from_source(
        source, camera_id, **kwargs
    ):
        entry = build_standard_input(readings, meta)
        try:
            result = edge_inference.infer(entry, use_grammar=True)
            out = result["output"]
            results.append(out)

            sev = out["decision"]["category"]
            sev_name = out["decision"]["category_name"]
            conf = out["confidence"]
            lat = out["latency_ms"]
            escalate = "->CLOUD" if out.get("escalate_to_cloud") else ""
            print(f"  [batch {batch_idx}] severity={sev}({sev_name}) "
                  f"conf={conf:.0%} lat={lat:.0f}ms {escalate}")
        except Exception as e:
            print(f"  [batch {batch_idx}] ERROR: {e}")
            results.append({"task_id": entry["task_id"], "error": str(e)})

    # 汇总
    if results:
        ok = [r for r in results if "error" not in r]
        lats = [r["latency_ms"] for r in ok]
        sevs = [r["decision"]["category"] for r in ok]
        from collections import Counter

        print(f"\n  {'='*50}")
        print(f"  [video_adapter] 推理完成: {len(results)} 批次")
        print(f"  成功: {len(ok)}/{len(results)}")
        if lats:
            print(f"  时延: avg={sum(lats)/len(lats):.0f}ms, max={max(lats):.0f}ms")
            print(f"  严重度分布: {dict(Counter(sevs))}")
            print(f"  甩云率: {sum(1 for r in ok if r.get('escalate_to_cloud'))/len(ok)*100:.0f}%")


# ══════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="视频/摄像头 -> 数值特征 -> 标准 readings -> 现有推理管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 视频文件 → JSON → edge_run.py 消费
  python video_adapter.py --input video.mp4 --output features.json
  python edge_run.py --data features.json

  # 摄像头 → TCP → stream_adapter.py 消费
  # 终端1: python stream_adapter.py --mode tcp --port 9000
  # 终端2: python video_adapter.py --camera 0 --mode tcp --port 9000

  # 直接推理 (需要 llama-server 在 8080)
  python video_adapter.py --input video.mp4 --mode direct
        """,
    )
    # 输入源
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--input", "-i", default=None,
                           help="视频文件路径, 或网络流URL (rtsp:// rtmp:// http://)")
    src_group.add_argument("--camera", "-c", type=int, default=None, help="摄像头ID (0=默认)")

    # 输出模式
    parser.add_argument("--mode", choices=["file", "tcp", "direct"], default="file",
                        help="输出模式: file(JSON文件,默认) / tcp(TCP推送) / direct(直接推理)")
    parser.add_argument("--output", "-o", default="video_features.json", help="file模式输出路径")
    parser.add_argument("--port", "-p", type=int, default=9000, help="tcp模式端口 (默认9000)")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080", help="direct模式 llama-server 地址")

    # 处理参数
    parser.add_argument("--frame-skip", type=int, default=3, help="每N帧取一帧 (默认3)")
    parser.add_argument("--batch-size", type=int, default=30, help="每批聚合帧数 (默认30)")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")
    parser.add_argument("--node", default="edge-camera-01", help="节点ID")

    args = parser.parse_args()

    if not HAS_CV2:
        print("错误: 需要 opencv-python")
        print("  pip install opencv-python")
        return

    print("=" * 60)
    print("  [video_adapter] 视频特征提取 -> 现有推理管线")
    print(f"  源: {args.input or f'摄像头 #{args.camera}'}")
    print(f"  模式: {args.mode}")
    print(f"  降采样: 每{args.frame_skip}帧, 批大小: {args.batch_size}帧")
    print("=" * 60)

    extract_kw = dict(
        frame_skip=args.frame_skip,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
    )

    if args.mode == "file":
        run_file_mode(args.input, args.camera, args.output, **extract_kw)
    elif args.mode == "tcp":
        run_tcp_mode(args.input, args.camera, args.port, **extract_kw)
    elif args.mode == "direct":
        run_direct_mode(args.input, args.camera, args.endpoint, **extract_kw)


if __name__ == "__main__":
    main()
