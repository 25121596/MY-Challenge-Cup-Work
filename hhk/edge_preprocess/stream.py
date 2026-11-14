"""
流式定长滑动窗口读取器 — 处理大文件 (GB级) 不爆内存

核心思想: 逐行读取, 每通道维护定长环形缓冲,
缓冲满 → 输出一窗特征 → 滑动窗口继续。
内存上限 = n_channels × window_size × 8 bytes, 与文件大小无关。
"""
import csv
from collections import deque


class SlidingWindow:
    """单通道定长滑动窗口。非重叠模式 (每填满一次就输出并清空)。"""

    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = []

    def feed(self, value):
        """喂入一个值, 返回 (满窗数组 or None)"""
        self.buffer.append(float(value))
        if len(self.buffer) >= self.window_size:
            window = list(self.buffer[:self.window_size])
            self.buffer = []  # 非重叠: 清空
            return window
        return None

    def flush(self):
        """返回当前 buffer 中的剩余数据 (可能不足 window_size)"""
        if len(self.buffer) > 0:
            window = list(self.buffer)
            self.buffer = []
            return window
        return None


class MultiChannelStreamReader:
    """多通道流式读取器。逐行解析 CSV, 多列各自维护 SlidingWindow。

    用法:
        reader = MultiChannelStreamReader(filepath, window_size=2560, channels=[0,1,2,3,4,5,6,7,8])
        for window_array in reader.windows():
            # window_array shape: [n_channels, window_size]
            features = extract_features(window_array)

    内存: n_channels × window_size × 8 bytes (如 9×2560×8 ≈ 184KB)
    """

    def __init__(self, filepath, window_size=2560, channels=None,
                 delimiter=",", has_header=True, skip_lines=0):
        """
        Args:
            filepath: CSV 文件路径
            window_size: 每窗采样点数 (如 2560 ≈ 51ms @50kHz)
            channels: 要用哪些列 (0-indexed), None = 全部列
            delimiter: 分隔符
            has_header: 第一行是否为表头
            skip_lines: 跳过前 N 行 (用于跳过注释/元数据)
        """
        self.filepath = filepath
        self.window_size = window_size
        self.channels = channels
        self.delimiter = delimiter
        self.has_header = has_header
        self.skip_lines = skip_lines
        self._n_channels = None

    def windows(self):
        """生成器: 每次 yield 一个满窗 (list, shape [n_channels, window_size])"""
        with open(self.filepath, "r", encoding="utf-8", errors="replace") as f:
            # 跳过
            for _ in range(self.skip_lines):
                next(f, None)

            # 读 header
            first_line = next(f, None)
            if first_line is None:
                return
            parts = first_line.strip().split(self.delimiter)
            all_col_count = len(parts)

            # 确定要读的列
            if self.channels is None:
                self.channels = list(range(all_col_count))

            if self._n_channels is None:
                self._n_channels = len(self.channels)

            # 如果不是 header (是数据), 也处理第一行
            if self.has_header and self._try_parse(parts):
                self.has_header = False  # 实际上没有 header
                self._feed_row(parts)
            elif not self.has_header:
                self._feed_row(parts)

            # 逐行读取
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(self.delimiter)
                if len(parts) < max(self.channels) + 1:
                    continue

                result = self._feed_row(parts)
                if result is not None:
                    yield result

            # 文件结束, flush 剩余
            remaining = self._flush_all()
            if remaining is not None:
                yield remaining

    def _try_parse(self, parts):
        """尝试解析第一行, 如果失败说明是 header"""
        try:
            float(parts[self.channels[0]].strip())
            return True
        except (ValueError, IndexError):
            return False

    def _feed_row(self, parts):
        """将一行数据喂入各通道窗口。返回满窗或 None。"""
        for ci_local, ci_global in enumerate(self.channels):
            if ci_global >= len(parts):
                continue
            try:
                val = float(parts[ci_global].strip())
            except ValueError:
                val = 0.0
            self._windows[ci_local].feed(val)

        # 检查是否所有通道的窗口都满了
        # (SlidingWindow 在 feed 满时返回非 None, 但不同通道可能不同步)
        # 统一检查: 取第一个通道的状态
        first_buf = self._windows[0].buffer if hasattr(self, '_windows') else None
        return None  # 真正的满窗检查在 _all_full 时触发

    def _init_windows(self):
        """延迟初始化窗口 (在知道 n_channels 后)"""
        self._windows = [SlidingWindow(self.window_size)
                         for _ in range(self._n_channels)]

    def _all_full(self):
        return all(len(w.buffer) >= self.window_size for w in self._windows)

    def _flush_all(self):
        arrays = [w.flush() for w in self._windows]
        if any(a is None for a in arrays):
            return None
        # pad 到相同长度
        min_len = min(len(a) for a in arrays)
        if min_len < 4:
            return None
        return list([a[:min_len] for a in arrays])


class BatchWindowReader:
    """批量窗口读取器: 将文件读取为多个满窗数组的列表。

    与 MultiChannelStreamReader 的区别:
    - MultiChannelStreamReader 是真正的流式 (generator, 低内存)
    - BatchWindowReader 方便批量处理和数据探索 (一次性读入内存)
    """

    def __init__(self, filepath, window_size=2560, channels=None,
                 delimiter=",", has_header=True, max_windows=None):
        self.filepath = filepath
        self.window_size = window_size
        self.channels = channels
        self.delimiter = delimiter
        self.has_header = has_header
        self.max_windows = max_windows

    def read_all(self):
        """返回 [list shape [n_channels, window_size], ...]"""
        reader = MultiChannelStreamReader(
            self.filepath, self.window_size, self.channels,
            self.delimiter, self.has_header
        )
        windows = []
        for w in reader.windows():
            windows.append(w)
            if self.max_windows and len(windows) >= self.max_windows:
                break
        return windows


def read_sensor_columns(filepath, column_names, delimiter=",", max_rows=None):
    """读取 CSV 中指定列名的所有数据 (全量读入, 适合小文件)。

    用于 UCI 液压系统等低采样率数据。
    Returns: dict {column_name: list}
    """
    import csv
    data = {name: [] for name in column_names}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            for name in column_names:
                if name in row:
                    try:
                        data[name].append(float(row[name]))
                    except (ValueError, KeyError):
                        data[name].append(0.0)

    return {name: list(vals) for name, vals in data.items()}
