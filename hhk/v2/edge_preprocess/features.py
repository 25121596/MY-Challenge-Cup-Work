"""
边侧信号特征提取 — 12 时域 (纯 Python) + 5 频域 (numpy 可选)
零重依赖: 纯标准库可运行, numpy 可选增强频域特征
"""
import math

# numpy 可选
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False


# ══════════════════════════════════════════════════════
#  时域特征 (12 维) — 纯 Python, 零依赖
# ══════════════════════════════════════════════════════

def _to_list(x):
    """确保输入是 Python list[float]"""
    if np is not None and isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def compute_time_features(x):
    """对一维信号 x 提取 12 个时域特征。纯 Python 实现。"""
    vals = _to_list(x)
    n = len(vals)
    if n == 0:
        return {f: 0.0 for f in TIME_FEATURE_NAMES}

    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(variance)

    rms_val = math.sqrt(sum(v ** 2 for v in vals) / n)
    abs_mean = sum(abs(v) for v in vals) / n
    peak = max(abs(v) for v in vals)
    min_val = min(vals)
    max_val = max(vals)

    if std > 1e-12:
        kurtosis = sum((v - mean) ** 4 for v in vals) / n / (std ** 4) - 3.0
        skewness = sum((v - mean) ** 3 for v in vals) / n / (std ** 3)
        crest_factor = peak / rms_val if rms_val > 1e-12 else 0.0
        shape_factor = rms_val / abs_mean if abs_mean > 1e-12 else 0.0
    else:
        kurtosis = skewness = crest_factor = shape_factor = 0.0

    impulse_factor = peak / abs_mean if abs_mean > 1e-12 else 0.0

    # 过零率
    zcr = 0.0
    if n > 1:
        crossings = 0
        prev_sign = 1 if vals[0] >= 0 else -1
        for v in vals[1:]:
            cur_sign = 1 if v >= 0 else -1
            if cur_sign != prev_sign:
                crossings += 1
            prev_sign = cur_sign
        zcr = crossings / (2.0 * (n - 1))

    return {
        "mean": mean, "std": std, "min": min_val, "max": max_val,
        "peak2peak": max_val - min_val, "rms": rms_val,
        "kurtosis": kurtosis, "skewness": skewness,
        "crest_factor": crest_factor, "shape_factor": shape_factor,
        "impulse_factor": impulse_factor, "zcr": zcr,
    }


# ══════════════════════════════════════════════════════
#  频域特征 (5 维) — numpy 可选
# ══════════════════════════════════════════════════════

def compute_freq_features(x, fs=1.0):
    """对一维信号 x 提取 5 个频域特征。

    需要 numpy, 否则返回全 0 (仅时域特征可用)。
    """
    if not _HAS_NUMPY:
        return {f: 0.0 for f in FREQ_FEATURE_NAMES}

    x_arr = np.asarray(x, dtype=np.float64)
    n = len(x_arr)
    if n < 4:
        return {f: 0.0 for f in FREQ_FEATURE_NAMES}

    X = np.fft.rfft(x_arr)
    mag = np.abs(X)
    power = mag ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    peak_idx = int(np.argmax(mag))
    dominant_freq = float(freqs[peak_idx])
    spectral_energy = float(np.sum(power))

    mag_sum = float(np.sum(mag))
    spectral_centroid = float(np.sum(freqs * mag) / mag_sum) if mag_sum > 1e-12 else 0.0

    energy_sum = float(spectral_energy)
    if energy_sum > 1e-12:
        prob = power / energy_sum
        prob = prob[prob > 0]
        spectral_entropy = float(-np.sum(prob * np.log2(prob)))
    else:
        spectral_entropy = 0.0

    cumsum = np.cumsum(power)
    total = float(cumsum[-1])
    if total > 1e-12:
        rolloff_idx = int(np.searchsorted(cumsum, 0.85 * total))
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        spectral_rolloff = float(freqs[rolloff_idx])
    else:
        spectral_rolloff = 0.0

    return {
        "dominant_freq": dominant_freq,
        "spectral_energy": spectral_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "spectral_rolloff": spectral_rolloff,
    }


# ══════════════════════════════════════════════════════
#  统一接口
# ══════════════════════════════════════════════════════

TIME_FEATURE_NAMES = [
    "mean", "std", "min", "max", "peak2peak", "rms",
    "kurtosis", "skewness", "crest_factor", "shape_factor",
    "impulse_factor", "zcr",
]

FREQ_FEATURE_NAMES = [
    "dominant_freq", "spectral_energy", "spectral_centroid",
    "spectral_entropy", "spectral_rolloff",
]

ALL_FEATURE_NAMES = TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES  # 17 维


def extract_channel_features(signal, fs=1.0):
    """对单通道信号提取特征 (12 时域 + 5 频域 if numpy)。

    Returns:
        (feature_values: list[17 floats], feature_names: list[17 str])
    """
    time_feats = compute_time_features(signal)
    freq_feats = compute_freq_features(signal, fs)

    values = [time_feats[k] for k in TIME_FEATURE_NAMES] + \
             [freq_feats[k] for k in FREQ_FEATURE_NAMES]

    return values, list(ALL_FEATURE_NAMES)


def extract_multichannel_features(channels, fs=1.0):
    """对多通道信号提取拼接特征向量。

    Args:
        channels: list of 1D lists/arrays, shape [n_channels, n_samples]
        fs: 采样率 Hz

    Returns:
        (flat_features, feature_names, channel_count, total_dim)
    """
    flat_features = []
    flat_names = []
    for ci, signal in enumerate(channels):
        vals, names = extract_channel_features(signal, fs)
        flat_features.extend(vals)
        for name in names:
            flat_names.append(f"ch{ci}_{name}")

    return flat_features, flat_names, len(channels), len(flat_features)


def has_numpy():
    """检查 numpy 是否可用"""
    return _HAS_NUMPY
