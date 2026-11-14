#pragma once

// ── 边端→云端协同推理客户端 ────────────────────────────────
//
// 基于 LLaMA* 架构（动态头数扩展），支持三种传输模式：
//   1. Split Inference  — 边端跑前 K 层，发送中间 hidden states
//   2. Query Offload    — 纯文本卸载（cloud.cpp 升级版）
//   3. Feature Offload  — 多模态特征传输（对接 media_source.h）
//
// 协议统一采用 JSON + SSE 流式响应（OpenAI 兼容风格扩展）。

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ═══════════════════════════════════════════════════════════════
// 枚举与常量
// ═══════════════════════════════════════════════════════════════

// 传输模式
enum class EdgeCloudMode {
    SplitInference,   // 中间层 hidden states 传输（LLaMA* 核心）
    QueryOffload,     // 纯文本卸载（兼容现有 cloud.cpp 逻辑）
    FeatureOffload,   // 多模态特征传输
    HealthCheck       // 健康检查 + 延迟探测
};

// 支持的数值精度
enum class DataDtype {
    Float32,
    Float16,
    BFloat16,
    Int8,
};

inline const char * dtype_str(DataDtype dt) {
    switch (dt) {
        case DataDtype::Float32:  return "float32";
        case DataDtype::Float16:  return "float16";
        case DataDtype::BFloat16: return "bfloat16";
        case DataDtype::Int8:     return "int8";
    }
    return "float16";
}

inline size_t dtype_bytes(DataDtype dt) {
    switch (dt) {
        case DataDtype::Float32:  return 4;
        case DataDtype::Float16:  return 2;
        case DataDtype::BFloat16: return 2;
        case DataDtype::Int8:     return 1;
    }
    return 2;
}

// ═══════════════════════════════════════════════════════════════
// 数据结构体
// ═══════════════════════════════════════════════════════════════

// ── 中间层输出（Split Inference 核心载荷） ─────────────────

struct HiddenStates {
    int         split_layer   = 12;      // 在模型的第几层切分
    int         current_heads = 12;      // 当前边端实际使用的 head 数
    int         head_dim      = 128;     // 每个 head 的维度
    int         batch_size    = 1;
    int         seq_len       = 0;
    DataDtype   dtype         = DataDtype::Float16;

    // 实际数据 = batch_size * seq_len * (current_heads * head_dim)
    std::vector<uint8_t> data;           // 二进制 blob（已序列化为连续内存）
    size_t      data_bytes   = 0;

    // 校验
    std::string checksum_md5;

    // 便捷方法
    size_t total_elements() const {
        return (size_t)batch_size * seq_len * current_heads * head_dim;
    }
    size_t total_bytes() const {
        return total_elements() * dtype_bytes(dtype);
    }
    bool   valid() const {
        return !data.empty() && data.size() == total_bytes();
    }
};

// ── 推理上下文（云端继续推理需要的信息） ──────────────────

struct InferenceContext {
    std::vector<int32_t> prompt_tokens;  // 原始输入 token ID 序列
    int32_t     n_past          = 0;     // 已处理的 token 总数
    float       temperature     = 0.7f;
    float       top_p           = 0.9f;
    float       min_p           = 0.1f;
    int32_t     top_k           = 40;
    float       repeat_penalty  = 1.0f;
    int32_t     max_new_tokens  = 512;
    uint32_t    seed            = 42;
    std::vector<std::string> stop_strings;

    // 可选：KV Cache 二进制快照（大文件，谨慎使用）
    bool        has_kv_cache    = false;
    std::vector<uint8_t> kv_cache_data;
};

// ── 边端设备信息（云端调度 & 自适应决策） ──────────────────

struct EdgeDeviceInfo {
    std::string device_id;
    std::string device_type;     // "jetson-orin-nano", "raspberry-pi-5", etc.
    int         battery_pct   = -1;   // -1 = 未知
    bool        on_ac         = true;
    int         avail_mem_mb  = 0;
    double      current_tps   = 0.0;  // 当前推理速度
    int         network_rtt_ms = 0;   // 到云端的网络延迟
};

// ── 多模态特征（Feature Offload 模式） ────────────────────

struct MultimodalFeatures {
    std::string media_type;          // "image", "video_frame", "audio"
    int         width        = 0;
    int         height       = 0;
    int         channels     = 3;
    std::string original_mime;       // "image/jpeg", "video/mp4"
    int64_t     timestamp_us = 0;

    // 边端提取的特征（如 Vision Encoder 中间层输出）
    std::vector<float> visual_features;
    std::vector<int64_t> feature_shape; // [num_patches, feature_dim]

    // 或原始数据（让云端自己编码）
    std::vector<uint8_t> raw_media_bytes;
};

// ═══════════════════════════════════════════════════════════════
// 回调类型
// ═══════════════════════════════════════════════════════════════

// 流式 token 回调
// token_text — 新生成的 token 文本
// finished   — 推理是否结束
// error_msg  — 错误信息（空 = 正常）
using edge_cloud_token_cb = std::function<void(
    const std::string & token_text,
    bool finished,
    const std::string & error_msg)>;

// 云端性能统计回调
struct CloudPerf {
    bool   valid        = false;
    double ttft_ms      = 0.0;   // Time-To-First-Token（云端）
    double tps          = 0.0;   // Tokens Per Second（云端）
    int    total_tokens = 0;
    int    cloud_layers = 0;     // 云端实际运行的层数
    double upload_ms    = 0.0;   // 数据上传耗时
    double download_ms  = 0.0;   // 结果下载耗时
};
using edge_cloud_perf_cb = std::function<void(const CloudPerf &)>;

// ═══════════════════════════════════════════════════════════════
// EdgeCloudClient — 边端协同客户端
// ═══════════════════════════════════════════════════════════════

class EdgeCloudClient {
public:
    EdgeCloudClient();
    ~EdgeCloudClient();

    // ── 配置 ─────────────────────────────────────────────────

    void set_endpoint(const std::string & base_url);  // e.g. "http://192.168.1.100:8080"
    void set_api_key(const std::string & key);
    void set_model(const std::string & model_id);
    void set_timeout(int seconds);
    void set_dtype(DataDtype dtype);

    // ── 健康检查 ─────────────────────────────────────────────

    // 快速探测云端是否可达 + 返回延迟
    struct HealthStatus {
        bool   reachable     = false;
        int    latency_ms    = -1;
        int    cloud_load    = -1;     // 0-100, 云端当前负载
        std::string cloud_version;
        int    max_supported_split_layer = 32;
    };
    HealthStatus check_health();

    // ── 核心接口 ─────────────────────────────────────────────

    // 模式 1: Split Inference — 发送 hidden states
    void split_infer(
        const HiddenStates    & hidden,
        const InferenceContext & ctx,
        const EdgeDeviceInfo  & device,
        edge_cloud_token_cb     on_token,
        edge_cloud_perf_cb      on_perf = nullptr);

    // 模式 2: Query Offload — 发送文本消息（升级版）
    void query_offload(
        const std::string     & messages_json,       // OpenAI 格式
        const InferenceContext & ctx,
        edge_cloud_token_cb     on_token,
        edge_cloud_perf_cb      on_perf = nullptr);

    // 模式 3: Feature Offload — 发送多模态特征
    void feature_offload(
        const std::string      & text_prompt,
        const MultimodalFeatures & media,
        const InferenceContext  & ctx,
        edge_cloud_token_cb      on_token,
        edge_cloud_perf_cb       on_perf = nullptr);

    // ── 控制 ─────────────────────────────────────────────────

    void stop();
    bool is_running() const;
    bool is_reachable() const { return _reachable.load(); }

    // ── 自适应决策辅助 ───────────────────────────────────────

    // 根据当前设备状态，建议最佳切分层
    // 返回 -1 = 全部本地，0..31 = 在第 N 层切分
    int  recommend_split_layer(int model_total_layers,
                                int avail_mem_mb,
                                double target_tps) const;

private:
    // 内部：公共 HTTP 请求逻辑
    void run_split_request(
        std::string body_json,
        edge_cloud_token_cb on_token,
        edge_cloud_perf_cb  on_perf);

    void run_sse_request(
        const std::string & path,
        const std::string & body_json,
        edge_cloud_token_cb on_token,
        edge_cloud_perf_cb  on_perf);

    // SSE 解析
    void parse_sse_chunk(const std::string & chunk,
                         edge_cloud_token_cb on_token,
                         CloudPerf & perf);

    // 序列化工具
    std::string serialize_hidden_to_base64(const HiddenStates & h);
    std::string build_split_request_json(
        const HiddenStates    & hidden,
        const InferenceContext & ctx,
        const EdgeDeviceInfo  & device);

    // 状态
    std::string _base_url;
    std::string _api_key;
    std::string _model_id  = "llama-star-7b";
    int         _timeout   = 30;
    DataDtype   _dtype     = DataDtype::Float16;

    std::mutex  _mutex;
    std::thread _thread;
    std::atomic<bool> _running{false};
    std::atomic<bool> _stop_flag{false};
    std::atomic<bool> _reachable{false};

    // Per-request SSE state (cleared on each new request)
    std::string _sse_leftover;
    bool        _sse_ttft_recorded = false;
};
