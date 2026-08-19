// EdgeCloudClient implementation — cloud-edge collaborative inference
//
// Implements the class declared in edge_cloud.h:
//   Split Inference, Query Offload, Feature Offload, Health Check.
//
// Reuses common/http.h (common_http_client) and nlohmann/json for
// serialization.  SSE parsing follows the same pattern as cloud.cpp.

#include "edge_cloud.h"

#include "http.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════

// Simple base64 encoder (no external dependency).
static std::string base64_encode(const uint8_t * data, size_t len) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= (uint32_t)data[i + 2];
        out.push_back(table[(v >> 18) & 0x3f]);
        out.push_back(table[(v >> 12) & 0x3f]);
        if (i + 1 < len) out.push_back(table[(v >> 6) & 0x3f]);
        else             out.push_back('=');
        if (i + 2 < len) out.push_back(table[v & 0x3f]);
        else             out.push_back('=');
    }
    return out;
}

// Simple MD5-like checksum (Fowler-Noll-Vo hash for speed).
// Not cryptographically secure; used only for data integrity verification.
static std::string fnv_checksum(const uint8_t * data, size_t len) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ULL;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)hash);
    return std::string(buf);
}

// Convert DataDtype enum to JSON string.
static std::string dtype_to_str(DataDtype dt) {
    return dtype_str(dt);
}

// Convert edge_cloud_token_cb to the simpler cloud_token_callback style
// used internally for SSE stream parsing.
using stream_token_cb = std::function<void(const std::string & text, bool finished, const std::string & error)>;

// ═══════════════════════════════════════════════════════════════
// EdgeCloudClient — lifecycle
// ═══════════════════════════════════════════════════════════════

EdgeCloudClient::EdgeCloudClient()  = default;
EdgeCloudClient::~EdgeCloudClient() { stop(); }

void EdgeCloudClient::set_endpoint(const std::string & base_url) {
    _base_url = base_url;
}

void EdgeCloudClient::set_api_key(const std::string & key) {
    _api_key = key;
}

void EdgeCloudClient::set_model(const std::string & model_id) {
    _model_id = model_id;
}

void EdgeCloudClient::set_timeout(int seconds) {
    _timeout = seconds;
}

void EdgeCloudClient::set_dtype(DataDtype dtype) {
    _dtype = dtype;
}

void EdgeCloudClient::stop() {
    _stop_flag.store(true);
    if (_thread.joinable()) {
        _thread.join();
    }
}

bool EdgeCloudClient::is_running() const {
    return _running.load();
}

// ═══════════════════════════════════════════════════════════════
// Health check
// ═══════════════════════════════════════════════════════════════

EdgeCloudClient::HealthStatus EdgeCloudClient::check_health() {
    HealthStatus status;
    try {
        std::string url = _base_url + "/api/v1/health";
        auto [cli, parts] = common_http_client(url);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(2, 0);

        auto t0 = std::chrono::steady_clock::now();
        auto res = cli.Get(parts.path.c_str());
        auto t1 = std::chrono::steady_clock::now();

        if (res && res->status == 200) {
            status.reachable  = true;
            status.latency_ms = (int)std::chrono::duration_cast<
                std::chrono::milliseconds>(t1 - t0).count();

            try {
                auto j = json::parse(res->body);
                status.cloud_load   = j.value("load", -1);
                status.cloud_version = j.value("version", "unknown");
                status.max_supported_split_layer =
                    j.value("max_split_layer", 32);
            } catch (...) {
                // Minimal response is fine — server may return just 200.
            }
        } else {
            status.reachable = false;
            status.latency_ms = -1;
        }
        _reachable.store(status.reachable);
    } catch (...) {
        status.reachable  = false;
        status.latency_ms = -1;
        _reachable.store(false);
    }
    return status;
}

// ═══════════════════════════════════════════════════════════════
// Core inference methods
// ═══════════════════════════════════════════════════════════════

void EdgeCloudClient::split_infer(
    const HiddenStates    & hidden,
    const InferenceContext & ctx,
    const EdgeDeviceInfo  & device,
    edge_cloud_token_cb     on_token,
    edge_cloud_perf_cb      on_perf)
{
    std::string body = build_split_request_json(hidden, ctx, device);
    run_split_request(std::move(body), on_token, on_perf);
}

void EdgeCloudClient::query_offload(
    const std::string     & messages_json,
    const InferenceContext & ctx,
    edge_cloud_token_cb     on_token,
    edge_cloud_perf_cb      on_perf)
{
    // Build a request body compatible with the OpenAI chat completions format
    // but extended with inference context for sampling consistency.
    json body;
    try {
        body["messages"] = json::parse(messages_json);
    } catch (...) {
        body["messages"] = json::array();
    }
    body["model"]          = _model_id;
    body["temperature"]    = ctx.temperature;
    body["top_p"]          = ctx.top_p;
    body["min_p"]          = ctx.min_p;
    body["top_k"]          = ctx.top_k;
    body["repeat_penalty"] = ctx.repeat_penalty;
    body["max_tokens"]     = ctx.max_new_tokens;
    body["seed"]           = ctx.seed;
    body["stream"]         = true;

    if (!ctx.stop_strings.empty()) {
        body["stop"] = ctx.stop_strings;
    }

    run_sse_request("/v1/chat/completions", body.dump(), on_token, on_perf);
}

void EdgeCloudClient::feature_offload(
    const std::string      & text_prompt,
    const MultimodalFeatures & media,
    const InferenceContext  & ctx,
    edge_cloud_token_cb      on_token,
    edge_cloud_perf_cb       on_perf)
{
    json body;
    body["prompt"]      = text_prompt;
    body["model"]       = _model_id;
    body["temperature"] = ctx.temperature;
    body["max_tokens"]  = ctx.max_new_tokens;
    body["stream"]      = true;

    // Serialize media
    json jmedia;
    jmedia["type"]         = media.media_type;
    jmedia["width"]        = media.width;
    jmedia["height"]       = media.height;
    jmedia["channels"]     = media.channels;
    jmedia["mime"]         = media.original_mime;
    jmedia["timestamp_us"] = media.timestamp_us;

    if (!media.visual_features.empty()) {
        // Quantize to fp16 for bandwidth efficiency
        jmedia["feature_shape"] = media.feature_shape;
        jmedia["features_b64"] = base64_encode(
            reinterpret_cast<const uint8_t *>(media.visual_features.data()),
            media.visual_features.size() * sizeof(float));
    } else if (!media.raw_media_bytes.empty()) {
        jmedia["raw_bytes_b64"] = base64_encode(
            media.raw_media_bytes.data(),
            media.raw_media_bytes.size());
    }

    body["media"] = jmedia;

    run_sse_request("/api/v1/infer/features", body.dump(), on_token, on_perf);
}

// ═══════════════════════════════════════════════════════════════
// Recommend split layer (heuristic)
// ═══════════════════════════════════════════════════════════════

int EdgeCloudClient::recommend_split_layer(
    int model_total_layers,
    int avail_mem_mb,
    double target_tps) const
{
    // Basic heuristic:
    //   - More memory → run more layers locally
    //   - High network RTT → run more layers locally (reduce data transfer)
    //   - Low target TPS → can afford more cloud layers

    if (model_total_layers <= 0) return -1;  // invalid input
    if (avail_mem_mb <= 0)       return -1;  // unknown memory → all local

    // Rough estimate: each layer costs ~50 MB for a 1.5B model.
    const double mb_per_layer = 50.0;
    int max_local_layers = (int)(avail_mem_mb / mb_per_layer);

    // Clamp to valid range
    max_local_layers = std::max(1, std::min(max_local_layers, model_total_layers));

    // If we have enough memory for all layers, run entirely locally.
    if (max_local_layers >= model_total_layers) return -1;

    // TODO: If target TPS is already satisfied locally, don't split.
    // (Requires access to actual measured local TPS.)

    // Return the recommended split point (last local layer).
    // -1 = all local, 0..N = split after this layer.
    return max_local_layers;
}

// ═══════════════════════════════════════════════════════════════
// Serialization helpers
// ═══════════════════════════════════════════════════════════════

std::string EdgeCloudClient::serialize_hidden_to_base64(const HiddenStates & h) {
    return base64_encode(h.data.data(), h.data.size());
}

std::string EdgeCloudClient::build_split_request_json(
    const HiddenStates    & hidden,
    const InferenceContext & ctx,
    const EdgeDeviceInfo  & device)
{
    json body;

    // Hidden states
    body["hidden"] = {
        {"split_layer",   hidden.split_layer},
        {"current_heads", hidden.current_heads},
        {"head_dim",      hidden.head_dim},
        {"batch_size",    hidden.batch_size},
        {"seq_len",       hidden.seq_len},
        {"dtype",         dtype_to_str(hidden.dtype)},
        {"data_b64",      serialize_hidden_to_base64(hidden)},
        {"data_bytes",    hidden.data.size()},
        {"checksum",      hidden.checksum_md5.empty()
                              ? fnv_checksum(hidden.data.data(), hidden.data.size())
                              : hidden.checksum_md5},
    };

    // Inference context
    json jctx;
    jctx["n_past"]          = ctx.n_past;
    jctx["temperature"]     = ctx.temperature;
    jctx["top_p"]           = ctx.top_p;
    jctx["min_p"]           = ctx.min_p;
    jctx["top_k"]           = ctx.top_k;
    jctx["repeat_penalty"]  = ctx.repeat_penalty;
    jctx["max_new_tokens"]  = ctx.max_new_tokens;
    jctx["seed"]            = ctx.seed;
    if (!ctx.stop_strings.empty()) {
        jctx["stop"] = ctx.stop_strings;
    }
    // Token IDs — send as base64 to keep JSON compact
    if (!ctx.prompt_tokens.empty()) {
        jctx["prompt_tokens_b64"] = base64_encode(
            reinterpret_cast<const uint8_t *>(ctx.prompt_tokens.data()),
            ctx.prompt_tokens.size() * sizeof(int32_t));
        jctx["n_prompt_tokens"] = ctx.prompt_tokens.size();
    }
    if (ctx.has_kv_cache && !ctx.kv_cache_data.empty()) {
        jctx["kv_cache_b64"] = base64_encode(
            ctx.kv_cache_data.data(), ctx.kv_cache_data.size());
    }
    body["context"] = jctx;

    // Device info (for cloud-side scheduling)
    body["device"] = {
        {"device_id",      device.device_id},
        {"device_type",    device.device_type},
        {"battery_pct",    device.battery_pct},
        {"on_ac",          device.on_ac},
        {"avail_mem_mb",   device.avail_mem_mb},
        {"current_tps",    device.current_tps},
        {"network_rtt_ms", device.network_rtt_ms},
    };

    body["model"] = _model_id;

    return body.dump();
}

// ═══════════════════════════════════════════════════════════════
// HTTP request execution (private)
// ═══════════════════════════════════════════════════════════════

void EdgeCloudClient::run_split_request(
    std::string body_json,
    edge_cloud_token_cb on_token,
    edge_cloud_perf_cb  on_perf)
{
    run_sse_request("/api/v1/infer/split", std::move(body_json), on_token, on_perf);
}

void EdgeCloudClient::run_sse_request(
    const std::string & path,
    const std::string & body_json,
    edge_cloud_token_cb on_token,
    edge_cloud_perf_cb  on_perf)
{
    if (_running.load()) return;

    _stop_flag.store(false);
    _running.store(true);

    if (_thread.joinable()) _thread.join();
    _thread = std::thread([this, path, body_json, on_token, on_perf]() {
        // Clear per-request SSE state
        _sse_leftover.clear();
        _sse_ttft_recorded = false;

        auto t_start = std::chrono::steady_clock::now();

        try {
            std::string full_url = _base_url + path;
            auto [cli, parts] = common_http_client(full_url);
            cli.set_connection_timeout(_timeout, 0);
            cli.set_read_timeout(0, 0);  // no limit for streaming

            if (!_api_key.empty()) {
                cli.set_bearer_token_auth(_api_key.c_str());
            }

            httplib::Headers headers = {
                {"Content-Type", "application/json"},
                {"Accept",       "text/event-stream"},
            };

            CloudPerf perf;
            perf.valid = true;

            // Timing: upload phase
            auto t_upload_done = t_start;

            // Per-request state (not static — fresh for each request)
            bool first_chunk = true;

            httplib::ContentReceiver receiver =
                [&, first_chunk](const char * data, size_t len) mutable -> bool {
                if (_stop_flag.load()) return false;

                // Mark upload complete on first data
                if (first_chunk) {
                    t_upload_done = std::chrono::steady_clock::now();
                    first_chunk = false;
                }

                parse_sse_chunk(std::string(data, len), on_token, perf);
                return true;
            };

            auto res = cli.Post(parts.path, headers, body_json,
                                "application/json", receiver);

            auto t_end = std::chrono::steady_clock::now();

            if (!res) {
                on_token("", true, "[Cloud unreachable]");
                _reachable.store(false);
                if (on_perf) { perf.valid = false; on_perf(perf); }
            } else {
                // Finalize performance counters
                perf.upload_ms = std::chrono::duration<double, std::milli>(
                    t_upload_done - t_start).count();
                perf.download_ms = std::chrono::duration<double, std::milli>(
                    t_end - t_upload_done).count();
                if (on_perf) on_perf(perf);
            }
        } catch (...) {
            on_token("", true, "[Cloud error]");
            _reachable.store(false);
            if (on_perf) {
                CloudPerf perf;
                perf.valid = false;
                on_perf(perf);
            }
        }

        _running.store(false);
    });
}

// ═══════════════════════════════════════════════════════════════
// SSE chunk parsing
// ═══════════════════════════════════════════════════════════════

void EdgeCloudClient::parse_sse_chunk(
    const std::string & chunk,
    edge_cloud_token_cb on_token,
    CloudPerf & perf)
{
    // Parse SSE: "data: {...}\n\n"
    // Use member _sse_leftover (not static) so each request has a clean slate.
    std::string buf = _sse_leftover + chunk;
    _sse_leftover.clear();

    size_t pos = 0;
    while (pos < buf.size() && !_stop_flag.load()) {
        // Find "data: "
        auto data_start = buf.find("data: ", pos);
        if (data_start == std::string::npos) {
            _sse_leftover = buf.substr(pos);
            break;
        }
        data_start += 6;  // skip "data: "

        auto data_end = buf.find('\n', data_start);
        if (data_end == std::string::npos) {
            _sse_leftover = buf.substr(pos);
            break;
        }

        std::string line = buf.substr(data_start, data_end - data_start);
        pos = data_end + 1;

        // Skip empty lines between SSE events
        if (line.empty()) continue;

        // Check for [DONE]
        if (line == "[DONE]") {
            on_token("", true, "");
            return;
        }

        try {
            auto j = json::parse(line);

            // Handle errors from cloud
            if (j.contains("error")) {
                std::string err = j["error"].value("message",
                    j["error"].is_string() ? j["error"].get<std::string>() : "Unknown cloud error");
                on_token("", true, err);
                return;
            }

            // Standard OpenAI-format streaming chunk
            auto choices = j.value("choices", json::array());
            if (!choices.empty()) {
                auto & choice = choices[0];
                auto delta = choice.value("delta", json::object());
                auto content = delta.value("content", "");

                // Check finish reason
                auto finish = choice.value("finish_reason", "");
                if (!finish.empty() && finish != "null") {
                    if (!content.empty()) {
                        on_token(content, false, "");
                    }
                    on_token("", true, "");
                    return;
                }

                if (!content.empty()) {
                    on_token(content, false, "");
                    ++perf.total_tokens;
                }
            }

            // Extract cloud performance data if embedded
            if (j.contains("usage")) {
                auto & usage = j["usage"];
                perf.total_tokens = usage.value("total_tokens", perf.total_tokens);
            }

        } catch (const json::parse_error &) {
            // Skip malformed SSE data lines
        }
    }

    // Time-to-first-token tracking (member, reset per request)
    if (!_sse_ttft_recorded && perf.total_tokens > 0) {
        _sse_ttft_recorded = true;
        // ttft_ms is set in run_sse_request from timing
    }
}
