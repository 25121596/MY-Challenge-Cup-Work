#include "heartbeat.h"

#include "http.h"
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════

HeartbeatManager::HeartbeatManager()  = default;
HeartbeatManager::~HeartbeatManager() { stop(); }

void HeartbeatManager::configure(const HeartbeatConfig & cfg) { _cfg = cfg; }

void HeartbeatManager::set_endpoint(const std::string & url)  { _cfg.cloud_endpoint = url; }
void HeartbeatManager::set_device_id(const std::string & id)  { _cfg.device_id = id; }
void HeartbeatManager::set_device_type(const std::string & t) { _cfg.device_type = t; }

void HeartbeatManager::start() {
    if (_running.load()) return;
    _running.store(true);
    if (_thread.joinable()) _thread.join();
    _thread = std::thread(&HeartbeatManager::run_loop, this);
}

void HeartbeatManager::stop() {
    _running.store(false);
    if (_thread.joinable()) _thread.join();
}

bool HeartbeatManager::is_running() const { return _running.load(); }

void HeartbeatManager::update_perf(double tps, int queue_len) {
    _current_tps    = tps;
    _task_queue_len = queue_len;
}

void HeartbeatManager::set_network_rtt(int rtt_ms, bool reachable) {
    _network_rtt_ms  = rtt_ms;
    _cloud_reachable = reachable;
}

// ═══════════════════════════════════════════════════════════════
// Main loop
// ═══════════════════════════════════════════════════════════════

void HeartbeatManager::run_loop() {
    while (_running.load()) {
        // Wait for the current interval
        int interval_sec = _cfg.stable_interval_sec;
        // Check if we need to use shorter interval
        {
            DeviceSnapshot snap = _monitor.snapshot();
            snap.current_tps    = _current_tps;
            snap.task_queue_len = _task_queue_len;
            snap.network.rtt_ms  = _network_rtt_ms;
            snap.network.cloud_reachable = _cloud_reachable;

            if (_prev_valid && is_fluctuating(snap, _prev_snapshot)) {
                interval_sec = _cfg.fluctuate_interval_sec;
            }
            _prev_snapshot = snap;
            _prev_valid     = true;

            // Build and send
            std::string body = build_report_json(snap);
            int latency_ms = -1;
            bool ok = send_heartbeat(body, latency_ms);

            _last_success    = ok;
            _last_latency_ms = latency_ms;
            _last_report_us  = snap.timestamp_us;

            if (ok) _consecutive_failures = 0;
            else    _consecutive_failures++;

            if (on_status) {
                on_status(ok, latency_ms, ok ? "" : "heartbeat failed");
            }
        }

        // Sleep for the interval (check _running periodically to allow fast stop)
        int slept = 0;
        while (slept < interval_sec && _running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            ++slept;
        }
    }
}

bool HeartbeatManager::send_now() {
    DeviceSnapshot snap = _monitor.snapshot();
    snap.current_tps    = _current_tps;
    snap.task_queue_len = _task_queue_len;
    snap.network.rtt_ms  = _network_rtt_ms;
    snap.network.cloud_reachable = _cloud_reachable;

    std::string body = build_report_json(snap);
    int latency_ms = -1;
    bool ok = send_heartbeat(body, latency_ms);
    _last_success    = ok;
    _last_latency_ms = latency_ms;
    _last_report_us  = snap.timestamp_us;

    if (ok) _consecutive_failures = 0;
    else    _consecutive_failures++;

    _prev_snapshot   = snap;
    _prev_valid      = true;
    return ok;
}

// ═══════════════════════════════════════════════════════════════
// Fluctuation detection
// ═══════════════════════════════════════════════════════════════

bool HeartbeatManager::is_fluctuating(const DeviceSnapshot & cur,
                                       const DeviceSnapshot & prev) {
    double thresh = _cfg.fluctuation_threshold;

    // CPU usage swing
    if (std::abs(cur.cpu.usage_pct - prev.cpu.usage_pct) > thresh)
        return true;

    // Memory usage swing (> 10% of total RAM)
    if (cur.total_ram_mb > 0 && prev.total_ram_mb > 0) {
        double cur_used = cur.total_ram_mb - cur.avail_ram_mb;
        double prev_used = prev.total_ram_mb - prev.avail_ram_mb;
        if (std::abs(cur_used - prev_used) > cur.total_ram_mb * 0.10)
            return true;
    }

    // High network latency
    if (cur.network.rtt_ms > 500) return true;
    // RTT spike
    if (prev.network.rtt_ms > 0 &&
        cur.network.rtt_ms > prev.network.rtt_ms * 3)
        return true;

    // Cloud reachability changed
    if (cur.network.cloud_reachable != prev.network.cloud_reachable)
        return true;

    // Task queue buildup
    if (cur.task_queue_len > 5) return true;

    return false;
}

// ═══════════════════════════════════════════════════════════════
// JSON serialization
// ═══════════════════════════════════════════════════════════════

std::string HeartbeatManager::build_report_json(const DeviceSnapshot & snap) {
    json j;

    j["device_id"]      = _cfg.device_id;
    j["device_type"]    = _cfg.device_type;
    j["timestamp_ms"]   = snap.timestamp_us / 1000;

    // CPU
    j["cpu"] = {
        {"usage_pct",  snap.cpu.usage_pct},
        {"core_count", snap.cpu.core_count},
        {"freq_mhz",   snap.cpu.freq_mhz},
    };

    // Memory
    j["memory"] = {
        {"total_mb",  snap.total_ram_mb},
        {"avail_mb", snap.avail_ram_mb},
    };

    // GPUs
    json gpus = json::array();
    for (const auto & g : snap.gpus) {
        gpus.push_back({
            {"name",       g.name},
            {"total_mb",   g.total_mem_mb},
            {"free_mb",    g.free_mem_mb},
            {"used_mb",    g.used_mem_mb},
            {"device_type", g.device_type},
        });
    }
    j["gpus"] = gpus;

    // Network
    j["network"] = {
        {"rtt_ms",          snap.network.rtt_ms},
        {"packet_loss_pct", snap.network.packet_loss_pct},
        {"cloud_reachable", snap.network.cloud_reachable},
    };

    // Inference
    j["inference"] = {
        {"current_tps",    snap.current_tps},
        {"task_queue_len", snap.task_queue_len},
    };

    // Fluctuation status
    j["is_fluctuating"] = is_fluctuating(snap, _prev_snapshot);

    return j.dump();
}

// ═══════════════════════════════════════════════════════════════
// HTTP POST
// ═══════════════════════════════════════════════════════════════

bool HeartbeatManager::send_heartbeat(const std::string & json_body,
                                       int & out_latency_ms) {
    if (_cfg.cloud_endpoint.empty()) {
        out_latency_ms = -1;
        return false;
    }

    try {
        auto [cli, parts] = common_http_client(_cfg.cloud_endpoint);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(3, 0);

        auto t0 = std::chrono::steady_clock::now();
        httplib::Headers headers = {
            {"Content-Type", "application/json"},
        };
        auto res = cli.Post(parts.path, headers, json_body,
                            "application/json");
        auto t1 = std::chrono::steady_clock::now();

        out_latency_ms = (int)std::chrono::duration_cast<
            std::chrono::milliseconds>(t1 - t0).count();

        return (res && res->status >= 200 && res->status < 300);
    } catch (...) {
        out_latency_ms = -1;
        return false;
    }
}
