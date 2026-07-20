#pragma once

// ── Adaptive heartbeat / state reporter ─────────────────────────
//
// Periodically POSTs device snapshots to the cloud so the scheduler
// can make informed task-routing decisions (Module 2.1).
//
// Frequency adapts automatically:
//   stable      → stable_interval_sec (default 30 s)
//   fluctuating → fluctuate_interval_sec (default 5 s)

#include "device_monitor.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

// ── Heartbeat configuration ─────────────────────────────────────

struct HeartbeatConfig {
    std::string cloud_endpoint;           // "http://host:port/api/v1/edge/heartbeat"
    std::string device_id;
    std::string device_type;              // "jetson-orin", "raspberry-pi-5", etc.
    int   stable_interval_sec      = 30;
    int   fluctuate_interval_sec   = 5;
    double fluctuation_threshold   = 20.0;   // % change that triggers "fluctuating"
};

// ── Heartbeat status callback ───────────────────────────────────

using heartbeat_status_cb = std::function<void(bool success, int latency_ms,
                                                 const std::string & msg)>;

// ── Heartbeat reporter ──────────────────────────────────────────

class HeartbeatManager {
public:
    HeartbeatManager();
    ~HeartbeatManager();

    // ── Configuration ──────────────────────────────────────────
    void configure(const HeartbeatConfig & cfg);
    void set_endpoint(const std::string & url);
    void set_device_id(const std::string & id);
    void set_device_type(const std::string & type);

    // ── Lifecycle ──────────────────────────────────────────────
    void start();
    void stop();
    bool is_running() const;

    // ── External data feeds ────────────────────────────────────
    // Called each frame (or periodically) to update inference stats
    void update_perf(double current_tps, int task_queue_len);

    // Pre-set network RTT (avoids a separate HTTP call in collect)
    void set_network_rtt(int rtt_ms, bool cloud_reachable);

    // ── Callbacks ──────────────────────────────────────────────
    heartbeat_status_cb on_status;

    // ── Manual trigger ─────────────────────────────────────────
    // Force an immediate heartbeat regardless of interval
    bool send_now();

    // ── Read last state ────────────────────────────────────────
    bool     last_success()  const { return _last_success; }
    int      last_latency()  const { return _last_latency_ms; }
    int64_t  last_report_at() const { return _last_report_us; }

    // Number of consecutive failed heartbeat attempts since last success.
    int consecutive_failures() const { return _consecutive_failures; }

private:
    void run_loop();
    bool send_heartbeat(const std::string & json_body, int & out_latency_ms);
    std::string build_report_json(const DeviceSnapshot & snap);
    bool is_fluctuating(const DeviceSnapshot & cur, const DeviceSnapshot & prev);

    // ── State ──────────────────────────────────────────────────
    HeartbeatConfig _cfg;
    DeviceMonitor   _monitor;

    double _current_tps    = 0.0;
    int    _task_queue_len = 0;
    int    _network_rtt_ms = -1;
    bool   _cloud_reachable = false;

    DeviceSnapshot _prev_snapshot;
    bool           _prev_valid = false;

    std::thread _thread;
    std::atomic<bool> _running{false};
    std::mutex _mutex;

    bool    _last_success    = false;
    int     _last_latency_ms = -1;
    int64_t _last_report_us  = 0;

    // Consecutive failed heartbeat count (reset on success)
    int     _consecutive_failures = 0;
};
