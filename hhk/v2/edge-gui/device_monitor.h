#pragma once

// ── Device metrics collection ──────────────────────────────────
//
// Provides a unified snapshot of edge-device resource state:
// CPU, RAM, GPU (via ggml_backend), and inference performance.
//
// Dedicated to Module 2.1: state reporting & heartbeat.

#include <cstdint>
#include <string>
#include <vector>

// ── CPU metrics ─────────────────────────────────────────────────

struct CpuMetrics {
    double usage_pct  = 0.0;   // 0..100 (all-core average)
    int    core_count = 0;
    double freq_mhz   = 0.0;   // best-effort, 0 if unavailable
};

// ── GPU metrics (one entry per GPU-type backend device) ────────

struct GpuMetrics {
    std::string name;
    std::string description;
    size_t  total_mem_mb = 0;
    size_t  free_mem_mb  = 0;
    size_t  used_mem_mb  = 0;   // total - free
    int     device_type  = 0;   // ggml_backend_dev_type enum value
};

// ── Network metrics (measured against cloud endpoint) ──────────

struct NetworkMetrics {
    int     rtt_ms          = -1;    // -1 = not measured
    double  packet_loss_pct = 0.0;   // 0..100
    bool    cloud_reachable = false;
};

// ── Full device snapshot ────────────────────────────────────────

struct DeviceSnapshot {
    CpuMetrics                cpu;
    std::vector<GpuMetrics>   gpus;
    NetworkMetrics            network;
    size_t  avail_ram_mb   = 0;
    size_t  total_ram_mb   = 0;
    double  current_tps    = 0.0;   // tokens/sec from last inference
    int     task_queue_len = 0;     // pending inference tasks
    int64_t timestamp_us   = 0;     // Posix-us timestamp
};

// ── Device monitor ──────────────────────────────────────────────

class DeviceMonitor {
public:
    DeviceMonitor();

    // Take a full snapshot of current device state.
    // This is cheap enough to call every 1-5 seconds.
    DeviceSnapshot snapshot();

    // Measure RTT to a given endpoint via a lightweight HTTP HEAD/GET.
    // Returns -1 on failure, latency in ms on success.
    int measure_rtt_ms(const std::string & endpoint, int timeout_ms = 2000);

    // Query GPU devices via ggml_backend_dev_* APIs.
    std::vector<GpuMetrics> query_gpu_memory();

private:
    CpuMetrics query_cpu();
    void       query_system_ram(size_t & avail_mb, size_t & total_mb);

    // For CPU usage delta calculation on Windows.
#ifdef _WIN32
    struct CpuDelta {
        unsigned long long idle, kernel, user;
    };
    CpuDelta _prev_cpu = {};
    bool     _prev_cpu_valid = false;
#endif

    // For CPU usage delta calculation on Linux.
#if defined(__linux__) && !defined(_WIN32)
    struct CpuDelta {
        unsigned long long user, nice, system, idle;
    };
    CpuDelta _prev_cpu = {};
    bool     _prev_cpu_valid = false;
#endif
};
