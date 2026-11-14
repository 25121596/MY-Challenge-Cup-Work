#include "device_monitor.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "http.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <psapi.h>
#else
#  include <sys/sysinfo.h>
#  include <unistd.h>
#endif

// ═══════════════════════════════════════════════════════════════
// DeviceMonitor — lifecycle
// ═══════════════════════════════════════════════════════════════

DeviceMonitor::DeviceMonitor() {}

// ═══════════════════════════════════════════════════════════════
// Full snapshot
// ═══════════════════════════════════════════════════════════════

DeviceSnapshot DeviceMonitor::snapshot() {
    DeviceSnapshot s;
    s.cpu     = query_cpu();
    s.gpus    = query_gpu_memory();
    query_system_ram(s.avail_ram_mb, s.total_ram_mb);
    s.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return s;
}

// ═══════════════════════════════════════════════════════════════
// CPU usage
// ═══════════════════════════════════════════════════════════════

CpuMetrics DeviceMonitor::query_cpu() {
    CpuMetrics m;

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    m.core_count = (int)si.dwNumberOfProcessors;

    FILETIME idle, kernel, user;
    if (GetSystemTimes(&idle, &kernel, &user)) {
        auto to_u64 = [](const FILETIME & ft) -> unsigned long long {
            return ((unsigned long long)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
        };
        auto idle_v  = to_u64(idle);
        auto kern_v  = to_u64(kernel);
        auto user_v  = to_u64(user);

        if (_prev_cpu_valid) {
            auto d_idle  = idle_v - _prev_cpu.idle;
            auto d_kern  = kern_v - _prev_cpu.kernel;
            auto d_user  = user_v - _prev_cpu.user;
            auto d_total = d_kern + d_user;
            if (d_total > 0) {
                m.usage_pct = (double)(d_total - d_idle) / d_total * 100.0;
                if (m.usage_pct < 0.0) m.usage_pct = 0.0;
                if (m.usage_pct > 100.0) m.usage_pct = 100.0;
            }
        }
        _prev_cpu = {idle_v, kern_v, user_v};
        _prev_cpu_valid = true;
    }

#elif defined(__linux__)
    // Read /proc/stat
    FILE * fp = fopen("/proc/stat", "r");
    if (fp) {
        char line[256];
        if (fgets(line, sizeof(line), fp)) {
            unsigned long long user, nice, system, idle;
            if (sscanf(line, "cpu %llu %llu %llu %llu",
                       &user, &nice, &system, &idle) == 4) {
                if (_prev_cpu_valid) {
                    auto d_user   = user   - _prev_cpu.user;
                    auto d_nice   = nice   - _prev_cpu.nice;
                    auto d_system = system - _prev_cpu.system;
                    auto d_idle   = idle   - _prev_cpu.idle;
                    auto d_total  = d_user + d_nice + d_system + d_idle;
                    if (d_total > 0) {
                        m.usage_pct = (double)(d_total - d_idle) / d_total * 100.0;
                        if (m.usage_pct < 0.0) m.usage_pct = 0.0;
                        if (m.usage_pct > 100.0) m.usage_pct = 100.0;
                    }
                }
                _prev_cpu = {user, nice, system, idle};
                _prev_cpu_valid = true;
            }
        }
        fclose(fp);
    }
    m.core_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif

    return m;
}

// ═══════════════════════════════════════════════════════════════
// System RAM
// ═══════════════════════════════════════════════════════════════

void DeviceMonitor::query_system_ram(size_t & avail_mb, size_t & total_mb) {
#ifdef _WIN32
    MEMORYSTATUSEX msx;
    msx.dwLength = sizeof(msx);
    if (GlobalMemoryStatusEx(&msx)) {
        total_mb = msx.ullTotalPhys / (1024 * 1024);
        avail_mb = msx.ullAvailPhys / (1024 * 1024);
    }
#else
    // Match ggml-cpu.cpp pattern
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    total_mb = ((size_t)pages * (size_t)page_size) / (1024 * 1024);
    // Linux: read MemAvailable from /proc/meminfo for accurate available RAM
    FILE * fp = fopen("/proc/meminfo", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            unsigned long kb;
            if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) {
                avail_mb = kb / 1024;
                break;
            }
        }
        fclose(fp);
    }
    if (avail_mb == 0) avail_mb = total_mb;  // fallback
#endif
}

// ═══════════════════════════════════════════════════════════════
// GPU memory via ggml_backend
// ═══════════════════════════════════════════════════════════════

std::vector<GpuMetrics> DeviceMonitor::query_gpu_memory() {
    std::vector<GpuMetrics> result;

    // ggml_backend may not have been loaded yet; guard against that.
    // We call ggml_backend_dev_count() which is always safe.
    size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;

        int dev_type = (int)ggml_backend_dev_type(dev);

        // Report GPU, IGPU, and ACCEL devices.  Skip CPU and META.
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU &&
            dev_type != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }

        GpuMetrics g;
        g.name        = ggml_backend_dev_name(dev)
                            ? ggml_backend_dev_name(dev) : "unknown";
        g.description = ggml_backend_dev_description(dev)
                            ? ggml_backend_dev_description(dev) : "";
        g.device_type = dev_type;

        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        g.total_mem_mb = total / (1024 * 1024);
        g.free_mem_mb  = free  / (1024 * 1024);
        if (g.total_mem_mb >= g.free_mem_mb) {
            g.used_mem_mb = g.total_mem_mb - g.free_mem_mb;
        }

        result.push_back(std::move(g));
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════
// Network RTT measurement
// ═══════════════════════════════════════════════════════════════

int DeviceMonitor::measure_rtt_ms(const std::string & endpoint,
                                  int timeout_ms) {
    try {
        auto [cli, parts] = common_http_client(endpoint);
        cli.set_connection_timeout(timeout_ms / 1000,
                                    (timeout_ms % 1000) * 1000);
        cli.set_read_timeout(timeout_ms / 1000,
                              (timeout_ms % 1000) * 1000);

        auto t0 = std::chrono::steady_clock::now();
        auto res = cli.Get(parts.path.empty() ? "/" : parts.path.c_str());
        auto t1 = std::chrono::steady_clock::now();

        if (res) {
            return (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                t1 - t0).count();
        }
    } catch (...) {
        // Endpoint unreachable
    }
    return -1;
}
