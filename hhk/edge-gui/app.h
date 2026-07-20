#pragma once

#include "engine.h"
#include "cloud.h"
#include "edge_cloud.h"
#include "chat.h"
#include "device_monitor.h"
#include "heartbeat.h"
#include "task_scheduler.h"
#include "model_sync.h"
#include "p2p_mesh.h"
#include "conflict_detector.h"
#include "edge_local_decision.h"

#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

// ── Power status (edge device monitoring) ─────────────────────

struct PowerInfo {
    bool valid           = false;
    int  battery_percent = 0;    // 0 .. 100
    int  seconds_left    = -1;   // -1 = unknown
    bool on_ac           = true; // plugged in
};

// ── Model entry for the model browser ─────────────────────────

struct ModelEntry {
    std::string path;
    std::string filename;
    size_t      file_size_bytes = 0;
    bool        compatible      = false;  // fits within user memory budget
};

// ── Memory budget recommendation ──────────────────────────────

enum class MemFit { Good, Tight, Over };
const char * mem_fit_str(MemFit f);

// ── Main application ──────────────────────────────────────────

class EdgeApp {
public:
    EdgeApp();
    ~EdgeApp();

    // Load a model, optionally showing progress. Returns error string or "" on success.
    std::string load_model(const std::string & path);

    // ── Cloud-edge subsystems (Module 2 + 3) ─────────────────
    EdgeCloudClient  _cloud_client;
    DeviceMonitor    _device_monitor;
    HeartbeatManager _heartbeat;
    TaskScheduler    _task_scheduler;
    ModelSyncManager _model_sync;
    P2PMesh          _p2p;
    ConflictDetector _conflict_detector;
    EdgeLocalDecision  _local_decision;

    // Render the full UI. Called once per frame.
    void render();

    // Cleanup before exit.
    void shutdown();

    // ── Callbacks for custom integration ──────────────────────
    std::function<void(const std::string & title, const std::string & msg)>
        on_error;  // set before render() to override error display

private:
    // ── UI helpers ────────────────────────────────────────────
    void render_top_bar();
    void render_chat_area();
    void render_input_area();
    void render_settings_panel();
    void render_cloud_panel();
    void render_model_browser_panel();
    void render_error_popup();
    void render_heartbeat_panel();
    void render_peers_panel();
    void render_conflicts_panel();
    void render_tasks_panel();

    // ── Layout constants ──────────────────────────────────────
    static constexpr float TOP_BAR_H     = 36.0f;
    static constexpr float INPUT_AREA_H  = 55.0f;

    // ── Logic ─────────────────────────────────────────────────
    void submit_message(const std::string & text);
    void process_incoming_tokens();

    void save_current_session();
    void load_session_dialog();

    // Scan a directory for GGUF files
    std::vector<ModelEntry> scan_models_dir(const std::string & dir);

    // Auto-detect the llama.cpp project root from the executable path.
    // Walks up from the exe to find the directory containing ggml/, src/, models/.
    // Returns an absolute path, or empty if detection fails.
    std::string detect_project_root();

    // Check if a model fits within the user-specified memory budget
    MemFit check_model_fit(const ModelEntry & entry);

    // Estimate runtime memory: model_size + KV cache overhead
    size_t estimate_runtime_mem(size_t model_file_size) const;

    // ── Power ─────────────────────────────────────────────────
    PowerInfo get_power_info();

    // ── State ─────────────────────────────────────────────────
    EdgeEngine _engine;
    EdgeCloud  _cloud;

    // Messages use the standard common_chat_msg format
    // (supports role, content, tool_calls, reasoning_content, etc.)
    std::deque<common_chat_msg> _messages;
    std::string _system_prompt;       // optional, set in settings

    // Input buffer
    char _input_buf[4096] = {};

    // Streaming state
    std::string _streaming_text;
    std::mutex  _streaming_mutex;
    std::vector<std::string> _pending_tokens;
    bool _generation_finished = true;

    // Performance from last generation
    EnginePerf _last_perf;

    // UI toggles
    bool _show_settings      = false;
    bool _show_cloud_cfg     = false;
    bool _show_model_browser = false;

    // Model info
    bool   _model_loaded   = false;
    bool   _model_loading   = false;    // background load in progress
    float  _model_load_prog = 0.0f;
    std::string _model_load_stage;
    std::string _model_error;           // last error message

    // Model browser state
    std::vector<ModelEntry> _available_models;
    std::string _models_dir;               // set by detect_project_root() on startup
    char _models_dir_buf[512] = {};       // editable copy for Settings panel
    bool _models_scanned    = false;

    // ── Cloud settings (persistent via edge-gui.ini) ─────────
    char _cloud_endpoint[256] = "http://localhost:8080/v1/chat/completions";
    char _cloud_api_key[128]  = {};
    char _cloud_model[128]    = {};
    bool _cloud_enabled       = false;

    // ── Memory budget (user-specified, persistent via ini) ───
    int _avail_mem_mb = 0;     // 0 = not set, use default behavior

    // ── Model loading thread (joined on shutdown to prevent use-after-free)
    std::thread _model_load_thread;

    // ── Power status (updated periodically) ──────────────────
    PowerInfo _power_info;
    double    _power_last_check = 0;  // ImGui::GetTime() at last check

    // ── Cloud-edge subsystem UI state ────────────────────────
    bool _show_heartbeat  = false;
    bool _show_peers      = false;
    bool _show_conflicts  = false;
    bool _show_tasks      = false;

    // ── P2P / heartbeat config (persistable via ini) ─────────
    char _p2p_node_id[64]       = "edge-node-01";
    int  _p2p_udp_port          = 15555;
    int  _p2p_tcp_port          = 15556;
    char _heartbeat_endpoint[256] = "http://localhost:8080/api/v1/edge/heartbeat";

    // ── Cloud-edge callbacks ─────────────────────────────────
    void on_peer_perception(const PerceptionReport & report);
    void on_peer_intent(const DecisionIntent & intent);
    void on_conflict(const ConflictRecord & cr);
    void on_conflict_done(const ConflictRecord & cr);
    void update_device_metrics();
    double _last_metrics_update = 0;
};
