#pragma once

// ── Cloud task dispatch & result fusion (Module 2.2) ─────────────
//
// TaskScheduler bridges the gap between "local inference" and
// "cloud offload".  It:
//   1. Polls the cloud for pending collaborative tasks
//   2. Routes tasks to EdgeCloudClient::split_infer / query_offload / feature_offload
//   3. Fuses cloud results with local decisions
//
// Owns no heavyweight resources — receives pointers to EdgeEngine
// and EdgeCloudClient from EdgeApp.

#include "edge_cloud.h"

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward-declare EdgeEngine (avoids circular include).
class EdgeEngine;

// ═══════════════════════════════════════════════════════════════
// Task types
// ═══════════════════════════════════════════════════════════════

enum class TaskType {
    NoOp,              // nothing to do
    UploadFeatures,    // extract & upload intermediate features
    CloudReview,       // send context to cloud for full-model review
    ModelUpdate,       // download & apply LoRA weights
    RuleSync,          // update decision thresholds / priorities
};

// ── Cloud-dispatched task ──────────────────────────────────────

struct CloudTask {
    TaskType    type        = TaskType::NoOp;
    std::string task_id;
    std::string description;
    int         priority    = 0;     // higher = more urgent
    std::string payload_json;        // task-specific parameters
};

// ── Feature vector (lightweight, for upload) ───────────────────

struct FeatureVector {
    std::string layer_name;                // e.g. "output_embeddings"
    std::vector<int64_t> shape;
    std::vector<float> values;             // can be quantized to int8
    std::string dtype = "float32";
};

// ── Task execution result ──────────────────────────────────────

struct TaskResult {
    std::string task_id;
    bool        success     = false;
    std::string result_json;              // cloud response
    std::string error_msg;
};

// ── Fused result (local + cloud) ───────────────────────────────

struct FusedResult {
    std::string final_text;               // text to present to user
    std::string source;                   // "local", "cloud", "fused"
    double      confidence  = 0.0;
    std::vector<std::string> components;  // individual fused results
};

// ═══════════════════════════════════════════════════════════════
// Callbacks
// ═══════════════════════════════════════════════════════════════

using task_received_cb  = std::function<void(const CloudTask &)>;
using task_complete_cb   = std::function<void(const TaskResult &)>;
using fused_result_cb    = std::function<void(const FusedResult &)>;

// ═══════════════════════════════════════════════════════════════
// Task scheduler
// ═══════════════════════════════════════════════════════════════

class TaskScheduler {
public:
    TaskScheduler();
    ~TaskScheduler();

    // ── Pointers (set before start) ────────────────────────────
    void set_cloud(EdgeCloudClient * cloud)  { _cloud = cloud; }
    void set_engine(EdgeEngine * engine)     { _engine = engine; }
    void set_cloud_endpoint(const std::string & url) { _cloud_endpoint = url; }
    void set_device_id(const std::string & id)       { _device_id = id; }

    // ── Lifecycle ──────────────────────────────────────────────
    void start();
    void stop();
    bool is_running() const { return _running.load(); }

    // ── Configuration ──────────────────────────────────────────
    int poll_interval_ms = 2000;   // how often to check for new tasks

    // ── Task I/O ───────────────────────────────────────────────
    // Enqueue a task (from cloud poll or local trigger)
    void enqueue(const CloudTask & task);

    // Process the queue (call each frame; non-blocking)
    void process_queue();

    // Poll cloud for pending tasks (HTTP GET)
    int poll_cloud_tasks();

    // ── Manual triggers ────────────────────────────────────────
    // Request cloud to review current inference context
    bool request_cloud_review(const std::string & context_json,
                              const InferenceContext & ictx);

    // Extract hidden states from the engine at a given layer index
    FeatureVector extract_features(int layer_idx) const;

    // ── Result fusion ──────────────────────────────────────────
    FusedResult fuse(const std::string & local_result,
                     const std::string & cloud_result,
                     double local_conf,
                     double cloud_conf) const;

    // ── Pending count ──────────────────────────────────────────
    int pending_count() const;

    // ── Callbacks ──────────────────────────────────────────────
    task_received_cb  on_task_received;
    task_complete_cb  on_task_complete;
    fused_result_cb   on_fused_result;

private:
    void execute_task(const CloudTask & task);
    TaskResult run_feature_upload(const CloudTask & task);
    TaskResult run_cloud_review(const CloudTask & task);

    EdgeCloudClient * _cloud   = nullptr;
    EdgeEngine      * _engine  = nullptr;
    std::string       _cloud_endpoint;
    std::string       _device_id = "edge-unknown";

    std::deque<CloudTask> _queue;
    mutable std::mutex    _mutex;

    std::thread _thread;
    std::atomic<bool> _running{false};
};
