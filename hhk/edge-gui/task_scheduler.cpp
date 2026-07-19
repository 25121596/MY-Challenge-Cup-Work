#include "task_scheduler.h"

// We only need the engine's forward declaration; the full header
// is included by app.h which owns both EdgeEngine and TaskScheduler.
#include "engine.h"
#include "http.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

// NOTE: "json" alias is already defined in chat.h via common/include path.
// Use nlohmann::json directly or the existing alias.

// ═══════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════

TaskScheduler::TaskScheduler()  = default;
TaskScheduler::~TaskScheduler() { stop(); }

void TaskScheduler::start() {
    if (_running.load()) return;
    _running.store(true);
    if (_thread.joinable()) _thread.join();
    _thread = std::thread([this]() {
        while (_running.load()) {
            poll_cloud_tasks();
            // Sleep in small increments for responsive stop
            int slept = 0;
            int interval = poll_interval_ms;
            while (slept < interval && _running.load()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(std::min(200, interval - slept)));
                slept += 200;
            }
        }
    });
}

void TaskScheduler::stop() {
    _running.store(false);
    if (_thread.joinable()) _thread.join();
}

// ═══════════════════════════════════════════════════════════════
// Task queue
// ═══════════════════════════════════════════════════════════════

void TaskScheduler::enqueue(const CloudTask & task) {
    std::lock_guard<std::mutex> lock(_mutex);
    _queue.push_back(task);
    // Sort by priority (higher first)
    std::stable_sort(_queue.begin(), _queue.end(),
        [](const CloudTask & a, const CloudTask & b) {
            return a.priority > b.priority;
        });
}

int TaskScheduler::pending_count() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return (int)_queue.size();
}

void TaskScheduler::process_queue() {
    CloudTask task;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_queue.empty()) return;
        task = _queue.front();
        _queue.pop_front();
    }

    if (on_task_received) on_task_received(task);
    execute_task(task);
}

// ═══════════════════════════════════════════════════════════════
// Cloud polling
// ═══════════════════════════════════════════════════════════════

int TaskScheduler::poll_cloud_tasks() {
    if (_cloud_endpoint.empty()) return 0;

    std::string url = _cloud_endpoint +
        "/api/v1/edge/tasks?device_id=" + _device_id;

    try {
        auto [cli, parts] = common_http_client(url);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(2, 0);

        auto res = cli.Get(parts.path.c_str());
        if (!res || res->status != 200) return 0;

        auto j = nlohmann::json::parse(res->body);
        if (!j.is_array()) return 0;

        int count = 0;
        for (const auto & item : j) {
            CloudTask t;
            std::string type_str = item.value("type", "");
            if (type_str == "upload_features")   t.type = TaskType::UploadFeatures;
            else if (type_str == "cloud_review")  t.type = TaskType::CloudReview;
            else if (type_str == "model_update")  t.type = TaskType::ModelUpdate;
            else if (type_str == "rule_sync")     t.type = TaskType::RuleSync;
            else continue;  // unknown type → skip

            t.task_id      = item.value("task_id", "");
            t.description  = item.value("description", "");
            t.priority     = item.value("priority", 0);
            t.payload_json = item.value("payload", nlohmann::json::object()).dump();

            enqueue(t);
            ++count;
        }
        return count;
    } catch (...) {
        return 0;
    }
}

// ═══════════════════════════════════════════════════════════════
// Local triggers
// ═══════════════════════════════════════════════════════════════

bool TaskScheduler::request_cloud_review(const std::string & context_json,
                                          const InferenceContext & ictx) {
    if (!_cloud || !_cloud->is_reachable()) return false;

    // Build messages JSON from context
    _cloud->query_offload(
        context_json, ictx,
        [this](const std::string & text, bool finished, const std::string & err) {
            // Forward to fused result callback when complete
            if (finished) {
                FusedResult fr;
                fr.final_text = text;
                fr.source     = err.empty() ? "cloud" : "local";
                fr.confidence = err.empty() ? 0.9 : 0.5;
                if (on_fused_result) on_fused_result(fr);
            }
        },
        nullptr  // no per-perf callback for now
    );
    return true;
}

// ═══════════════════════════════════════════════════════════════
// Feature extraction
// ═══════════════════════════════════════════════════════════════

FeatureVector TaskScheduler::extract_features(int /*layer_idx*/) const {
    FeatureVector fv;
    if (!_engine) return fv;

    // For a first implementation, extract the output logits/embeddings
    // rather than intermediate layer hidden states (which require
    // llama.cpp internals modifications).
    //
    // This provides the cloud with a compressed representation of
    // what the edge model "sees", enabling cloud-side re-analysis
    // without sending raw input data.
    //
    // Future optimization: add true layer-split extraction.

    fv.layer_name = "output_embeddings";
    fv.dtype      = "float32";
    // Values are populated by the engine's export_hidden_states()
    // when that API is implemented (Phase 4).
    return fv;
}

// ═══════════════════════════════════════════════════════════════
// Result fusion
// ═══════════════════════════════════════════════════════════════

FusedResult TaskScheduler::fuse(const std::string & local_result,
                                 const std::string & cloud_result,
                                 double local_conf,
                                 double cloud_conf) const {
    FusedResult fr;

    if (cloud_result.empty()) {
        fr.final_text  = local_result;
        fr.source      = "local";
        fr.confidence  = local_conf;
        fr.components  = {local_result};
        return fr;
    }

    if (local_result.empty()) {
        fr.final_text  = cloud_result;
        fr.source      = "cloud";
        fr.confidence  = cloud_conf;
        fr.components  = {cloud_result};
        return fr;
    }

    // Both available — weighted fusion
    double total_conf = local_conf + cloud_conf;
    double alpha = total_conf > 0 ? local_conf / total_conf : 0.5;

    fr.final_text  = local_result;   // local as base
    fr.source      = "fused";
    fr.confidence  = std::max(local_conf, cloud_conf);
    fr.components  = {local_result, cloud_result};

    // If cloud confidence is significantly higher, prefer cloud result
    if (cloud_conf > local_conf * 1.5) {
        fr.final_text = cloud_result;
    }

    return fr;
}

// ═══════════════════════════════════════════════════════════════
// Task execution (dispatches to EdgeCloudClient)
// ═══════════════════════════════════════════════════════════════

void TaskScheduler::execute_task(const CloudTask & task) {
    TaskResult result;
    result.task_id = task.task_id;

    switch (task.type) {
    case TaskType::UploadFeatures:
        result = run_feature_upload(task);
        break;
    case TaskType::CloudReview:
        result = run_cloud_review(task);
        break;
    case TaskType::ModelUpdate:
        // Forwarded to ModelSyncManager (Phase 4 integration)
        result.success = true;
        result.result_json = R"({"status":"forwarded_to_model_sync"})";
        break;
    case TaskType::RuleSync:
        // Forwarded to ModelSyncManager (Phase 4 integration)
        result.success = true;
        result.result_json = R"({"status":"forwarded_to_rule_sync"})";
        break;
    default:
        result.success = false;
        result.error_msg = "Unknown task type";
        break;
    }

    if (on_task_complete) on_task_complete(result);
}

TaskResult TaskScheduler::run_feature_upload(const CloudTask & task) {
    TaskResult r;
    r.task_id = task.task_id;

    if (!_cloud || !_cloud->is_reachable()) {
        r.success = false;
        r.error_msg = "Cloud unreachable";
        return r;
    }

    // Parse task payload for feature extraction parameters
    try {
        auto payload = nlohmann::json::parse(task.payload_json);
        int layer_idx = payload.value("layer_idx", -1);
        auto fv = extract_features(layer_idx);

        // Build MultimodalFeatures from feature vector
        MultimodalFeatures media;
        media.media_type      = "feature_vector";
        media.visual_features = fv.values;
        media.feature_shape   = fv.shape;

        InferenceContext ictx;
        ictx.temperature = 0.7f;
        ictx.max_new_tokens = 256;

        // Capture by shared_ptr: feature_offload runs async, so [&r] would dangle.
        auto result_ptr = std::make_shared<TaskResult>(r);

        _cloud->feature_offload(
            "Analyze these edge features and provide a second opinion.",
            media, ictx,
            [this, result_ptr](const std::string & text, bool finished,
                               const std::string & err) {
                if (finished) {
                    result_ptr->success = err.empty();
                    result_ptr->result_json = text;
                    result_ptr->error_msg   = err;
                    if (on_task_complete) on_task_complete(*result_ptr);
                }
            },
            nullptr);

        r.success = false;
        r.error_msg = "async: result will arrive via callback";
    } catch (...) {
        r.success = false;
        r.error_msg = "Failed to parse feature upload payload";
    }

    return r;
}

TaskResult TaskScheduler::run_cloud_review(const CloudTask & task) {
    TaskResult r;
    r.task_id = task.task_id;

    if (!_cloud || !_cloud->is_reachable()) {
        r.success = false;
        r.error_msg = "Cloud unreachable";
        return r;
    }

    try {
        auto payload = nlohmann::json::parse(task.payload_json);
        // Extract messages — if present as JSON array, dump it; otherwise wrap in array
        nlohmann::json msgs = payload.value("messages", nlohmann::json::array());
        std::string messages_json = msgs.dump();

        InferenceContext ictx;
        ictx.temperature    = payload.value("temperature", 0.7);
        ictx.max_new_tokens = payload.value("max_tokens", 512);

        auto result_ptr = std::make_shared<TaskResult>(r);

        _cloud->query_offload(
            messages_json, ictx,
            [this, result_ptr](const std::string & text, bool finished,
                               const std::string & err) {
                if (finished) {
                    result_ptr->success = err.empty();
                    result_ptr->result_json = text;
                    result_ptr->error_msg   = err;
                    if (on_task_complete) on_task_complete(*result_ptr);
                }
            },
            nullptr);

        r.success = false;
        r.error_msg = "async: result will arrive via callback";
    } catch (...) {
        r.success = false;
        r.error_msg = "Failed to parse cloud review payload";
    }

    return r;
}
