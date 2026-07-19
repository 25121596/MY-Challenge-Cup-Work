#pragma once

// ── Model parameter & decision rule sync (Module 2.3) ────────────
//
// Handles incremental model updates (LoRA hot-loading) and global
// decision-rule synchronization pushed from the cloud.
//
// Dependencies:
//   - llama.h   (llama_adapter_lora_init / llama_set_adapters_lora)
//   - EdgeEngine (raw_model / raw_context access)

#include "llama.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <map>

// Forward declaration
class EdgeEngine;

// ═══════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════

// A single decision rule pushed from cloud
struct DecisionRule {
    std::string rule_id;
    std::string rule_type;       // "threshold", "priority", "override"
    std::string target;          // e.g. "anomaly_score", "class_priority"
    std::string value_json;      // the new value / configuration
    int64_t     version    = 0;
    int64_t     effective_at_ms = 0;  // when to apply (0 = immediate)
};

// A LoRA update package from cloud
struct LoRAUpdate {
    std::string update_id;
    std::string lora_file_url;   // download URL
    std::string lora_file_path;  // local cache path after download
    float       scale       = 1.0f;
    int64_t     version     = 0;
    std::string checksum_md5;
};

// ═══════════════════════════════════════════════════════════════
// Model sync manager
// ═══════════════════════════════════════════════════════════════

class ModelSyncManager {
public:
    ModelSyncManager();
    ~ModelSyncManager();

    // ── Pointers (set before use) ──────────────────────────────
    void set_engine(EdgeEngine * engine) { _engine = engine; }
    void set_cloud_endpoint(const std::string & url) { _base_url = url; }

    // ── LoRA ──────────────────────────────────────────────────
    // Download and hot-load a LoRA adapter. Returns "" on success.
    std::string download_and_apply_lora(const LoRAUpdate & update);

    // Remove all active LoRA adapters, revert to base model.
    std::string revert_loras();

    // Version tracking
    int64_t current_lora_version() const { return _current_lora_version; }

    // ── Decision rules ────────────────────────────────────────
    // Apply a rule immediately to the local decision engine
    void apply_rule(const DecisionRule & rule);

    // ── Cloud polling ─────────────────────────────────────────
    // Pending updates (LoRA + rules) from cloud
    struct PendingUpdates {
        std::vector<LoRAUpdate>    lora_updates;
        std::vector<DecisionRule>  rule_updates;
    };
    PendingUpdates poll_updates();

    // ── Active rules (exposed for ConflictDetector integration) ─
    const std::vector<DecisionRule> & active_rules() const { return _active_rules; }

    // ── Callbacks ──────────────────────────────────────────────
    std::function<void(const std::string & status)> on_status;

private:
    bool download_file(const std::string & url, const std::string & local_path);

    EdgeEngine * _engine = nullptr;
    std::string  _base_url;

    // LoRA state (per-adapter scale)
    struct LoraEntry {
        llama_adapter_lora * adapter = nullptr;
        float scale = 1.0f;
    };
    std::vector<LoraEntry> _loaded_loras;
    int64_t _current_lora_version = 0;

    // Rules
    std::vector<DecisionRule> _active_rules;

    std::mutex _mutex;
};
