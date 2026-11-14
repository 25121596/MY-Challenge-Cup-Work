#include "model_sync.h"
#include "engine.h"

#include "http.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════

ModelSyncManager::ModelSyncManager()  = default;
ModelSyncManager::~ModelSyncManager() { revert_loras(); }

// ═══════════════════════════════════════════════════════════════
// File download
// ═══════════════════════════════════════════════════════════════

bool ModelSyncManager::download_file(const std::string & url,
                                      const std::string & local_path) {
    try {
        auto [cli, parts] = common_http_client(url);
        cli.set_connection_timeout(30, 0);
        cli.set_read_timeout(120, 0);

        std::ofstream out(local_path, std::ios::binary);
        if (!out) return false;

        bool success = false;
        auto receiver = [&](const char * data, size_t len) -> bool {
            out.write(data, len);
            return !out.fail();
        };

        auto res = cli.Get(parts.path, httplib::Headers{}, receiver);
        out.close();
        success = (res && res->status == 200);

        if (!success) {
            std::remove(local_path.c_str());
        }
        return success;
    } catch (...) {
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════
// LoRA hot-load
// ═══════════════════════════════════════════════════════════════

std::string ModelSyncManager::download_and_apply_lora(const LoRAUpdate & update) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_engine) return "No engine set";

    // Step 1: Download the LoRA file
    std::string local_path = update.lora_file_path;
    if (local_path.empty()) {
        // Place in models/loras/
        local_path = "models/loras/" + update.update_id + ".gguf";
    }

    if (!update.lora_file_url.empty()) {
        if (on_status) on_status("Downloading LoRA: " + update.lora_file_url);
        if (!download_file(update.lora_file_url, local_path)) {
            return "Failed to download LoRA from: " + update.lora_file_url;
        }
    }

    if (on_status) on_status("Applying LoRA: " + local_path);

    // Step 2: Load the LoRA adapter
    llama_model * model = _engine->raw_model();
    if (!model) return "Engine has no loaded model";

    llama_adapter_lora * adapter = llama_adapter_lora_init(model, local_path.c_str());
    if (!adapter) {
        return "llama_adapter_lora_init failed for: " + local_path;
    }

    // Step 3: Apply to context (hot-swap) with per-adapter scales
    _loaded_loras.push_back({adapter, update.scale});

    llama_context * ctx = _engine->raw_context();
    if (ctx) {
        std::vector<llama_adapter_lora *> adapters;
        std::vector<float> scales;
        for (const auto & e : _loaded_loras) {
            adapters.push_back(e.adapter);
            scales.push_back(e.scale);
        }
        int ret = llama_set_adapters_lora(
            ctx, adapters.data(), adapters.size(), scales.data());
        if (ret != 0) {
            return "llama_set_adapters_lora returned error code: " +
                   std::to_string(ret);
        }
    }

    _current_lora_version = update.version;

    if (on_status) on_status("LoRA applied successfully (v" +
                             std::to_string(update.version) + ")");
    return "";
}

std::string ModelSyncManager::revert_loras() {
    std::lock_guard<std::mutex> lock(_mutex);

    // Clear from context
    if (_engine) {
        llama_context * ctx = _engine->raw_context();
        if (ctx) {
            llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
        }
    }

    // Free adapters
    for (auto & e : _loaded_loras) {
        llama_adapter_lora_free(e.adapter);
    }
    _loaded_loras.clear();
    _current_lora_version = 0;

    if (on_status) on_status("LoRA adapters removed");
    return "";
}

// ═══════════════════════════════════════════════════════════════
// Decision rules
// ═══════════════════════════════════════════════════════════════

void ModelSyncManager::apply_rule(const DecisionRule & rule) {
    std::lock_guard<std::mutex> lock(_mutex);

    // Update or add the rule
    auto it = std::find_if(_active_rules.begin(), _active_rules.end(),
        [&](const DecisionRule & r) { return r.rule_id == rule.rule_id; });
    if (it != _active_rules.end()) {
        *it = rule;
    } else {
        _active_rules.push_back(rule);
    }

    if (on_status) {
        on_status("Rule applied: " + rule.rule_id + " (" + rule.rule_type + ")");
    }
}

// ═══════════════════════════════════════════════════════════════
// Cloud polling
// ═══════════════════════════════════════════════════════════════

ModelSyncManager::PendingUpdates ModelSyncManager::poll_updates() {
    PendingUpdates result;
    if (_base_url.empty()) return result;

    std::string url = _base_url +
        "/api/v1/edge/model-updates"
        "?current_lora_version=" + std::to_string(_current_lora_version);

    try {
        auto [cli, parts] = common_http_client(url);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(5, 0);

        auto res = cli.Get(parts.path.c_str());
        if (!res || res->status != 200) return result;

        auto j = nlohmann::json::parse(res->body);

        // Parse LoRA updates
        if (j.contains("lora_updates")) {
            for (const auto & item : j["lora_updates"]) {
                LoRAUpdate lu;
                lu.update_id     = item.value("update_id", "");
                lu.lora_file_url = item.value("url", "");
                lu.lora_file_path = item.value("local_path", "");
                lu.scale         = item.value("scale", 1.0f);
                lu.version       = item.value("version", 0);
                lu.checksum_md5  = item.value("checksum", "");
                result.lora_updates.push_back(lu);
            }
        }

        // Parse rule updates
        if (j.contains("rule_updates")) {
            for (const auto & item : j["rule_updates"]) {
                DecisionRule dr;
                dr.rule_id   = item.value("rule_id", "");
                dr.rule_type = item.value("rule_type", "");
                dr.target    = item.value("target", "");
                dr.value_json = item.value("value", nlohmann::json::object()).dump();
                dr.version   = item.value("version", 0);
                dr.effective_at_ms = item.value("effective_at_ms", 0);
                result.rule_updates.push_back(dr);
            }
        }
    } catch (...) {
        // Network error — return empty
    }

    return result;
}
