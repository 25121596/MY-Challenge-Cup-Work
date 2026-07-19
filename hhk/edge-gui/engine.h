#pragma once

#include "llama.h"
#include "common.h"
#include "chat.h"
#include "sampling.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

// ── Callback types ────────────────────────────────────────────

// Invoked when a new token is generated.
// text     - the decoded token text
// finished - true if generation completed (EOS or stop)
using engine_token_callback = std::function<void(const std::string & text, bool finished)>;

// Invoked when model load progress updates.
// progress - 0.0 .. 1.0
// stage    - human-readable description of current stage
using engine_progress_callback = std::function<void(float progress, const std::string & stage)>;

// ── Performance snapshot ──────────────────────────────────────

struct EnginePerf {
    double pp_tokens_per_sec = 0;   // prompt processing speed
    double tg_tokens_per_sec = 0;   // token generation speed
    double ttft_ms           = 0;   // time-to-first-token
    int32_t n_p_tokens       = 0;   // prompt tokens processed
    int32_t n_g_tokens       = 0;   // generated tokens
    bool    valid            = false;
};

// ── Local inference engine ────────────────────────────────────
//
// Owns the model, context, sampler, and chat templates.
// Runs generation on a background thread to keep the GUI responsive.
class EdgeEngine {
public:
    EdgeEngine();
    ~EdgeEngine();

    // ── lifecycle ─────────────────────────────────────────────

    // Load a model from a GGUF file path.
    // Returns {true, ""} on success, {false, error_msg} on failure.
    std::pair<bool, std::string> load_model(
        const std::string & model_path,
        int n_gpu_layers = 99,
        engine_progress_callback on_progress = nullptr);

    // Release all resources (model, context, templates, sampler).
    void unload();

    bool is_loaded() const;

    // ── inference ─────────────────────────────────────────────

    // Begin generating a response for the given prompt.
    // on_token is invoked from a worker thread for each token.
    // on_perf is invoked on completion with performance data.
    bool generate(const std::string & prompt,
                  const engine_token_callback & on_token,
                  std::function<void(const EnginePerf &)> on_perf = nullptr);

    // Request the current generation to stop as soon as possible.
    void stop();

    bool is_running() const;

    // ── chat template ─────────────────────────────────────────

    // Are chat templates available from the loaded model?
    bool has_chat_template() const;

    // Render a list of common_chat_msg into a formatted prompt string.
    // Returns the rendered prompt ready for tokenization.
    std::string render_chat_prompt(
        const std::vector<common_chat_msg> & messages,
        bool add_generation_prompt = true) const;

    // ── KV cache / session ────────────────────────────────────

    // Save the current KV-cache state + token history to a file.
    bool save_session(const std::string & filepath,
                      const std::vector<llama_token> & prompt_tokens);

    // Restore KV-cache state from a file.  Returns the saved token list.
    bool load_session(const std::string & filepath,
                      std::vector<llama_token> & prompt_tokens,
                      int32_t n_token_capacity);

    // Clear all KV-cache state (start a fresh conversation).
    void clear_kv_cache();

    // Current number of tokens in the cache.
    int32_t token_count() const { return _n_past; }

    // ── parameters ────────────────────────────────────────────

    int32_t n_predict      = 512;    // max tokens to generate
    int32_t n_ctx           = 2048;   // context size
    int32_t n_keep          = 4;      // tokens to keep during context shift
    int32_t n_batch         = 512;    // batch size
    float   temperature     = 0.7f;
    float   top_p           = 0.9f;
    float   min_p           = 0.1f;
    int32_t top_k           = 40;
    float   repeat_penalty  = 1.0f;   // 1.0 = disabled
    float   dry_multiplier  = 0.0f;   // 0.0 = disabled
    float   xtc_probability = 0.0f;   // 0.0 = disabled
    uint32_t seed           = 42;

    // ── info ──────────────────────────────────────────────────

    const std::string & model_path()   const { return _model_path; }
    size_t              model_size()   const { return _model_size; }
    size_t              model_params() const { return _n_params; }

    // ── LoRA adapter management (Module 2.3) ──────────────────

    // Hot-load a LoRA adapter from file.  Can be called while model
    // is loaded.  Returns "" on success, error message on failure.
    std::string load_lora(const std::string & lora_path, float scale = 1.0f);

    // Remove all active LoRA adapters, reverting to base model.
    std::string clear_loras();

    // ── Raw accessors (for ModelSyncManager / TaskScheduler) ──
    llama_model   * raw_model()   const { return _model; }
    llama_context * raw_context() const { return _ctx; }

    // ── Model info ────────────────────────────────────────────
    int32_t model_n_layers() const;
    int32_t model_n_embd()  const;

private:
    void run_loop(std::string prompt,
                  engine_token_callback on_token,
                  std::function<void(const EnginePerf &)> on_perf);

    // Rebuild the sampler chain from current parameter values.
    void rebuild_sampler();
    bool sampler_needs_rebuild() const;

    llama_model    * _model   = nullptr;
    llama_context  * _ctx     = nullptr;
    const llama_vocab * _vocab = nullptr;
    common_chat_templates_ptr _tmpls;
    common_sampler * _gsmpl   = nullptr;

    // LoRA adapters (hot-loadable) with per-adapter scales
    struct LoraEntry {
        llama_adapter_lora * adapter = nullptr;
        float scale = 1.0f;
    };
    std::vector<LoraEntry> _loaded_loras;

    std::string      _model_path;
    size_t           _model_size = 0;
    size_t           _n_params   = 0;

    // KV cache tracking
    int32_t             _n_past = 0;      // total tokens in cache
    std::vector<llama_token> _cached_tokens;  // all tokens in the current session

    // Sampler parameter snapshot for change detection
    struct SamplerSnapshot {
        float   temp, top_p, min_p, repeat_penalty, dry_mult, xtc_prob;
        int32_t top_k;
        uint32_t seed;
        bool operator!=(const SamplerSnapshot & o) const;
    } _sampler_snap = {};

    std::thread        _thread;
    std::atomic<bool>  _running{false};
    std::atomic<bool>  _stop_requested{false};
    mutable std::mutex _mutex;
};
