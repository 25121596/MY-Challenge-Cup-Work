#include "engine.h"

#include "sampling.h"

#include <algorithm>
#include <cstring>
#include <chrono>

EdgeEngine::EdgeEngine()  = default;
EdgeEngine::~EdgeEngine() { unload(); }

// ── Sampler snapshot comparison ───────────────────────────────

bool EdgeEngine::SamplerSnapshot::operator!=(const SamplerSnapshot & o) const {
    return temp != o.temp || top_p != o.top_p || min_p != o.min_p ||
           repeat_penalty != o.repeat_penalty || dry_mult != o.dry_mult ||
           xtc_prob != o.xtc_prob || top_k != o.top_k || seed != o.seed;
}

bool EdgeEngine::sampler_needs_rebuild() const {
    SamplerSnapshot cur;
    cur.temp            = temperature;
    cur.top_p           = top_p;
    cur.min_p           = min_p;
    cur.repeat_penalty  = repeat_penalty;
    cur.dry_mult        = dry_multiplier;
    cur.xtc_prob        = xtc_probability;
    cur.top_k           = top_k;
    cur.seed            = seed;
    return cur != _sampler_snap;
}

void EdgeEngine::rebuild_sampler() {
    if (_gsmpl) {
        common_sampler_free(_gsmpl);
        _gsmpl = nullptr;
    }

    common_params_sampling sparams;
    sparams.temp             = temperature;
    sparams.top_k            = top_k;
    sparams.top_p            = top_p;
    sparams.min_p            = min_p;
    sparams.penalty_repeat   = repeat_penalty;
    sparams.seed             = seed;
    // Use the default sampler order from common_params_sampling,
    // with PENALTIES first, then DRY/TOP_N_SIGMA/TOP_K/TYPICAL_P/
    // TOP_P/MIN_P/XTC/TEMPERATURE.
    // The "distribution" sampler is always appended by common_sampler_init.
    sparams.samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };

    _gsmpl = common_sampler_init((const llama_model *)_model, sparams);

    // Update snapshot
    _sampler_snap.temp            = temperature;
    _sampler_snap.top_p           = top_p;
    _sampler_snap.min_p           = min_p;
    _sampler_snap.repeat_penalty  = repeat_penalty;
    _sampler_snap.dry_mult        = dry_multiplier;
    _sampler_snap.xtc_prob        = xtc_probability;
    _sampler_snap.top_k           = top_k;
    _sampler_snap.seed            = seed;
}

// ── Lifecycle ─────────────────────────────────────────────────

std::pair<bool, std::string> EdgeEngine::load_model(
    const std::string & model_path,
    int n_gpu_layers,
    engine_progress_callback on_progress)
{
    unload();

    _model_path = model_path;

    if (on_progress) on_progress(0.0f, "Initializing backends...");

    // Initialize backends
    ggml_backend_load_all();

    if (on_progress) on_progress(0.1f, "Loading model file...");

    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    _model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!_model) {
        return {false, "Failed to load model from: " + model_path +
                "\nCheck that the file exists and is a valid GGUF file."};
    }

    _vocab      = llama_model_get_vocab(_model);
    _n_params   = llama_model_n_params(_model);
    _model_size = llama_model_size(_model);

    if (on_progress) on_progress(0.4f, "Initializing chat templates...");

    // Extract chat template from GGUF metadata
    _tmpls = common_chat_templates_init(_model, "");

    if (on_progress) on_progress(0.5f, "Creating context...");

    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = n_ctx;
    cparams.n_batch = n_batch;

    _ctx = llama_init_from_model(_model, cparams);
    if (!_ctx) {
        llama_model_free(_model);
        _model = nullptr;
        return {false, "Failed to create inference context.\n"
                "Try reducing the context size or check available memory."};
    }

    if (on_progress) on_progress(0.8f, "Building sampler...");

    // Initialize sampler
    rebuild_sampler();

    // Clear KV cache tracking
    _n_past = 0;
    _cached_tokens.clear();

    if (on_progress) on_progress(1.0f, "Ready");

    return {true, ""};
}

void EdgeEngine::unload() {
    stop();
    if (_thread.joinable()) {
        _thread.join();
    }

    if (_gsmpl) {
        common_sampler_free(_gsmpl);
        _gsmpl = nullptr;
    }
    if (_tmpls) {
        common_chat_templates_free(_tmpls.release());
    }
    if (_ctx) {
        llama_free(_ctx);
        _ctx = nullptr;
    }
    // Free LoRA adapters before freeing the model
    for (auto & e : _loaded_loras) {
        llama_adapter_lora_free(e.adapter);
    }
    _loaded_loras.clear();

    if (_model) {
        llama_model_free(_model);
        _model = nullptr;
    }
    _model_path.clear();
    _model_size = 0;
    _n_params   = 0;
    _n_past     = 0;
    _cached_tokens.clear();
}

bool EdgeEngine::is_loaded() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _model != nullptr && _ctx != nullptr;
}

// ── Inference ─────────────────────────────────────────────────

bool EdgeEngine::generate(const std::string & prompt,
                          const engine_token_callback & on_token,
                          std::function<void(const EnginePerf &)> on_perf) {
    if (!is_loaded()) return false;
    if (_running.load()) return false;

    _stop_requested.store(false);
    _running.store(true);

    if (_thread.joinable()) _thread.join();
    _thread = std::thread(&EdgeEngine::run_loop, this, prompt, on_token, on_perf);
    return true;
}

void EdgeEngine::stop() {
    _stop_requested.store(true);
}

bool EdgeEngine::is_running() const {
    return _running.load();
}

// ── Chat template ─────────────────────────────────────────────

bool EdgeEngine::has_chat_template() const {
    return _tmpls != nullptr;
}

std::string EdgeEngine::render_chat_prompt(
    const std::vector<common_chat_msg> & messages,
    bool add_generation_prompt) const
{
    if (!_tmpls) {
        // Fallback: simple ChatML concatenation
        std::string prompt;
        for (const auto & msg : messages) {
            prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
        }
        if (add_generation_prompt) {
            prompt += "<|im_start|>assistant\n";
        }
        return prompt;
    }

    common_chat_templates_inputs inputs;
    inputs.messages              = messages;
    inputs.use_jinja             = true;
    inputs.add_generation_prompt = add_generation_prompt;

    auto chat_params = common_chat_templates_apply(_tmpls.get(), inputs);
    return chat_params.prompt;
}

// ── KV cache / session ────────────────────────────────────────

void EdgeEngine::clear_kv_cache() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_ctx) {
        auto * mem = llama_get_memory(_ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
        _n_past = 0;
        _cached_tokens.clear();
    }
}

bool EdgeEngine::save_session(const std::string & filepath,
                              const std::vector<llama_token> & prompt_tokens) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_ctx) return false;

    return llama_state_save_file(_ctx, filepath.c_str(),
                                 prompt_tokens.data(),
                                 (int64_t)prompt_tokens.size());
}

bool EdgeEngine::load_session(const std::string & filepath,
                              std::vector<llama_token> & prompt_tokens,
                              int32_t n_token_capacity) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_ctx) return false;

    prompt_tokens.resize(n_token_capacity);
    size_t n_out = 0;
    bool ok = llama_state_load_file(_ctx, filepath.c_str(),
                                    prompt_tokens.data(),
                                    (size_t)n_token_capacity, &n_out);
    if (ok) {
        prompt_tokens.resize((size_t)n_out);
        _cached_tokens = prompt_tokens;
        _n_past = (int32_t)n_out;
    }
    return ok;
}

// ── Inference loop ────────────────────────────────────────────

void EdgeEngine::run_loop(std::string prompt,
                          engine_token_callback on_token,
                          std::function<void(const EnginePerf &)> on_perf) {
    using Clock = std::chrono::steady_clock;
    auto t_start = Clock::now();

    // ── Tokenize prompt ──────────────────────────────────────
    const int n_prompt = -llama_tokenize(
        _vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        _running.store(false);
        on_token("[Error: tokenization failed]", true);
        return;
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(_vocab, prompt.c_str(), prompt.size(),
                   prompt_tokens.data(), n_prompt, true, true);

    // ── Decode prompt ────────────────────────────────────────
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);

    if (llama_decode(_ctx, batch) != 0) {
        _running.store(false);
        on_token("[Error: prompt decode failed]", true);
        return;
    }

    _n_past += n_prompt;
    _cached_tokens.insert(_cached_tokens.end(),
                          prompt_tokens.begin(), prompt_tokens.end());

    auto t_first_token = Clock::now();
    double ttft_ms = std::chrono::duration<double, std::milli>(
        t_first_token - t_start).count();

    // Rebuild sampler if needed
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (sampler_needs_rebuild()) {
            rebuild_sampler();
        }
    }

    // ── Generate tokens ──────────────────────────────────────
    llama_token new_token_id;
    int n_generated = 0;

    while (n_generated < n_predict && !_stop_requested.load()) {
        // ── Context shift if needed ──────────────────────────
        if (_n_past + 1 >= n_ctx) {
            auto * mem = llama_get_memory(_ctx);
            if (mem && llama_memory_can_shift(mem)) {
                int32_t n_discard = (_n_past - n_keep) / 2;
                if (n_discard > 0) {
                    llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
                    llama_memory_seq_add(mem, 0, n_keep + n_discard, _n_past, -n_discard);

                    // Mirror changes in our token cache
                    _cached_tokens.erase(
                        _cached_tokens.begin() + n_keep,
                        _cached_tokens.begin() + n_keep + n_discard);
                    _n_past -= n_discard;
                }
            } else {
                // Cannot shift — stop generation to avoid undefined behavior
                on_token("\n[Context limit reached]", true);
                break;
            }
        }

        // ── Sample ───────────────────────────────────────────
        {
            std::lock_guard<std::mutex> lock(_mutex);
            new_token_id = common_sampler_sample(_gsmpl, _ctx, -1, false);
        }

        if (_vocab && llama_vocab_is_eog(_vocab, new_token_id)) {
            common_sampler_accept(_gsmpl, new_token_id, true);
            break;
        }

        // Detokenize
        char buf[256];
        int n = llama_token_to_piece(_vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            on_token(std::string(buf, n), false);
        }

        // Accept and decode next
        common_sampler_accept(_gsmpl, new_token_id, true);

        batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(_ctx, batch) != 0) {
            break;
        }

        _n_past++;
        _cached_tokens.push_back(new_token_id);
        ++n_generated;
    }

    // ── Perf ─────────────────────────────────────────────────
    auto t_end = Clock::now();

    if (on_perf) {
        EnginePerf perf;
        perf.valid = true;

        auto pdata = llama_perf_context(_ctx);
        perf.n_p_tokens = pdata.n_p_eval;
        perf.n_g_tokens = pdata.n_eval;
        perf.ttft_ms    = ttft_ms;

        double pp_ms = pdata.t_p_eval_ms;
        double tg_ms = pdata.t_eval_ms;
        perf.pp_tokens_per_sec = pp_ms > 0 ? (pdata.n_p_eval / pp_ms * 1000.0) : 0;
        perf.tg_tokens_per_sec = tg_ms > 0 ? (pdata.n_eval / tg_ms * 1000.0) : 0;

        llama_perf_context_reset(_ctx);
        on_perf(perf);
    }

    on_token("", true);
    _running.store(false);
}

// ── LoRA adapter management ─────────────────────────────────────

std::string EdgeEngine::load_lora(const std::string & lora_path, float scale) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_model) return "No model loaded";

    llama_adapter_lora * adapter = llama_adapter_lora_init(_model, lora_path.c_str());
    if (!adapter) return "Failed to load LoRA from: " + lora_path;

    _loaded_loras.push_back({adapter, scale});

    if (_ctx) {
        std::vector<llama_adapter_lora *> adapters;
        std::vector<float> scales;
        for (const auto & e : _loaded_loras) {
            adapters.push_back(e.adapter);
            scales.push_back(e.scale);
        }
        int ret = llama_set_adapters_lora(
            _ctx, adapters.data(), adapters.size(), scales.data());
        if (ret != 0) {
            return "llama_set_adapters_lora returned: " + std::to_string(ret);
        }
    }

    return "";
}

std::string EdgeEngine::clear_loras() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_ctx) {
        llama_set_adapters_lora(_ctx, nullptr, 0, nullptr);
    }
    for (auto & e : _loaded_loras) {
        llama_adapter_lora_free(e.adapter);
    }
    _loaded_loras.clear();
    return "";
}

int32_t EdgeEngine::model_n_layers() const {
    if (!_model) return 0;
    return llama_model_n_layer(_model);
}

int32_t EdgeEngine::model_n_embd() const {
    if (!_model) return 0;
    return llama_model_n_embd(_model);
}
