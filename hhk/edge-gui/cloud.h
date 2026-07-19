#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

// Callback for streaming tokens, same signature as engine callback.
using cloud_token_callback = std::function<void(const std::string & text, bool finished)>;

// Lightweight HTTP client for offloading requests to a cloud LLM endpoint
// (OpenAI-compatible /chat/completions). Falls back gracefully when unreachable.
class EdgeCloud {
public:
    EdgeCloud();
    ~EdgeCloud();

    void set_endpoint(const std::string & url, const std::string & api_key = "");
    void set_model_name(const std::string & model_name);

    // Try to reach the endpoint. Returns true if reachable.
    bool check_reachable();
    bool is_reachable() const;

    // Send a chat message asynchronously.
    // Messages are provided as a JSON array string (OpenAI format).
    void send(const std::string & messages_json,
              cloud_token_callback on_token);

    void stop();
    bool is_running() const;

    // Configuration
    int timeout_ms = 5000;
    float temperature = 0.7f;
    int   max_tokens   = 512;

private:
    void run_thread(std::string messages_json, cloud_token_callback on_token);

    std::string _endpoint;
    std::string _api_key;
    std::string _model_name = "default";

    std::mutex _mutex;
    std::thread _thread;
    std::atomic<bool> _running{false};
    std::atomic<bool> _stop_requested{false};
    std::atomic<bool> _reachable{false};
};
