#include "cloud.h"
#include "http.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

EdgeCloud::EdgeCloud()  = default;
EdgeCloud::~EdgeCloud() { stop(); }

void EdgeCloud::set_endpoint(const std::string & url, const std::string & api_key) {
    _endpoint = url;
    _api_key  = api_key;
}

void EdgeCloud::set_model_name(const std::string & name) {
    _model_name = name;
}

bool EdgeCloud::check_reachable() {
    try {
        auto [cli, parts] = common_http_client(_endpoint);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(2, 0);
        auto res = cli.Get(parts.path.c_str());
        _reachable.store(res != nullptr);
    } catch (...) {
        _reachable.store(false);
    }
    return _reachable.load();
}

bool EdgeCloud::is_reachable() const {
    return _reachable.load();
}

void EdgeCloud::send(const std::string & messages_json,
                     cloud_token_callback on_token) {
    if (_running.load()) return;

    _stop_requested.store(false);
    _running.store(true);

    if (_thread.joinable()) _thread.join();
    _thread = std::thread(&EdgeCloud::run_thread, this, messages_json, on_token);
}

void EdgeCloud::stop() {
    _stop_requested.store(true);
    if (_thread.joinable()) {
        _thread.join();
    }
}

bool EdgeCloud::is_running() const {
    return _running.load();
}

void EdgeCloud::run_thread(std::string messages_json, cloud_token_callback on_token) {
    try {
        json body;
        body["model"]       = _model_name;
        body["messages"]    = json::parse(messages_json);
        body["temperature"] = temperature;
        body["max_tokens"]  = max_tokens;
        body["stream"]      = true;

        auto [cli, parts] = common_http_client(_endpoint);
        cli.set_connection_timeout(timeout_ms / 1000, 0);
        cli.set_read_timeout(0, 0);  // no read timeout for streaming

        if (!_api_key.empty()) {
            cli.set_bearer_token_auth(_api_key.c_str());
        }

                        httplib::Headers headers;
        httplib::ContentReceiver receiver = [&](const char * data, size_t len) -> bool {
            if (_stop_requested.load()) return false;

            std::string chunk(data, len);
            // Parse SSE lines: "data: {...}\n\n"
            size_t pos = 0;
            while (pos < chunk.size() && !_stop_requested.load()) {
                auto data_start = chunk.find("data: ", pos);
                if (data_start == std::string::npos) break;
                data_start += 6;  // skip "data: "
                auto data_end = chunk.find('\n', data_start);
                if (data_end == std::string::npos) break;

                std::string line = chunk.substr(data_start, data_end - data_start);
                pos = data_end + 1;

                if (line == "[DONE]") {
                    return true;
                }

                try {
                    auto j = json::parse(line);
                    auto choices = j.value("choices", json::array());
                    if (!choices.empty()) {
                        auto delta = choices[0].value("delta", json::object());
                        auto content = delta.value("content", "");
                        if (!content.empty()) {
                            on_token(content, false);
                        }
                    }
                } catch (...) {
                    // Skip malformed chunks
                                }
            }
            return true;
        };

        auto res = cli.Post(parts.path,
                            headers,
                            body.dump(),
                            std::string("application/json"),
                            receiver);

        if (!res) {
            on_token("\n[Cloud unreachable]", true);
            _reachable.store(false);
        }
    } catch (...) {
        on_token("\n[Cloud error]", true);
        _reachable.store(false);
    }

    on_token("", true);
    _running.store(false);
}
