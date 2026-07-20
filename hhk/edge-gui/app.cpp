#include "app.h"
#include "file_dialog.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "common.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

// ── JSON 消息构建（用于云端请求）────────────────────────────────
static std::string build_messages_json(const std::deque<common_chat_msg> & msgs) {
    std::string json = "[";
    for (size_t i = 0; i < msgs.size(); ++i) {
        if (i > 0) json += ",";
        json += msgs[i].to_json_oaicompat(false).dump();
    }
    json += "]";
    return json;
}

const char * mem_fit_str(MemFit f) {
    switch (f) {
        case MemFit::Good:  return "\xe5\x85\x85\xe8\xa3\x95";   // 充裕
        case MemFit::Tight: return "\xe4\xb8\xb4\xe7\x95\x8c";   // 临界
        case MemFit::Over:  return "\xe8\xb6\x85\xe9\x99\x90";   // 超限
    }
    return "";
}

// ── 构造 / 析构 ─────────────────────────────────────────────

EdgeApp::EdgeApp()
    : _task_scheduler()
{
    memset(_input_buf, 0, sizeof(_input_buf));
    _models_dir = detect_project_root() + "/models";
    if (_models_dir.empty() || _models_dir == "/models") {
        _models_dir = "./models";
    }
    strncpy(_models_dir_buf, _models_dir.c_str(), sizeof(_models_dir_buf) - 1);

    // ── Wire cloud-edge subsystems ───────────────────────────
    _task_scheduler.set_cloud(&_cloud_client);
    _task_scheduler.set_engine(&_engine);
    _model_sync.set_engine(&_engine);
    _conflict_detector.set_cloud_client(&_cloud_client);
    _conflict_detector.set_node_id(_p2p_node_id);

    // P2P → conflict detector
    _p2p.on_perception = [this](const PerceptionReport & r) {
        _conflict_detector.receive_peer_perception(r);
        on_peer_perception(r);
    };
    _p2p.on_decision_intent = [this](const DecisionIntent & i) {
        _conflict_detector.receive_peer_intent(i);
        on_peer_intent(i);
    };

    // Conflict detector → UI
    _conflict_detector.on_conflict_detected = [this](const ConflictRecord & c) {
        on_conflict(c);
    };
    _conflict_detector.on_conflict_resolved = [this](const ConflictRecord & c) {
        on_conflict_done(c);
    };

    // Heartbeat status
    _heartbeat.on_status = [](bool ok, int lat_ms, const std::string & msg) {
        if (!ok) fprintf(stderr, "Heartbeat failed: %s\n", msg.c_str());
    };

    // Task scheduler → forward fused results
    _task_scheduler.on_task_complete = [this](const TaskResult & r) {
        if (!r.success) {
            fprintf(stderr, "Task %s failed: %s\n",
                    r.task_id.c_str(), r.error_msg.c_str());
        }
    };
}

EdgeApp::~EdgeApp() { shutdown(); }

std::string EdgeApp::load_model(const std::string & path) {
    _model_error.clear();
    auto [ok, err] = _engine.load_model(path, 99,
        [this](float progress, const std::string & stage) {
            _model_load_prog = progress;
            _model_load_stage = stage;
        });
    _model_loaded = ok;
    if (!ok) _model_error = err;
    return ok ? "" : err;
}

void EdgeApp::shutdown() {
    _engine.stop();
    _cloud.stop();
    _cloud_client.stop();
    _heartbeat.stop();
    _task_scheduler.stop();
    _p2p.stop();
    if (_model_load_thread.joinable()) _model_load_thread.join();
}

// ── 内存估算 ─────────────────────────────────────────────────

size_t EdgeApp::estimate_runtime_mem(size_t model_file_size) const {
    int32_t ctx = _engine.n_ctx > 0 ? _engine.n_ctx : 2048;
    size_t kv_overhead = ctx * 512;
    return model_file_size + kv_overhead;
}

MemFit EdgeApp::check_model_fit(const ModelEntry & entry) {
    if (_avail_mem_mb <= 0) return MemFit::Good;
    size_t est = estimate_runtime_mem(entry.file_size_bytes);
    size_t avail = (size_t)_avail_mem_mb * 1024 * 1024;
    if (est <= avail * 0.8)  return MemFit::Good;
    if (est <= avail)         return MemFit::Tight;
    return MemFit::Over;
}

// ── 项目根目录检测 ───────────────────────────────────────────

std::string EdgeApp::detect_project_root() {
#ifdef _WIN32
    char exe_path[MAX_PATH] = {};
    DWORD len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return {};

    std::string dir(exe_path, len);
    for (auto & c : dir) if (c == '\\') c = '/';
    auto pos = dir.rfind('/');
    if (pos == std::string::npos) return {};
    dir = dir.substr(0, pos);

    for (int i = 0; i < 10; i++) {
        std::string cmake_test = dir + "/CMakeLists.txt";
        std::string ggml_test  = dir + "/ggml";
        DWORD cmake_attr = GetFileAttributesA(cmake_test.c_str());
        DWORD ggml_attr  = GetFileAttributesA(ggml_test.c_str());
        bool has_cmake = (cmake_attr != INVALID_FILE_ATTRIBUTES);
        bool has_ggml  = (ggml_attr  != INVALID_FILE_ATTRIBUTES) && (ggml_attr & FILE_ATTRIBUTE_DIRECTORY);
        if (has_cmake && has_ggml) return dir;
        pos = dir.rfind('/');
        if (pos == std::string::npos) break;
        dir = dir.substr(0, pos);
    }
    // 兜底: 向上走4层
    dir.assign(exe_path, len);
    for (auto & c : dir) if (c == '\\') c = '/';
    pos = dir.rfind('/');
    if (pos != std::string::npos) dir = dir.substr(0, pos);
    for (int i = 0; i < 4; i++) {
        pos = dir.rfind('/');
        if (pos == std::string::npos) break;
        dir = dir.substr(0, pos);
    }
    return dir;
#else
    char buf[4096] = {};
#ifdef __APPLE__
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return {};
#else
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
#endif
    std::string dir(buf);
    for (auto & c : dir) if (c == '\\') c = '/';
    auto pos = dir.rfind('/');
    if (pos != std::string::npos) dir = dir.substr(0, pos);
    for (int i = 0; i < 10; i++) {
        struct stat st;
        bool has_ggml   = (stat((dir + "/ggml").c_str(), &st)   == 0 && S_ISDIR(st.st_mode));
        bool has_src    = (stat((dir + "/src").c_str(), &st)    == 0 && S_ISDIR(st.st_mode));
        bool has_models = (stat((dir + "/models").c_str(), &st) == 0 && S_ISDIR(st.st_mode));
        if ((has_ggml && has_src) || (has_ggml && has_models) || (has_src && has_models)) return dir;
        pos = dir.rfind('/');
        if (pos == std::string::npos) break;
        dir = dir.substr(0, pos);
    }
    return {};
#endif
}

// ── 模型扫描 ─────────────────────────────────────────────────

std::vector<ModelEntry> EdgeApp::scan_models_dir(const std::string & dir) {
    std::vector<ModelEntry> results;
#ifdef _WIN32
    std::string pattern = dir + "\\*.gguf";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return results;
    do {
        ModelEntry e;
        e.filename = fd.cFileName;
        e.path     = dir + "\\" + fd.cFileName;
        e.file_size_bytes = ((size_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
        results.push_back(e);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR * d = opendir(dir.c_str());
    if (!d) return results;
    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name(ent->d_name);
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
        ModelEntry e;
        e.filename = name;
        e.path     = dir + "/" + name;
        struct stat st;
        if (stat(e.path.c_str(), &st) == 0) e.file_size_bytes = (size_t)st.st_size;
        results.push_back(e);
    }
    closedir(d);
#endif
    return results;
}

// ── 电池信息 ─────────────────────────────────────────────────

PowerInfo EdgeApp::get_power_info() {
    PowerInfo info;
#ifdef _WIN32
    SYSTEM_POWER_STATUS sps;
    if (GetSystemPowerStatus(&sps)) {
        info.valid = true;
        if (sps.BatteryLifePercent != 255) info.battery_percent = sps.BatteryLifePercent;
        else info.valid = false;
        info.on_ac = (sps.ACLineStatus == 1);
        if (sps.BatteryLifeTime != (DWORD)-1) info.seconds_left = (int)sps.BatteryLifeTime;
    }
#endif
    return info;
}

// ── 主渲染 ───────────────────────────────────────────────────

void EdgeApp::render() {
    ImGuiViewport * vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);

    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    ImGui::Begin("EdgeGUI", nullptr, window_flags);
    ImGui::PopStyleVar(3);

    render_top_bar();

    float bottom_h = INPUT_AREA_H;
    ImGui::BeginChild("MainArea", ImVec2(0, -bottom_h), ImGuiChildFlags_None);
    render_chat_area();
    ImGui::EndChild();

    ImGui::Separator();
    render_input_area();
    ImGui::End();

    if (_show_settings)      render_settings_panel();
    if (_show_cloud_cfg)     render_cloud_panel();
    if (_show_model_browser) render_model_browser_panel();
    if (_show_heartbeat)     render_heartbeat_panel();
    if (_show_peers)         render_peers_panel();
    if (_show_conflicts)     render_conflicts_panel();
    if (_show_tasks)         render_tasks_panel();
    if (!_model_error.empty()) render_error_popup();

    // Periodic updates
    update_device_metrics();
    _task_scheduler.process_queue();

    process_incoming_tokens();

    double now = ImGui::GetTime();
    if (now - _power_last_check > 30.0) {
        _power_info = get_power_info();
        _power_last_check = now;
    }
}

// ── 顶部栏 ───────────────────────────────────────────────────

void EdgeApp::render_top_bar() {
    ImGui::BeginChild("TopBar", ImVec2(0, TOP_BAR_H), ImGuiChildFlags_AutoResizeY);

    // ── 左侧：模型状态 ─────────────────────────────────────
    if (_model_loading) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "[\xe5\x8a\xa0\xe8\xbd\xbd\xe4\xb8\xad %s]",
                           _model_load_stage.c_str());
    } else if (_model_loaded) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "[\xe6\xa8\xa1\xe5\x9e\x8b]");
        ImGui::SameLine();
        auto pos = _engine.model_path().find_last_of("/\\");
        std::string name = pos != std::string::npos
            ? _engine.model_path().substr(pos + 1) : _engine.model_path();
        ImGui::Text("%s", name.c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("| %.0f MB", _engine.model_size() / 1024.0f / 1024.0f);
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "[\xe6\x9c\xaa\xe5\x8a\xa0\xe8\xbd\xbd]");
    }

    // ── 中间：性能指标 ─────────────────────────────────────
    if (_last_perf.valid) {
        ImGui::SameLine(ImGui::GetWindowWidth() * 0.38f);
        ImGui::TextDisabled("\xe9\xa2\x84%e5\xa4\x84%e7\x90\x86: %.1f t/s | \xe7\x94\x9f%e6\x88\x90: %.1f t/s | \xe9\xa6\x96token: %.0f ms",
                            _last_perf.pp_tokens_per_sec,
                            _last_perf.tg_tokens_per_sec,
                            _last_perf.ttft_ms);
    }

    // ── 右侧：状态图标 ─────────────────────────────────────
    float right_x = ImGui::GetWindowWidth() - 280;
    ImGui::SameLine(right_x);

    if (_cloud_enabled && _cloud.is_reachable()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "[\xe4\xba\x91\xe7\xab\xaf]");
    } else if (_cloud_enabled) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "[\xe6\x9c\xac\xe5\x9c\xb0]");
    }

    // ── Network condition indicator (edge autonomy) ────────────
    {
        ImGui::SameLine();
        auto nc = _local_decision.current_condition();
        switch (nc) {
            case NetworkCondition::Healthy:
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "[net:ok]");
                break;
            case NetworkCondition::Degraded:
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "[net:degraded]");
                break;
            case NetworkCondition::Offline:
                ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "[net:offline]");
                break;
        }
    }

    if (_power_info.valid) {
        ImGui::SameLine();
        if (_power_info.on_ac) {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "[\xe7\x94\xb5\xe6\xba\x90]");
        } else {
            ImVec4 c = _power_info.battery_percent > 20
                ? ImVec4(0.5f, 1.0f, 0.5f, 1.0f)
                : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
            ImGui::TextColored(c, "[%d%%]", _power_info.battery_percent);
        }
    }

    if (_model_loaded) {
        ImGui::SameLine();
        int used = _engine.token_count();
        int ctx  = _engine.n_ctx;
        ImGui::TextDisabled("[%d/%d]", used, ctx);
    }

    // ── 按钮 ────────────────────────────────────────────────
    ImGui::SameLine();

    if (ImGui::Button("\xe6\x89\x93\xe5\xbc\x80")) {        // 打开
        std::string path = file_dialog_open("\xe9\x80\x89\xe6\x8b\xa9 GGUF \xe6\xa8\xa1\xe5\x9e\x8b", "*.gguf");
        if (!path.empty()) {
            _model_loading = true;
            _model_load_prog = 0.0f;
            _model_load_stage = "\xe5\x90\xaf\xe5\x8a\xa8\xe4\xb8\xad...";
            // Join previous load thread if still running
        if (_model_load_thread.joinable()) _model_load_thread.join();
        _model_load_thread = std::thread([this, path]() {
                auto err = load_model(path);
                _model_loading = false;
                if (!err.empty()) _model_error = err;
            });
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("\xe6\xa8\xa1\xe5\x9e\x8b")) {       // 模型
        _show_model_browser = !_show_model_browser;
        if (_show_model_browser && !_models_scanned) {
            _available_models = scan_models_dir(_models_dir);
            _models_scanned = true;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("\xe8\xae\xbe\xe7\xbd\xae"))          // 设置
        _show_settings = !_show_settings;
    ImGui::SameLine();
    if (ImGui::Button("\xe4\xba\x91\xe7\xab\xaf"))          // 云端
        _show_cloud_cfg = !_show_cloud_cfg;
    ImGui::SameLine();
    if (ImGui::Button("\xe5\xbf\x83\xe8\xb7\xb3"))          // 心跳
        _show_heartbeat = !_show_heartbeat;
    ImGui::SameLine();
    if (ImGui::Button("P2P"))                                // P2P
        _show_peers = !_show_peers;
    ImGui::SameLine();
    if (ImGui::Button("\xe5\x86\xb2\xe7\xaa\x81"))          // 冲突
        _show_conflicts = !_show_conflicts;
    ImGui::SameLine();
    if (_model_loaded) {
        if (ImGui::Button("\xe4\xbf\x9d\xe5\xad\x98"))      // 保存
            save_current_session();
        ImGui::SameLine();
        if (ImGui::Button("\xe5\x8a\xa0\xe8\xbd\xbd"))      // 加载
            load_session_dialog();
    }

    ImGui::EndChild();
    ImGui::Separator();
}

// ── 对话区域 ─────────────────────────────────────────────────

void EdgeApp::render_chat_area() {
    ImGui::BeginChild("ChatScroll", ImVec2(0, 0), ImGuiChildFlags_None,
                      ImGuiWindowFlags_AlwaysVerticalScrollbar);

    for (size_t i = 0; i < _messages.size(); ++i) {
        const auto & msg = _messages[i];
        bool is_user = (msg.role == "user");

        ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x - 20);

        ImVec4 role_color = is_user
            ? ImVec4(0.5f, 0.8f, 1.0f, 1.0f)
            : ImVec4(0.3f, 1.0f, 0.5f, 1.0f);

        std::string header = is_user ? "\xe4\xbd\xa0:" : "\xe5\x8a\xa9\xe6\x89\x8b:";
        if (msg.role == "system") {
            header = "\xe7\xb3\xbb\xe7\xbb\x9f:";
            role_color = ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
        }
        if (msg.role == "tool") {
            header = "\xe5\xb7\xa5\xe5\x85\xb7:";
            role_color = ImVec4(0.7f, 0.7f, 1.0f, 1.0f);
        }

        ImGui::TextColored(role_color, "%s", header.c_str());
        ImGui::SameLine();
        ImGui::TextWrapped("%s", msg.content.c_str());

        if (!msg.reasoning_content.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
            ImGui::TextWrapped("  [\xe6\x80\x9d%e8\x80\x83] %s", msg.reasoning_content.c_str());
            ImGui::PopStyleColor();
        }

        ImGui::PopTextWrapPos();
        ImGui::Spacing();
    }

    // 流式输出
    if (!_generation_finished) {
        ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x - 20);
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "\xe5\x8a\xa9\xe6\x89\x8b:");
        ImGui::SameLine();
        {
            std::lock_guard<std::mutex> lock(_streaming_mutex);
            ImGui::TextWrapped("%s", _streaming_text.c_str());
        }
        ImGui::PopTextWrapPos();
        ImGui::SameLine();
        ImGui::TextDisabled("|");
    }

    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 10) {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
}

// ── 输入区域 ─────────────────────────────────────────────────

void EdgeApp::render_input_area() {
    ImGui::BeginChild("InputArea", ImVec2(0, INPUT_AREA_H), ImGuiChildFlags_None);
    bool generating = !_generation_finished;

    if (generating) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button(" \xe5\x81\x9c\xe6\xad\xa2 ", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            _engine.stop();
            _cloud.stop();
        }
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        ImGui::TextDisabled(" \xe7\x94\x9f%e6\x88\x90\xe4\xb8\xad...");
    } else {
        float btn_w = 70.0f;
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - btn_w - 10);
        bool enter_pressed = ImGui::InputText("##Input", _input_buf, sizeof(_input_buf),
                                              ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::PopItemWidth();
        ImGui::SameLine();

        bool can_send = strlen(_input_buf) > 0 &&
                        (_model_loaded || (_cloud_enabled && _cloud.is_reachable()));
        if (!can_send) ImGui::BeginDisabled();
        bool send_clicked = ImGui::Button("\xe5\x8f\x91\xe9\x80\x81", ImVec2(btn_w, 0));
        if (!can_send) ImGui::EndDisabled();

        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere(-1);

        if ((enter_pressed || send_clicked) && can_send) {
            std::string text(_input_buf);
            memset(_input_buf, 0, sizeof(_input_buf));
            submit_message(text);
            ImGui::SetKeyboardFocusHere(-1);
        }
    }
    ImGui::EndChild();
}

// ── 设置面板 ─────────────────────────────────────────────────

void EdgeApp::render_settings_panel() {
    ImGui::SetNextWindowSize(ImVec2(380, 520), ImGuiCond_FirstUseEver);
    ImGui::Begin("\xe8\xae\xbe\xe7\xbd\xae", &_show_settings);

    // ── 生成参数 ──────────────────────────────────────────
    if (ImGui::CollapsingHeader("\xe7\x94\x9f%e6\x88\x90\xe5\x8f\x82\xe6\x95\xb0", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("\xe6\x9c\x80\xe5\xa4\xa7 Token", &_engine.n_predict, 64, 4096);
        ImGui::SliderInt("\xe4\xb8\x8a\xe4\xb8\x8b\xe6\x96\x87\xe9\x95\xbf\xe5\xba\xa6", &_engine.n_ctx, 256, 32768);
        ImGui::SliderInt("\xe6\x89\xb9\xe6\xac\xa1\xe5\xa4\xa7\xe5\xb0\x8f", &_engine.n_batch, 64, 2048);
        ImGui::SliderInt("\xe4\xbf\x9d\xe7\x95\x99 Token", &_engine.n_keep, 1, 100);
        ImGui::Separator();
        ImGui::SliderFloat("\xe6\xb8\xa9\xe5\xba\xa6", &_engine.temperature, 0.0f, 2.0f);
        ImGui::SliderFloat("Top-P", &_engine.top_p, 0.0f, 1.0f);
        ImGui::SliderFloat("Min-P", &_engine.min_p, 0.0f, 0.5f);
        ImGui::SliderInt("Top-K", &_engine.top_k, 1, 200);
        ImGui::Separator();
        ImGui::SliderFloat("\xe9\x87\x8d\xe5\xa4\x8d\xe6\x83\xa9\xe7\xbd\x9a", &_engine.repeat_penalty, 1.0f, 2.0f);
        ImGui::SliderFloat("DRY \xe5\xbc\xba\xe5\xba\xa6", &_engine.dry_multiplier, 0.0f, 1.0f);
        ImGui::SliderFloat("XTC \xe6\xa6\x82\xe7\x8e\x87", &_engine.xtc_probability, 0.0f, 1.0f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("0.0 = \xe7\xa6\x81\xe7\x94\xa8\xef\xbc\x8c\xe8\xb6\x8a\xe9\xab\x98\xe8\xb6\x8a\xe9\x9a\x8f\xe6\x9c\xba");
    }

    // ── 模型目录 ──────────────────────────────────────────
    if (ImGui::CollapsingHeader("\xe6\xa8\xa1\xe5\x9e\x8b\xe7\x9b\xae\xe5\xbd\x95", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputText("\xe8\xb7\xaf\xe5\xbe\x84", _models_dir_buf, sizeof(_models_dir_buf));
        ImGui::SameLine();
        if (ImGui::Button("\xe6\xb5\x8f\xe8\xa7\x88##ModelsDir")) {
            std::string picked = file_dialog_open("\xe9\x80\x89\xe6\x8b\xa9\xe6\xa8\xa1\xe5\x9e\x8b\xe7\x9b\xae\xe5\xbd\x95", "");
            if (!picked.empty())
                strncpy(_models_dir_buf, picked.c_str(), sizeof(_models_dir_buf) - 1);
        }
        if (ImGui::Button("\xe5\xba\x94\xe7\x94\xa8\xe5\xb9\xb6\xe5\x88\xb7\xe6\x96\xb0")) {
            _models_dir = _models_dir_buf;
            _available_models = scan_models_dir(_models_dir);
            _models_scanned = true;
        }
        ImGui::SameLine();
        ImGui::Text("(%zu \xe4\xb8\xaa\xe6\xa8\xa1\xe5\x9e\x8b)", _available_models.size());
    }

    // ── 内存预算 ──────────────────────────────────────────
    if (ImGui::CollapsingHeader("\xe5\x86\x85\xe5\xad\x98\xe9\xa2\x84\xe7\xae\x97")) {
        ImGui::SliderInt("\xe5\x8f\xaf\xe7\x94\xa8\xe5\x86\x85\xe5\xad\x98 (MB)", &_avail_mem_mb, 0, 32768);
        if (_avail_mem_mb == 0) {
            ImGui::TextDisabled("\xe6\x9c\xaa\xe8\xae\xbe\xe7\xbd\xae\xe9\xa2\x84\xe7\xae\x97\xef\xbc\x8c\xe6\x89\x80\xe6\x9c\x89\xe6\xa8\xa1\xe5\x9e\x8b\xe5\x9d\x87\xe6\x98\xbe\xe7\xa4\xba\xe4\xb8\xba\xe5\x85\xbc\xe5\xae\xb9");
        } else {
            ImGui::Text("\xe9\xa2\x84\xe7\xae\x97: %.1f GB", _avail_mem_mb / 1024.0f);
        }
    }

    // ── 系统提示词 ────────────────────────────────────────
    if (ImGui::CollapsingHeader("\xe7\xb3\xbb\xe7\xbb\x9f\xe6\x8f\x90\xe7\xa4\xba\xe8\xaf\x8d")) {
        static char sys_buf[2048] = {};
        ImGui::InputTextMultiline("##SysPrompt", sys_buf, sizeof(sys_buf), ImVec2(-1, 80));
        if (ImGui::Button("\xe5\xba\x94\xe7\x94\xa8")) _system_prompt = sys_buf;
        ImGui::SameLine();
        if (ImGui::Button("\xe6\xb8\x85\xe7\xa9\xba")) { memset(sys_buf, 0, sizeof(sys_buf)); _system_prompt.clear(); }
    }

    // ── 模型信息 ──────────────────────────────────────────
    if (ImGui::CollapsingHeader("\xe6\xa8\xa1\xe5\x9e\x8b\xe4\xbf\xa1\xe6\x81\xaf")) {
        if (_model_loaded) {
            ImGui::Text("\xe8\xb7\xaf\xe5\xbe\x84:   %s", _engine.model_path().c_str());
            ImGui::Text("\xe5\xa4\xa7\xe5\xb0\x8f:   %.1f MB", _engine.model_size() / 1024.0f / 1024.0f);
            ImGui::Text("\xe5\x8f\x82\xe6\x95\xb0: %.2f B", _engine.model_params() / 1e9f);
            ImGui::Text("\xe6\xa8\xa1\xe6\x9d\xbf: %s",
                        _engine.has_chat_template() ? "\xe5\x8f\xaf\xe7\x94\xa8" : "\xe6\x97\xa0 (ChatML \xe5\x9b\x9e\xe9\x80\x80)");
        } else {
            ImGui::TextDisabled("\xe6\x9c\xaa\xe5\x8a\xa0\xe8\xbd\xbd\xe6\xa8\xa1\xe5\x9e\x8b");
        }
    }

    // ── 上次推理 ──────────────────────────────────────────
    if (ImGui::CollapsingHeader("\xe4\xb8\x8a\xe6\xac\xa1\xe6\x8e\xa8\xe7\x90\x86")) {
        if (_last_perf.valid) {
            ImGui::Text("Prompt Token:     %d", _last_perf.n_p_tokens);
            ImGui::Text("\xe7\x94\x9f%e6\x88\x90 Token:     %d", _last_perf.n_g_tokens);
            ImGui::Text("\xe9\xa2\x84\xe5\xa4\x84\xe7\x90\x86\xe9\x80\x9f\xe5\xba\xa6:     %.1f t/s", _last_perf.pp_tokens_per_sec);
            ImGui::Text("\xe7\x94\x9f%e6\x88\x90\xe9\x80\x9f\xe5\xba\xa6:     %.1f t/s", _last_perf.tg_tokens_per_sec);
            ImGui::Text("\xe9\xa6\x96 Token \xe5\xbb\xb6\xe8\xbf\x9f: %.0f ms", _last_perf.ttft_ms);
        } else {
            ImGui::TextDisabled("\xe6\x9a\x82\xe6\x97\xa0\xe6\x8e\xa8\xe7\x90\x86\xe6\x95\xb0\xe6\x8d\xae");
        }
    }

    ImGui::Separator();
    if (ImGui::Button("\xe6\xb8\x85\xe9\x99\xa4\xe5\xaf\xb9\xe8\xaf\x9d")) {
        _messages.clear();
        _engine.clear_kv_cache();
        if (!_system_prompt.empty()) _messages.push_back({"system", _system_prompt});
    }
    ImGui::End();
}

// ── 云端面板 ─────────────────────────────────────────────────

void EdgeApp::render_cloud_panel() {
    ImGui::SetNextWindowSize(ImVec2(400, 250), ImGuiCond_FirstUseEver);
    ImGui::Begin("\xe4\xba\x91\xe7\xab\xaf\xe9\x85\x8d\xe7\xbd\xae", &_show_cloud_cfg);

    ImGui::Checkbox("\xe5\x90\xaf\xe7\x94\xa8\xe4\xba\x91\xe7\xab\xaf\xe6\x8e\xa8\xe7\x90\x86", &_cloud_enabled);
    ImGui::Separator();
    ImGui::InputText("\xe6\x8e\xa5\xe5\x8f\xa3\xe5\x9c\xb0\xe5\x9d\x80", _cloud_endpoint, sizeof(_cloud_endpoint));
    ImGui::InputText("API \xe5\xaf\x86\xe9\x92\xa5", _cloud_api_key, sizeof(_cloud_api_key), ImGuiInputTextFlags_Password);
    ImGui::InputText("\xe6\xa8\xa1\xe5\x9e\x8b\xe5\x90\x8d\xe7\xa7\xb0", _cloud_model, sizeof(_cloud_model));
    ImGui::Separator();

    if (ImGui::Button("\xe6\xb5\x8b\xe8\xaf\x95\xe8\xbf\x9e\xe6\x8e\xa5")) {
        _cloud.set_endpoint(_cloud_endpoint, _cloud_api_key);
        if (strlen(_cloud_model) > 0) _cloud.set_model_name(_cloud_model);
        _cloud.check_reachable();
    }
    ImGui::SameLine();
    if (_cloud.is_reachable()) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "\xe5\x8f\xaf\xe8\xbe\xbe");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "\xe4\xb8\x8d\xe5\x8f\xaf\xe8\xbe\xbe");
    }
    ImGui::End();
}

// ── 模型浏览器 ───────────────────────────────────────────────

void EdgeApp::render_model_browser_panel() {
    ImGui::SetNextWindowSize(ImVec2(500, 350), ImGuiCond_FirstUseEver);
    ImGui::Begin("\xe5\x8f\xaf\xe7\x94\xa8\xe6\xa8\xa1\xe5\x9e\x8b", &_show_model_browser);

    if (ImGui::Button("\xe5\x88\xb7\xe6\x96\xb0")) {
        _available_models = scan_models_dir(_models_dir);
        _models_scanned = true;
    }
    ImGui::SameLine();
    ImGui::Text("\xe7\x9b\xae\xe5\xbd\x95: %s (%zu \xe4\xb8\xaa\xe6\xa8\xa1\xe5\x9e\x8b)", _models_dir.c_str(), _available_models.size());
    ImGui::Separator();

    if (_available_models.empty()) {
        ImGui::TextDisabled("\xe5\x9c\xa8 %s \xe4\xb8\xad\xe6\x9c\xaa\xe6\x89\xbe\xe5\x88\xb0 GGUF \xe6\x96\x87\xe4\xbb\xb6", _models_dir.c_str());
        ImGui::TextDisabled("\xe8\xaf\xb7\xe5\xb0\x86 .gguf \xe6\xa8\xa1\xe5\x9e\x8b\xe6\x96\x87\xe4\xbb\xb6\xe6\x94\xbe\xe5\x85\xa5 models/ \xe7\x9b\xae\xe5\xbd\x95");
    } else {
        ImGui::Columns(4);
        ImGui::Text("\xe6\xa8\xa1\xe5\x9e\x8b"); ImGui::NextColumn();
        ImGui::Text("\xe5\xa4\xa7\xe5\xb0\x8f"); ImGui::NextColumn();
        ImGui::Text("\xe9\x80\x82\xe9\x85\x8d"); ImGui::NextColumn();
        ImGui::Text("");                ImGui::NextColumn();
        ImGui::Separator();

        for (auto & entry : _available_models) {
            MemFit fit = check_model_fit(entry);
            entry.compatible = (fit != MemFit::Over);

            ImGui::Text("%s", entry.filename.c_str()); ImGui::NextColumn();
            float size_mb = entry.file_size_bytes / 1024.0f / 1024.0f;
            ImGui::Text("%.0f MB", size_mb);            ImGui::NextColumn();

            switch (fit) {
                case MemFit::Good:
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "\xe5\x85\x85\xe8\xa3\x95"); break;
                case MemFit::Tight:
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "\xe4\xb8\xb4\xe7\x95\x8c"); break;
                case MemFit::Over:
                    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "\xe8\xb6\x85\xe9\x99\x90"); break;
            }
            ImGui::NextColumn();

            ImGui::PushID(entry.filename.c_str());
            bool disabled = (fit == MemFit::Over && _avail_mem_mb > 0);
            if (disabled) ImGui::BeginDisabled();
            if (ImGui::Button("\xe5\x8a\xa0\xe8\xbd\xbd")) {
                _engine.stop(); _cloud.stop();
                _model_loading = true;
                _model_load_prog = 0.0f;
                _model_load_stage = "\xe5\x90\xaf\xe5\x8a\xa8\xe4\xb8\xad...";
                std::string p = entry.path;
                if (_model_load_thread.joinable()) _model_load_thread.join();
                _model_load_thread = std::thread([this, p]() {
                    auto err = load_model(p);
                    _model_loading = false;
                    if (!err.empty()) _model_error = err;
                });
                _show_model_browser = false;
            }
            if (disabled) {
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("\xe6\xa8\xa1\xe5\x9e\x8b\xe8\xb6\x85\xe5\x87\xba\xe5\x86\x85\xe5\xad\x98\xe9\xa2\x84\xe7\xae\x97");
            }
            ImGui::PopID();
            ImGui::NextColumn();
        }
        ImGui::Columns(1);
    }
    ImGui::End();
}

// ── 错误弹窗 ─────────────────────────────────────────────────

void EdgeApp::render_error_popup() {
    ImGui::OpenPopup("\xe9\x94\x99\xe8\xaf\xaf");
    if (ImGui::BeginPopupModal("\xe9\x94\x99\xe8\xaf\xaf", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s", _model_error.c_str());
        ImGui::Separator();
        if (ImGui::Button("\xe7\xa1\xae\xe5\xae\x9a", ImVec2(120, 0))) {
            _model_error.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

// ── 消息提交 ─────────────────────────────────────────────────

void EdgeApp::submit_message(const std::string & text) {
    _messages.push_back({"user", text});
    _messages.push_back({"assistant", ""});

    _generation_finished = false;
    {
        std::lock_guard<std::mutex> lock(_streaming_mutex);
        _streaming_text.clear();
        _pending_tokens.clear();
    }

    auto on_token = [this](const std::string & token_text, bool finished) {
        std::lock_guard<std::mutex> lock(_streaming_mutex);
        if (finished) _pending_tokens.push_back("__FINISHED__");
        else          _pending_tokens.push_back(token_text);
    };

    auto on_perf = [this](const EnginePerf & perf) { _last_perf = perf; };

    // ── Network-aware autonomous decision (edge-local fallback) ──
    // Assess current network condition for edge autonomy (Module 3.1).
    int  rtt_ms   = _device_monitor.measure_rtt_ms(_cloud_endpoint);
    bool cloud_ok = _cloud_client.is_reachable();
    int  hb_fails = _heartbeat.consecutive_failures();
    NetworkCondition net = _local_decision.assess_network(rtt_ms, cloud_ok, hb_fails);

    bool force_local = false;
    bool use_p2p     = false;

    if (net != NetworkCondition::Healthy) {
        // Use a default confidence of 0.7 — wire to real inference
        // confidence once the engine exposes per-token logprobs.
        float confidence = 0.7f;
        LocalAction action = _local_decision.decide(confidence, net);

        switch (action) {
            case LocalAction::LocalInfer:
                force_local = true;
                break;
            case LocalAction::QueueForCloud:
                // Respond locally now; queue for later cloud review.
                force_local = true;
                // TODO: enqueue the message text for deferred cloud processing
                break;
            case LocalAction::P2PConsensus:
                if (_p2p.is_running() && !_p2p.known_peers().empty()) {
                    use_p2p     = true;
                    force_local = true;
                } else {
                    // No P2P peers available — best-effort local inference.
                    force_local = true;
                }
                break;
            case LocalAction::FollowCloud:
                // Network is degraded but decision engine says follow cloud.
                // Fall through to normal cloud/local path below.
                break;
        }
    }

    if (force_local) {
        // ── Autonomous local inference ──────────────────────────
        std::string prompt;
        std::vector<common_chat_msg> msgs(_messages.begin(), _messages.end() - 1);
        if (!_system_prompt.empty()) msgs.insert(msgs.begin(), {"system", _system_prompt});
        prompt = _engine.render_chat_prompt(msgs, true);
        _engine.generate(prompt, on_token, on_perf);

        // Broadcast decision intent to P2P peers for consensus
        if (use_p2p) {
            for (const auto & peer : _p2p.known_peers()) {
                if (peer.online) {
                    DecisionIntent intent;
                    intent.intent_id  = "intent_" + std::to_string(_messages.size());
                    intent.node_id    = _p2p_node_id;
                    intent.target_id  = "msg_" + std::to_string(_messages.size());
                    intent.proposed_action = "local_infer";
                    intent.confidence = 0.7f;
                    _p2p.send_decision_intent(peer.node_id, intent);
                }
            }
        }
    } else if (_cloud_enabled && _cloud.is_reachable()) {
        std::string json = build_messages_json(
            std::deque<common_chat_msg>(_messages.begin(), _messages.end() - 1));
        _cloud.set_endpoint(_cloud_endpoint, _cloud_api_key);
        if (strlen(_cloud_model) > 0) _cloud.set_model_name(_cloud_model);
        _cloud.send(json, on_token);
    } else {
        std::string prompt;
        std::vector<common_chat_msg> msgs(_messages.begin(), _messages.end() - 1);
        if (!_system_prompt.empty()) msgs.insert(msgs.begin(), {"system", _system_prompt});
        prompt = _engine.render_chat_prompt(msgs, true);
        _engine.generate(prompt, on_token, on_perf);
    }
}

// ── 会话保存/加载 ───────────────────────────────────────────

void EdgeApp::save_current_session() {
    std::string path = file_dialog_save("\xe4\xbf\x9d\xe5\xad\x98\xe4\xbc\x9a\xe8\xaf\x9d", "session.json");
    if (path.empty()) return;

    std::string json = "[\n";
    for (size_t i = 0; i < _messages.size(); ++i) {
        if (i > 0) json += ",\n";
        json += "  " + _messages[i].to_json_oaicompat(false).dump();
    }
    json += "\n]\n";

    std::ofstream fjson(path);
    if (!fjson) { _model_error = "\xe4\xbf\x9d\xe5\xad\x98\xe5\xa4\xb1\xe8\xb4\xa5: " + path; return; }
    fjson << json;
    fjson.close();

    std::string state_path = path + ".ggsn";
}

void EdgeApp::load_session_dialog() {
    std::string path = file_dialog_open("\xe5\x8a\xa0\xe8\xbd\xbd\xe4\xbc\x9a\xe8\xaf\x9d", "*.json");
    if (path.empty()) return;

    std::ifstream fjson(path);
    if (!fjson) { _model_error = "\xe5\x8a\xa0\xe8\xbd\xbd\xe5\xa4\xb1\xe8\xb4\xa5: " + path; return; }
    std::string json((std::istreambuf_iterator<char>(fjson)), std::istreambuf_iterator<char>());
    fjson.close();
    _messages.clear();

    std::string state_path = path + ".ggsn";
    std::vector<llama_token> tokens;
    if (!_engine.load_session(state_path, tokens, 32768)) _engine.clear_kv_cache();
}

// ── Token 处理 ───────────────────────────────────────────────

void EdgeApp::process_incoming_tokens() {
    std::lock_guard<std::mutex> lock(_streaming_mutex);
    for (const auto & tok : _pending_tokens) {
        if (tok == "__FINISHED__") {
            if (!_messages.empty()) _messages.back().content = _streaming_text;
            _streaming_text.clear();
            _generation_finished = true;
        } else {
            _streaming_text += tok;
        }
    }
    _pending_tokens.clear();
}

// ── Device metrics update (called each frame, throttled) ──────────

void EdgeApp::update_device_metrics() {
    double now = ImGui::GetTime();
    if (now - _last_metrics_update < 1.0) return;  // 1 Hz
    _last_metrics_update = now;

    // Feed inference perf to heartbeat
    if (_last_perf.valid) {
        _heartbeat.update_perf(_last_perf.tg_tokens_per_sec,
                               _engine.is_running() ? 1 : 0);
    }
    // Feed network status
    _heartbeat.set_network_rtt(
        _cloud_client.is_reachable() ? 0 : -1,
        _cloud_client.is_reachable());

    // ── Edge autonomy: track network transitions ───────────────
    // Re-assess network condition at 1 Hz so the UI stays current.
    NetworkCondition prev_cond = _local_decision.current_condition();
    int  rtt    = _device_monitor.measure_rtt_ms(_cloud_endpoint);
    bool cloud  = _cloud_client.is_reachable();
    int  hb_fail = _heartbeat.consecutive_failures();
    NetworkCondition new_cond = _local_decision.assess_network(rtt, cloud, hb_fail);

    // Flush offline decisions when network recovers
    if (new_cond == NetworkCondition::Healthy &&
        (prev_cond == NetworkCondition::Degraded ||
         prev_cond == NetworkCondition::Offline)) {
        auto decisions = _local_decision.flush_offline_decisions();
        if (!decisions.empty()) {
            fprintf(stderr, "[EdgeLocalDecision] Network recovered — "
                    "flushing %zu offline decision(s) for cloud PPO training\n",
                    decisions.size());
            // TODO: upload decisions to cloud via _cloud_client for PPO replay buffer
        }
    }
}

// ── Heartbeat panel ──────────────────────────────────────────────

void EdgeApp::render_heartbeat_panel() {
    ImGui::SetNextWindowSize(ImVec2(380, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("\xe5\xbf\x83\xe8\xb7\xb3\xe7\x8a\xb6\xe6\x80\x81", &_show_heartbeat);

    bool hb_running = _heartbeat.is_running();
    if (ImGui::Checkbox("\xe5\x90\xaf\xe7\x94\xa8\xe5\xbf\x83\xe8\xb7\xb3", &hb_running)) {
        if (hb_running) {
            HeartbeatConfig cfg;
            cfg.cloud_endpoint = _heartbeat_endpoint;
            cfg.device_id      = _p2p_node_id;
            cfg.device_type    = "edge-gui-win";
            _heartbeat.configure(cfg);
            _heartbeat.start();
        } else {
            _heartbeat.stop();
        }
    }

    ImGui::InputText("\xe4\xb8\x8a\xe6\x8a\xa5\xe5\x9c\xb0\xe5\x9d\x80",
                     _heartbeat_endpoint, sizeof(_heartbeat_endpoint));

    ImGui::Text("\xe4\xb8\x8a\xe6\xac\xa1\xe4\xb8\x8a\xe6\x8a\xa5: %s",
                _heartbeat.last_success() ? "\xe6\x88\x90\xe5\x8a\x9f" : "\xe5\xa4\xb1\xe8\xb4\xa5");
    ImGui::Text("\xe5\xbb\xb6\xe8\xbf\x9f: %d ms", _heartbeat.last_latency());

    if (ImGui::Button("\xe7\xab\x8b\xe5\x8d\xb3\xe4\xb8\x8a\xe6\x8a\xa5")) {
        _heartbeat.send_now();
    }

    ImGui::End();
}

// ── Peers (P2P) panel ────────────────────────────────────────────

void EdgeApp::render_peers_panel() {
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("P2P \xe8\x8a\x82\xe7\x82\xb9", &_show_peers);

    ImGui::InputText("\xe8\x8a\x82\xe7\x82\xb9 ID", _p2p_node_id, sizeof(_p2p_node_id));
    ImGui::InputInt("UDP \xe7\xab\xaf\xe5\x8f\xa3", &_p2p_udp_port);
    ImGui::InputInt("TCP \xe7\xab\xaf\xe5\x8f\xa3", &_p2p_tcp_port);

    bool p2p_running = _p2p.is_running();
    if (ImGui::Checkbox("\xe5\x90\xaf\xe7\x94\xa8 P2P", &p2p_running)) {
        if (p2p_running) {
            _p2p.set_node_id(_p2p_node_id);
            _p2p.set_listen_ports(_p2p_udp_port, _p2p_tcp_port);
            _conflict_detector.set_node_id(_p2p_node_id);
            _p2p.start();
        } else {
            _p2p.stop();
        }
    }

    if (ImGui::Button("\xe5\x8f\x91\xe7\x8e\xb0\xe8\x8a\x82\xe7\x82\xb9")) {
        _p2p.discover_peers();
    }

    ImGui::Separator();
    ImGui::Text("\xe5\xb7\xb2\xe7\x9f\xa5\xe8\x8a\x82\xe7\x82\xb9:");

    auto peers = _p2p.known_peers();
    if (peers.empty()) {
        ImGui::TextDisabled("  (\xe6\x97\xa0)");
    } else {
        for (const auto & p : peers) {
            ImGui::Text("  %s @ %s:%d [%s]",
                        p.node_id.c_str(), p.ip_address.c_str(),
                        p.tcp_port, p.online ? "\xe5\x9c\xa8\xe7\xba\xbf" : "\xe7\xa6\xbb\xe7\xba\xbf");
        }
    }

    ImGui::End();
}

// ── Conflicts panel ──────────────────────────────────────────────

void EdgeApp::render_conflicts_panel() {
    ImGui::SetNextWindowSize(ImVec2(500, 350), ImGuiCond_FirstUseEver);
    ImGui::Begin("\xe5\x86\xb2\xe7\xaa\x81\xe7\xae\xa1\xe7\x90\x86", &_show_conflicts);

    auto s = _conflict_detector.stats();
    ImGui::Text("\xe5\x86\xb3\xe7\xad\x96\xe6\x80\xbb\xe6\x95\xb0: %d", s.total_decisions);
    ImGui::Text("\xe6\xa3\x80\xe6\xb5\x8b\xe5\x86\xb2\xe7\xaa\x81: %d", s.conflicts_detected);
    ImGui::Text("\xe6\x9c\xac\xe5\x9c\xb0\xe8\xa7\xa3\xe5\x86\xb3: %d", s.resolved_locally);
    ImGui::Text("\xe5\x8d\x87\xe7\xba\xa7\xe4\xba\x91\xe7\xab\xaf: %d", s.escalated_to_cloud);

    ImGui::Separator();
    ImVec4 green = ImVec4(0.3f, 1.0f, 0.3f, 1.0f);
    ImVec4 red   = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);

    ImGui::Text("\xe5\x86\xb2\xe7\xaa\x81\xe6\xaf\x94\xe4\xbe\x8b: ");
    ImGui::SameLine();
    ImGui::TextColored(s.conflict_ratio() <= 5.0 ? green : red,
                       "%.1f%% (\xe7\x9b\xae\xe6\xa0\x87 <=5%%)", s.conflict_ratio());

    ImGui::Text("\xe8\xa7\xa3\xe5\x86\xb3\xe6\x88\x90\xe5\x8a\x9f\xe7\x8e\x87: ");
    ImGui::SameLine();
    ImGui::TextColored(s.resolution_rate() >= 90.0 ? green : red,
                       "%.1f%% (\xe7\x9b\xae\xe6\xa0\x87 >=90%%)", s.resolution_rate());

    ImGui::Separator();
    auto conflicts = _conflict_detector.check_all_conflicts();
    if (!conflicts.empty()) {
        ImGui::Text("\xe5\xbd\x93\xe5\x89\x8d\xe5\x86\xb2\xe7\xaa\x81:");
        for (auto & c : conflicts) {
            ImGui::BulletText("[%s] %s: \xe6\x88\x91\xe4\xbb\xac=%s(%.0f%%) vs %s=%s(%.0f%%)",
                              c.target_id.c_str(), c.reason.c_str(),
                              c.our_decision.c_str(), c.our_confidence * 100,
                              c.peer_node_id.c_str(), c.peer_decision.c_str(),
                              c.peer_confidence * 100);
            ImGui::SameLine();
            if (_conflict_detector.resolve_locally(c)) {
                ImGui::TextColored(green, "[\xe5\xb7\xb2\xe8\xa7\xa3\xe5\x86\xb3: %s]",
                                   c.resolution_method.c_str());
            } else {
                ImGui::TextColored(red, "[\xe5\x8d\x87\xe7\xba\xa7...]");
                _conflict_detector.escalate_to_cloud(c);
            }
        }
    }

    if (ImGui::Button("\xe9\x87\x8d\xe7\xbd\xae\xe7\xbb\x9f\xe8\xae\xa1")) {
        _conflict_detector.reset_stats();
    }

    ImGui::End();
}

// ── Tasks panel ──────────────────────────────────────────────────

void EdgeApp::render_tasks_panel() {
    ImGui::SetNextWindowSize(ImVec2(350, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("\xe4\xbb\xbb\xe5\x8a\xa1\xe9\x98\x9f\xe5\x88\x97", &_show_tasks);

    ImGui::Text("\xe5\xbe\x85\xe5\xa4\x84\xe7\x90\x86: %d", _task_scheduler.pending_count());

    if (ImGui::Button("\xe6\x9f\xa5\xe8\xaf\xa2\xe4\xba\x91\xe7\xab\xaf\xe4\xbb\xbb\xe5\x8a\xa1")) {
        int n = _task_scheduler.poll_cloud_tasks();
        if (n > 0) {
            fprintf(stderr, "Received %d tasks from cloud\n", n);
        }
    }

    ImGui::End();
}

// ── Callback implementations ─────────────────────────────────────

void EdgeApp::on_peer_perception(const PerceptionReport & report) {
    // Log for debugging; conflict detector handles the actual logic
    fprintf(stderr, "P2P: received perception from %s (%zu detections)\n",
            report.node_id.c_str(), report.detections.size());
}

void EdgeApp::on_peer_intent(const DecisionIntent & intent) {
    fprintf(stderr, "P2P: received intent from %s on target %s: %s\n",
            intent.node_id.c_str(), intent.target_id.c_str(),
            intent.proposed_action.c_str());
}

void EdgeApp::on_conflict(const ConflictRecord & cr) {
    fprintf(stderr, "CONFLICT: %s [%s] our=%s vs %s=%s\n",
            cr.target_id.c_str(), cr.reason.c_str(),
            cr.our_decision.c_str(), cr.peer_node_id.c_str(),
            cr.peer_decision.c_str());
}

void EdgeApp::on_conflict_done(const ConflictRecord & cr) {
    fprintf(stderr, "CONFLICT RESOLVED: %s via %s → %s\n",
            cr.target_id.c_str(), cr.resolution_method.c_str(),
            cr.final_decision.c_str());
}
