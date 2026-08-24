#include "sensor_pipeline.h"
#include "engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>

// ═══════════════════════════════════════════════════════════════
//  Embedded constants (from edge-io-protocol/)
// ═══════════════════════════════════════════════════════════════

static const char * FAULT_DIAGNOSIS_PROMPT = R"PROMPT(你是工业边缘分类器。根据传感器数据判断设备状态，输出一个类别码(0-6)。

## 输出格式: 只输出一个数字,不加任何其他字符

## 类别
0=正常 1=轴承磨损 2=轴承严重失效 3=电流不平衡 4=过热 5=管路堵塞 6=传感器漂移

## 参考范围
- 振动 0-3.0 正常, >4.5 报警, >8.0 严重
- 温度 <75 正常, >85 报警, >100 危急
- 流量 1.5-3.0 正常, <1.0 堵塞嫌疑
- 压力变化 <10% 正常, >30% 异常

## 示例
振动=0.8 温度=48 流量=2.3 → 0
振动=0.9 温度=52 流量=2.1 → 0
振动=5.2 温度=88 流量=2.0 → 1
振动=7.8 温度=95 流量=1.8 → 2
振动=1.5 温度=58 压力=158 → 3
振动=2.1 温度=108 流量=0.4 → 4
振动=1.0 温度=55 压力=165 流量=0.3 → 5

## 当前数据
{input_data})PROMPT";

static const char * SHORT_DECISION_GRAMMAR = "root ::= [0-6]";

// ── Column name signatures for data-type auto-detection ─────────

struct ChannelDef {
    std::string              standard_name;
    std::vector<std::string> aliases;
};

static const std::map<std::string, std::vector<ChannelDef>> CHANNEL_MAPS = {
    {"hydraulic", {
        {"ps1_bar",        {"PS1","ps1","pressure_1","Pressure1"}},
        {"ps2_bar",        {"PS2","ps2","pressure_2","Pressure2"}},
        {"ps3_bar",        {"PS3","ps3","pressure_3","Pressure3"}},
        {"flow_l_min",     {"FS1","fs1","flow_1","Flow1"}},
        {"temperature_c",  {"TS1","ts1","temp_1","Temperature1"}},
        {"vibration_mm_s", {"VS1","vs1","vib_1","Vibration1"}},
    }},
    {"motor", {
        {"vibration_x_mm_s", {"x"}},
        {"vibration_y_mm_s", {"y"}},
        {"vibration_z_mm_s", {"Z"}},
        {"current_a",        {"I1","current_a"}},
        {"current_b",        {"I2","current_b"}},
        {"current_c",        {"I3","current_c"}},
        {"voltage_a",        {"V1","voltage_a"}},
        {"voltage_b",        {"V2","voltage_b"}},
        {"voltage_c",        {"V3","voltage_c"}},
    }},
    {"milling", {
        {"temperature_c",    {"Air temperature [K]","Process temperature [K]","temperature","temp"}},
        {"cutting_force_n",  {"Torque [Nm]","force","cutting_force"}},
        {"tool_wear_um",     {"Tool wear [min]","tool_wear","wear"}},
        {"speed_rpm",        {"Rotational speed [rpm]","speed","rpm"}},
        {"vibration_mm_s",   {"vibration","vib"}},
    }},
};

// ── Data-type auto-detection signatures (from edge_run.py) ──────

static const std::map<std::string, std::vector<std::string>> TYPE_SIGNATURES = {
    {"hydraulic", {"PS1","PS2","FS1","TS1","VS1","EPS1"}},
    {"motor",     {"x","y","Z","I1","I2","I3","V1","V2","V3"}},
    {"milling",   {"Air temperature [K]","Torque [Nm]","Tool wear [min]",
                   "Rotational speed [rpm]","Process temperature [K]"}},
};

// ── Category lookup tables (from edge_inference.py) ─────────────

static const CategoryInfo CATEGORIES[] = {
    {0, "正常",       "normal",   "info",     "设备运行正常, 无需处理"},
    {1, "轴承磨损",   "warning",  "warning",  "疑似轴承磨损, 建议加强监测"},
    {2, "轴承严重失效","shutdown", "critical", "轴承严重失效, 建议立即停机检修"},
    {3, "电流不平衡", "alert",    "warning",  "三相电流不平衡, 需检查电气连接"},
    {4, "过热",       "alert",    "critical", "设备过热, 需检查冷却系统"},
    {5, "管路堵塞",   "maintain", "warning",  "管路堵塞嫌疑, 建议维护"},
    {6, "传感器漂移", "warning",  "info",     "传感器疑似漂移, 建议校准"},
};

const CategoryInfo * SensorPipeline::lookup_category(int code) {
    for (const auto & c : CATEGORIES)
        if (c.code == code) return &c;
    return nullptr;
}

// ═══════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════

static std::string trim(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e-1])) --e;
    return s.substr(b, e - b);
}

static std::string to_lower(const std::string & s) {
    std::string r = s;
    std::transform(r.begin(), r.end(), r.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return r;
}

static std::vector<std::string> split_line(const std::string & line, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, delim)) {
        out.push_back(trim(item));
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════
//  open() — parse CSV/TXT sensor file
// ═══════════════════════════════════════════════════════════════

char SensorPipeline::detect_delimiter(std::ifstream & file) {
    std::string line;
    if (!std::getline(file, line)) return ',';
    // Rewind
    file.clear();
    file.seekg(0);
    if (line.find('\t') != std::string::npos) return '\t';
    if (line.find(',')  != std::string::npos) return ',';
    if (line.find(';')  != std::string::npos) return ';';
    return ',';
}

void SensorPipeline::detect_data_type(const std::string & first_line) {
    auto cols = split_line(first_line, _delimiter);
    std::vector<std::string> cols_lower;
    for (const auto & c : cols)
        cols_lower.push_back(to_lower(c));

    for (const auto & [dtype, sigs] : TYPE_SIGNATURES) {
        int hits = 0;
        for (const auto & s : sigs) {
            for (const auto & cl : cols_lower) {
                if (cl == to_lower(s)) { ++hits; break; }
            }
        }
        if (hits >= 3) {
            _data_type = dtype;
            return;
        }
    }
    _data_type = "generic";
}

void SensorPipeline::map_headers() {
    _header_map.clear();

    auto it = CHANNEL_MAPS.find(_data_type);
    if (it == CHANNEL_MAPS.end()) return;

    for (const auto & ch : it->second) {
        for (const auto & alias : ch.aliases) {
            for (const auto & h : _headers) {
                if (to_lower(h) == to_lower(alias)) {
                    _header_map[h] = ch.standard_name;
                    goto next_channel;  // found mapping for this channel
                }
            }
        }
        next_channel:;
    }
}

bool SensorPipeline::open(const std::string & filepath) {
    close();
    _filepath = filepath;

    std::ifstream file(filepath);
    if (!file.is_open()) return false;

    // 1. Detect delimiter
    _delimiter = detect_delimiter(file);

    // 2. Read header line
    std::string first_line;
    if (!std::getline(file, first_line)) { file.close(); return false; }
    _headers = split_line(first_line, _delimiter);

    // 3. Auto-detect data type
    detect_data_type(first_line);

    // 4. Map columns to standard names
    map_headers();

    // 5. Parse data rows
    _rows.clear();
    std::string line;
    while (std::getline(file, line) && _rows.size() < 10000) {
        auto fields = split_line(line, _delimiter);
        std::vector<double> row;
        bool has_numeric = false;
        for (const auto & f : fields) {
            if (f.empty()) {
                row.push_back(0.0);  // sentinel for "no value"
                continue;
            }
            try {
                size_t pos = 0;
                double v = std::stod(f, &pos);
                row.push_back(v);
                has_numeric = true;
            } catch (...) {
                row.push_back(0.0);
            }
        }
        if (has_numeric) _rows.push_back(row);
    }
    file.close();

    _opened = !_rows.empty();
    return _opened;
}

void SensorPipeline::close() {
    _headers.clear();
    _rows.clear();
    _header_map.clear();
    _batches.clear();
    _results.clear();
    _data_type = "generic";
    _opened    = false;
}

// ═══════════════════════════════════════════════════════════════
//  prepare_batches() — batch rows, compute averages, build prompts
// ═══════════════════════════════════════════════════════════════

std::string SensorPipeline::build_prompt(
    const std::map<std::string, double> & readings)
{
    // Build compact "key=val key=val ..." representation
    std::string data_text;
    for (const auto & [k, v] : readings) {
        if (!data_text.empty()) data_text += " ";
        // Trim trailing zeros for cleaner output
        char buf[64];
        snprintf(buf, sizeof(buf), "%.4g", v);
        data_text += k + "=" + buf;
    }

    std::string prompt = FAULT_DIAGNOSIS_PROMPT;
    size_t pos = prompt.find("{input_data}");
    if (pos != std::string::npos) {
        prompt.replace(pos, 12, data_text);
    } else {
        prompt += "\n" + data_text;
    }
    return prompt;
}

int SensorPipeline::prepare_batches(int batch_size) {
    _batches.clear();
    if (_rows.empty() || _header_map.empty()) return 0;

    for (size_t start = 0; start < _rows.size(); start += (size_t)batch_size) {
        size_t end = std::min(start + (size_t)batch_size, _rows.size());
        if (end - start < 5) break;  // skip tiny trailing batch

        // Collect values per standard column across batch rows
        std::map<std::string, std::vector<double>> col_values;
        for (size_t i = start; i < end; ++i) {
            for (size_t j = 0; j < _headers.size() && j < _rows[i].size(); ++j) {
                auto it = _header_map.find(_headers[j]);
                if (it == _header_map.end()) continue;
                double v = _rows[i][j];
                if (v != 0.0)  // skip sentinel
                    col_values[it->second].push_back(v);
            }
        }

        // Compute per-column average
        std::map<std::string, double> readings;
        for (const auto & [name, vals] : col_values) {
            if (vals.empty()) continue;
            double sum = 0.0;
            for (double v : vals) sum += v;
            readings[name] = std::round(sum / vals.size() * 10000.0) / 10000.0;
        }
        if (readings.empty()) continue;

        DataBatch batch;
        batch.batch_index = (int)_batches.size();
        batch.row_count   = (int)(end - start);
        batch.readings    = readings;
        batch.prompt_text = build_prompt(readings);
        _batches.push_back(batch);
    }
    return (int)_batches.size();
}

// ═══════════════════════════════════════════════════════════════
//  classify_batch() — one batch through the model
// ═══════════════════════════════════════════════════════════════

ClassificationResult SensorPipeline::classify_batch(const DataBatch & batch) {
    auto t0 = std::chrono::steady_clock::now();

    // Call engine with grammar-constrained completion
    std::string raw_output;
    bool ok = _engine->generate_completion(
        batch.prompt_text,
        SHORT_DECISION_GRAMMAR,
        [&raw_output](const std::string & token, bool finished) {
            if (!finished) raw_output += token;
        },
        nullptr  // no per-perf callback needed per batch
    );

    auto t1 = std::chrono::steady_clock::now();
    double lat = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        ClassificationResult r;
        r.batch_index = batch.batch_index;
        r.success     = false;
        r.error       = "Engine call failed (model not loaded or busy)";
        r.latency_ms  = lat;
        return r;
    }

    return parse_model_output(raw_output, batch.batch_index, lat);
}

// ═══════════════════════════════════════════════════════════════
//  parse_model_output() — parse single-digit model output → result
// ═══════════════════════════════════════════════════════════════

ClassificationResult SensorPipeline::parse_model_output(
    const std::string & raw_output, int batch_idx, double lat_ms)
{
    ClassificationResult r;
    r.batch_index = batch_idx;
    r.raw_output  = raw_output;
    r.latency_ms  = lat_ms;

    // Clean: trim, strip ``` fences
    std::string text = trim(raw_output);
    if (text.size() >= 6 && text.substr(0, 3) == "```") {
        size_t end = text.find("```", 3);
        text = (end != std::string::npos) ? trim(text.substr(3, end - 3)) : trim(text.substr(3));
    }

    // Try: single digit 0-6 (grammar output)
    std::regex digit_re(R"(\b([0-6])\b)");
    std::smatch m;
    if (std::regex_search(text, m, digit_re)) {
        r.category = std::stoi(m[1].str());
    } else {
        r.success = false;
        r.error   = "无法解析模型输出: \"" + text + "\"";
        r.review_state = "required";
        r.review_reason = {"qwen_fallback_uncalibrated", "qwen_format_error"};
        r.safety_priority = "P1";
        r.action = "continue_with_review";
        r.level = "warning";
        return r;
    }

    // Lookup category info
    auto * info = SensorPipeline::lookup_category(r.category);
    if (info) {
        r.category_name = info->name;
        r.action        = info->action;
        r.level         = info->level;
        r.summary       = info->summary;
    } else {
        r.category_name = "unknown_" + std::to_string(r.category);
        r.action  = "warning";
        r.level   = "warning";
        r.summary = r.category_name;
    }

    // A class token is an uncalibrated observation.  It cannot become a
    // normal/fault prediction, a probability, or P0 by category alone.
    r.local_prediction.clear();
    r.review_state = "required";
    r.review_reason = {"qwen_fallback_uncalibrated"};
    r.safety_priority = "P1";
    r.safety_trigger.clear();
    r.action = "continue_with_review";
    r.level = "warning";
    r.summary = "Uncalibrated Qwen observation; cloud review required";

    r.success = true;
    return r;
}

// ═══════════════════════════════════════════════════════════════
//  compute_stats()
// ═══════════════════════════════════════════════════════════════

PipelineStats SensorPipeline::compute_stats() const {
    PipelineStats s;
    s.total_batches = (int)_results.size();

    std::vector<double> lats;
    for (const auto & r : _results) {
        if (r.success) {
            s.success_count++;
            lats.push_back(r.latency_ms);
            if (r.latency_ms <= 200.0) s.under_200ms++;
            if (r.review_state == "required") s.review_required_count++;
            s.category_distribution[r.category]++;
        }
    }

    if (!lats.empty()) {
        double sum = 0.0;
        for (double l : lats) sum += l;
        s.avg_latency_ms = sum / lats.size();

        std::sort(lats.begin(), lats.end());
        size_t p95_idx = (size_t)(lats.size() * 0.95);
        if (p95_idx >= lats.size()) p95_idx = lats.size() - 1;
        s.p95_latency_ms = lats[p95_idx];
    }

    return s;
}
