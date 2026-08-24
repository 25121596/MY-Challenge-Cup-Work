#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

class EdgeEngine;

// ── Category info (used by pipeline and UI) ────────────────────

struct CategoryInfo {
    int         code;
    const char * name;
    const char * action;
    const char * level;
    const char * summary;
};

// ── One averaged batch of sensor readings ──────────────────────

struct DataBatch {
    int    batch_index = 0;
    int    row_count   = 0;
    std::map<std::string, double> readings;   // standard_name → avg value
    std::string prompt_text;                   // filled template ready for inference
};

// ── Classification result for one batch ────────────────────────

struct ClassificationResult {
    int    batch_index      = -1;
    int    category         = -1;    // 0-6, -1 on error
    std::string category_name;       // "正常", "轴承磨损", ...
    std::string action;              // protect / continue_with_review / continue
    std::string level;               // info/warning/critical
    std::string summary;             // human-readable diagnosis
    // This native path emits a Qwen class token. It has no retained-set
    // calibration, so it never manufactures a probability or confidence.
    std::string local_prediction;    // empty for uncalibrated Qwen
    std::string review_state;        // required / not_required
    std::vector<std::string> review_reason;
    std::string safety_priority;     // P0 / P1 / P2 safety urgency only
    std::string safety_trigger;
    double  latency_ms       = 0.0;
    std::string raw_output;          // raw model response (for debug)
    std::string error;               // set on failure
    bool    success           = false;
};

// ── Aggregate statistics ───────────────────────────────────────

struct PipelineStats {
    int    total_batches   = 0;
    int    success_count   = 0;
    double avg_latency_ms  = 0.0;
    double p95_latency_ms  = 0.0;
    int    under_200ms     = 0;
    int    review_required_count = 0;
    std::map<int, int> category_distribution;
};

// ═══════════════════════════════════════════════════════════════
// SensorPipeline
// ═══════════════════════════════════════════════════════════════
//
// Ports the Python preprocessing + inference pipeline into C++:
//   sensor_csv_to_input.py  → CSV parsing, column mapping, batching
//   edge_inference.py       → prompt building, GBNF inference, output parsing
//
// Usage:
//   SensorPipeline pipe(&engine);
//   pipe.open("sensors.csv");
//   pipe.prepare_batches(60);
//   for (auto & b : pipe.batches())
//       pipe.results().push_back(pipe.classify_batch(b));
//   auto stats = pipe.compute_stats();

class SensorPipeline {
public:
    SensorPipeline() = default;
    explicit SensorPipeline(EdgeEngine * engine) : _engine(engine) {}

    void set_engine(EdgeEngine * engine) { _engine = engine; }

    // ── Lifecycle ───────────────────────────────────────────────
    bool open(const std::string & filepath);
    void close();
    bool is_open() const { return _opened; }

    // ── Detection ───────────────────────────────────────────────
    const std::string & data_type() const { return _data_type; }
    int row_count()    const { return (int)_rows.size(); }
    int column_count() const { return (int)_headers.size(); }
    const std::vector<std::string> & headers() const { return _headers; }

    // ── Batching ────────────────────────────────────────────────
    int prepare_batches(int batch_size = 60);
    const std::vector<DataBatch> & batches() const { return _batches; }

    // ── Inference (one batch, blocking, ~50ms) ──────────────────
    ClassificationResult classify_batch(const DataBatch & batch);

    // ── Results ─────────────────────────────────────────────────
    std::vector<ClassificationResult>       & results()       { return _results; }
    const std::vector<ClassificationResult> & results() const { return _results; }
    PipelineStats compute_stats() const;

    // ── Static helpers (also used by UI) ──────────────────────
    static const struct CategoryInfo * lookup_category(int code);

private:
    // CSV parsing internals
    char        detect_delimiter(std::ifstream & file);
    void        detect_data_type(const std::string & first_line);
    void        map_headers();
    std::string build_prompt(const std::map<std::string, double> & readings);
    ClassificationResult parse_model_output(const std::string & raw, int idx, double lat);

    EdgeEngine * _engine    = nullptr;
    std::string  _filepath;
    std::string  _data_type = "generic";  // hydraulic / motor / milling / generic
    char         _delimiter = ',';

    std::vector<std::string>              _headers;
    std::vector<std::vector<double>>      _rows;
    std::map<std::string, std::string>    _header_map;  // csv_col → standard_name

    std::vector<DataBatch>              _batches;
    std::vector<ClassificationResult>   _results;
    bool _opened = false;
};
