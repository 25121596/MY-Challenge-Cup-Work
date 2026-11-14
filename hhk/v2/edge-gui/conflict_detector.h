#pragma once

// ── Conflict detection & local resolution (Module 3.2) ───────────
//
// Detects inconsistent judgments between edge nodes on the same
// target/event.  Tries local rule-based resolution first; escalates
// to cloud if unresolvable.
//
// Competition targets:
//   Conflict ratio  ≤ 5%
//   Resolution rate ≥ 90%

#include "p2p_mesh.h"

#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Forward declaration
class EdgeCloudClient;

// ═══════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════

// A single detected conflict between two nodes
struct ConflictRecord {
    std::string conflict_id;
    std::string target_id;
    std::string target_type;

    // Our judgment
    std::string our_node_id;
    std::string our_decision;
    float       our_confidence  = 0.0f;
    std::string our_sensor_type;

    // Peer's judgment
    std::string peer_node_id;
    std::string peer_decision;
    float       peer_confidence = 0.0f;
    std::string peer_sensor_type;

    // Conflict details
    enum class Severity { Low, Medium, High, Critical };
    Severity    severity = Severity::Medium;
    std::string reason;       // "class_mismatch", "confidence_gap", "action_conflict"

    // Resolution
    bool        resolved        = false;
    bool        resolved_locally = false;
    std::string resolution_method;  // "confidence", "authority", "proximity", "consensus", "cloud"
    std::string final_decision;
    std::string final_rationale;
    int64_t     timestamp_ms    = 0;
};

// Resolution rule configuration
struct ConflictRule {
    std::string name;            // "proximity_priority", "sensor_authority", "confidence_margin"
    int         priority = 0;    // lower = higher priority

    // For sensor authority: sensor_type → weight (0..1)
    std::map<std::string, float> sensor_weights;

    // For confidence margin: if gap > this, higher confidence wins
    float confidence_margin = 0.3f;

    // Default sensor weights
    static std::map<std::string, float> default_sensor_weights() {
        return {
            {"lidar",    1.0f},
            {"radar",    0.9f},
            {"thermal",  0.8f},
            {"camera",   0.7f},
            {"audio",    0.5f},
            {"unknown",  0.5f},
        };
    }
};

// Statistics for competition metrics
struct ConflictStats {
    int total_decisions     = 0;
    int conflicts_detected  = 0;
    int resolved_locally    = 0;
    int escalated_to_cloud  = 0;
    int resolution_success  = 0;   // includes cloud

    double conflict_ratio() const {
        return total_decisions > 0
            ? (double)conflicts_detected / total_decisions * 100.0 : 0.0;
    }
    double resolution_rate() const {
        return conflicts_detected > 0
            ? (double)resolution_success / conflicts_detected * 100.0 : 100.0;
    }
};

// ═══════════════════════════════════════════════════════════════
// Callbacks
// ═══════════════════════════════════════════════════════════════

using conflict_detected_cb  = std::function<void(const ConflictRecord &)>;
using conflict_resolved_cb  = std::function<void(const ConflictRecord &)>;
using conflict_escalated_cb = std::function<void(const ConflictRecord &)>;

// ═══════════════════════════════════════════════════════════════
// Conflict detector
// ═══════════════════════════════════════════════════════════════

class ConflictDetector {
public:
    ConflictDetector();
    ~ConflictDetector();

    void set_node_id(const std::string & id) { _node_id = id; }
    void set_cloud_client(EdgeCloudClient * cloud) { _cloud = cloud; }

    // ── Feed in data ───────────────────────────────────────────
    // Our local perception (after local inference)
    void submit_local_perception(const PerceptionReport & report);

    // Peer's perception (from P2PMesh callback)
    void receive_peer_perception(const PerceptionReport & report);

    // Peer's decision intent (from P2PMesh callback)
    void receive_peer_intent(const DecisionIntent & intent);

    // ── Conflict detection & resolution ────────────────────────
    // Check all active targets for conflicts
    std::vector<ConflictRecord> check_all_conflicts();

    // Try to resolve a conflict locally. Returns true if resolved.
    bool resolve_locally(ConflictRecord & conflict);

    // Escalate an unresolved conflict to the cloud
    void escalate_to_cloud(const ConflictRecord & conflict);

    // ── Rule management ────────────────────────────────────────
    void add_rule(const ConflictRule & rule);
    void remove_rule(const std::string & name);
    void clear_rules();
    void load_default_rules();

    // ── Statistics ─────────────────────────────────────────────
    ConflictStats stats() const { return _stats; }
    void reset_stats();

    // ── Callbacks ──────────────────────────────────────────────
    conflict_detected_cb  on_conflict_detected;
    conflict_resolved_cb  on_conflict_resolved;
    conflict_escalated_cb on_conflict_escalated;

private:
    // Target matching
    bool is_same_target(const PerceptionReport::Detection & a,
                        const PerceptionReport::Detection & b) const;

    // Conflict detection predicates
    bool is_classification_conflict(const PerceptionReport::Detection & a,
                                     const PerceptionReport::Detection & b) const;
    bool is_confidence_conflict(const PerceptionReport::Detection & a,
                                 const PerceptionReport::Detection & b) const;
    ConflictRecord::Severity assess_severity(
        const PerceptionReport::Detection & a,
        const PerceptionReport::Detection & b) const;

    // Resolution strategies (try in priority order)
    bool resolve_by_confidence(ConflictRecord & c);
    bool resolve_by_authority(ConflictRecord & c);
    bool resolve_by_proximity(ConflictRecord & c);
    bool resolve_by_consensus(ConflictRecord & c);

    // IoU computation for bbox overlap
    double compute_iou(const PerceptionReport::Detection & a,
                       const PerceptionReport::Detection & b) const;

    std::string _node_id;
    EdgeCloudClient * _cloud = nullptr;

    // Cache: target_id → our detection
    std::map<std::string, PerceptionReport::Detection> _local_cache;
    // Cache: peer_node_id → (target_id → detection)
    std::map<std::string, std::map<std::string, PerceptionReport::Detection>> _peer_cache;
    // Cache: peer intents
    std::map<std::string, DecisionIntent> _peer_intents;

    std::vector<ConflictRule> _rules;
    std::deque<ConflictRecord> _conflict_history;   // last 200
    static constexpr size_t MAX_HISTORY = 200;

    ConflictStats _stats;
    mutable std::mutex _mutex;
};
