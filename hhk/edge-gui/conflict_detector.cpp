#include "conflict_detector.h"
#include "edge_cloud.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>

// ═══════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════

ConflictDetector::ConflictDetector()  { load_default_rules(); }
ConflictDetector::~ConflictDetector() {}

void ConflictDetector::load_default_rules() {
    _rules.clear();
    // Rule 1: Confidence — if gap > 30%, higher confidence wins
    _rules.push_back({"confidence_margin", 1, {}, 0.3f});
    // Rule 2: Sensor authority — prefer higher-authority sensor
    _rules.push_back({"sensor_authority", 2,
                      ConflictRule::default_sensor_weights(), 0.0f});
    // Rule 3: Proximity — closer node wins (placeholder)
    _rules.push_back({"proximity_priority", 3, {}, 0.0f});
    // Rule 4: Consensus — majority vote (needs >= 3 peers)
    _rules.push_back({"consensus_vote", 4, {}, 0.0f});
}

// ═══════════════════════════════════════════════════════════════
// Data input
// ═══════════════════════════════════════════════════════════════

void ConflictDetector::submit_local_perception(const PerceptionReport & report) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto & d : report.detections) {
        _local_cache[d.target_id] = d;
        ++_stats.total_decisions;
    }
}

void ConflictDetector::receive_peer_perception(const PerceptionReport & report) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto & d : report.detections) {
        _peer_cache[report.node_id][d.target_id] = d;
    }
}

void ConflictDetector::receive_peer_intent(const DecisionIntent & intent) {
    std::lock_guard<std::mutex> lock(_mutex);
    _peer_intents[intent.node_id + ":" + intent.intent_id] = intent;
}

// ═══════════════════════════════════════════════════════════════
// Conflict detection
// ═══════════════════════════════════════════════════════════════

bool ConflictDetector::is_same_target(
    const PerceptionReport::Detection & a,
    const PerceptionReport::Detection & b) const
{
    // Exact ID match
    if (!a.target_id.empty() && a.target_id == b.target_id) return true;

    // IoU-based match for camera detections
    double iou = compute_iou(a, b);
    if (iou > 0.5) return true;

    // Spatial proximity (if geo coordinates available)
    if (a.geo_x != 0.0f || a.geo_y != 0.0f) {
        double dx = a.geo_x - b.geo_x;
        double dy = a.geo_y - b.geo_y;
        if (std::sqrt(dx * dx + dy * dy) < 10.0) return true;  // within 10 meters
    }

    return false;
}

double ConflictDetector::compute_iou(
    const PerceptionReport::Detection & a,
    const PerceptionReport::Detection & b) const
{
    // If either has no bbox, can't compute IoU
    if (a.bbox.w == 0 || a.bbox.h == 0 || b.bbox.w == 0 || b.bbox.h == 0)
        return 0.0;

    int x1 = std::max(a.bbox.x, b.bbox.x);
    int y1 = std::max(a.bbox.y, b.bbox.y);
    int x2 = std::min(a.bbox.x + a.bbox.w, b.bbox.x + b.bbox.w);
    int y2 = std::min(a.bbox.y + a.bbox.h, b.bbox.y + b.bbox.h);

    if (x2 <= x1 || y2 <= y1) return 0.0;

    double inter = (double)(x2 - x1) * (y2 - y1);
    double area_a = (double)a.bbox.w * a.bbox.h;
    double area_b = (double)b.bbox.w * b.bbox.h;
    double union_area = area_a + area_b - inter;

    return union_area > 0.0 ? inter / union_area : 0.0;
}

bool ConflictDetector::is_classification_conflict(
    const PerceptionReport::Detection & a,
    const PerceptionReport::Detection & b) const
{
    // Different classification of the same target
    return a.classification != b.classification;
}

bool ConflictDetector::is_confidence_conflict(
    const PerceptionReport::Detection & a,
    const PerceptionReport::Detection & b) const
{
    // Same classification but significantly different confidence
    return std::abs(a.confidence - b.confidence) > 0.5;
}

ConflictRecord::Severity ConflictDetector::assess_severity(
    const PerceptionReport::Detection & a,
    const PerceptionReport::Detection & b) const
{
    // Critical: conflicting "critical" vs "normal" decisions
    if ((a.decision == "critical" && b.decision == "normal") ||
        (a.decision == "normal"  && b.decision == "critical"))
        return ConflictRecord::Severity::Critical;

    // High: conflicting classifications
    if (a.classification != b.classification)
        return ConflictRecord::Severity::High;

    // Medium: confidence gap
    if (std::abs(a.confidence - b.confidence) > 0.5)
        return ConflictRecord::Severity::Medium;

    return ConflictRecord::Severity::Low;
}

std::vector<ConflictRecord> ConflictDetector::check_all_conflicts() {
    std::vector<ConflictRecord> conflicts;
    std::vector<ConflictRecord> to_notify;  // notify outside lock

    {
        std::lock_guard<std::mutex> lock(_mutex);

        for (const auto & [target_id, our] : _local_cache) {
            for (const auto & [peer_id, peer_map] : _peer_cache) {
                for (const auto & [peer_target_id, peer] : peer_map) {
                    if (!is_same_target(our, peer)) continue;

                    bool class_conflict = is_classification_conflict(our, peer);
                    bool conf_conflict  = is_confidence_conflict(our, peer);
                    bool action_conflict = our.decision != peer.decision;

                    if (!class_conflict && !conf_conflict && !action_conflict)
                        continue;

                    ConflictRecord cr;
                    cr.conflict_id     = target_id + "_vs_" + peer_id;
                    cr.target_id       = target_id;
                    cr.target_type     = our.target_type;
                    cr.our_node_id     = _node_id;
                    cr.our_decision    = our.decision;
                    cr.our_confidence  = our.confidence;
                    cr.our_sensor_type = our.sensor_type;
                    cr.peer_node_id    = peer_id;
                    cr.peer_decision   = peer.decision;
                    cr.peer_confidence = peer.confidence;
                    cr.peer_sensor_type = peer.sensor_type;
                    cr.severity        = assess_severity(our, peer);
                    cr.reason          = class_conflict ? "class_mismatch" :
                                         conf_conflict  ? "confidence_gap" :
                                                          "action_conflict";
                    cr.timestamp_ms    = std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();

                    ++_stats.conflicts_detected;
                    conflicts.push_back(cr);
                    to_notify.push_back(cr);

                    // Add to history
                    _conflict_history.push_back(cr);
                    if (_conflict_history.size() > MAX_HISTORY)
                        _conflict_history.pop_front();
                }
            }
        }
    }  // unlock before invoking callbacks

    // Notify outside lock to prevent deadlock if callback calls back into us
    for (const auto & cr : to_notify) {
        if (on_conflict_detected) on_conflict_detected(cr);
    }

    return conflicts;
}

// ═══════════════════════════════════════════════════════════════
// Local resolution
// ═══════════════════════════════════════════════════════════════

bool ConflictDetector::resolve_locally(ConflictRecord & conflict) {
    // Try each rule in priority order
    for (const auto & rule : _rules) {
        bool resolved = false;
        if (rule.name == "confidence_margin")
            resolved = resolve_by_confidence(conflict);
        else if (rule.name == "sensor_authority")
            resolved = resolve_by_authority(conflict);
        else if (rule.name == "proximity_priority")
            resolved = resolve_by_proximity(conflict);
        else if (rule.name == "consensus_vote")
            resolved = resolve_by_consensus(conflict);

        if (resolved) {
            conflict.resolved         = true;
            conflict.resolved_locally = true;
            ++_stats.resolved_locally;
            ++_stats.resolution_success;
            if (on_conflict_resolved) on_conflict_resolved(conflict);
            return true;
        }
    }
    return false;
}

bool ConflictDetector::resolve_by_confidence(ConflictRecord & c) {
    // Find the confidence margin rule
    auto it = std::find_if(_rules.begin(), _rules.end(),
        [](const ConflictRule & r) { return r.name == "confidence_margin"; });
    float margin = (it != _rules.end()) ? it->confidence_margin : 0.3f;

    double gap = std::abs(c.our_confidence - c.peer_confidence);
    if (gap > margin) {
        if (c.our_confidence > c.peer_confidence) {
            c.final_decision   = c.our_decision;
            c.final_rationale = "Our confidence (" +
                std::to_string(c.our_confidence) + ") > peer (" +
                std::to_string(c.peer_confidence) + ")";
        } else {
            c.final_decision   = c.peer_decision;
            c.final_rationale = "Peer confidence (" +
                std::to_string(c.peer_confidence) + ") > ours (" +
                std::to_string(c.our_confidence) + ")";
        }
        c.resolution_method = "confidence";
        return true;
    }
    return false;
}

bool ConflictDetector::resolve_by_authority(ConflictRecord & c) {
    auto it = std::find_if(_rules.begin(), _rules.end(),
        [](const ConflictRule & r) { return r.name == "sensor_authority"; });
    if (it == _rules.end()) return false;

    auto & weights = it->sensor_weights;
    auto our_w   = weights.count(c.our_sensor_type)  ? weights[c.our_sensor_type]  : 0.5f;
    auto peer_w  = weights.count(c.peer_sensor_type) ? weights[c.peer_sensor_type] : 0.5f;

    // Only resolve if there's a clear authority advantage (> 20%)
    if (std::abs(our_w - peer_w) > 0.2f) {
        if (our_w > peer_w) {
            c.final_decision   = c.our_decision;
            c.final_rationale = "Our sensor (" + c.our_sensor_type +
                ") has higher authority (" + std::to_string(our_w) +
                ") than peer (" + c.peer_sensor_type + ": " +
                std::to_string(peer_w) + ")";
        } else {
            c.final_decision   = c.peer_decision;
            c.final_rationale = "Peer sensor (" + c.peer_sensor_type +
                ") has higher authority (" + std::to_string(peer_w) +
                ") than ours (" + c.our_sensor_type + ": " +
                std::to_string(our_w) + ")";
        }
        c.resolution_method = "authority";
        return true;
    }
    return false;
}

bool ConflictDetector::resolve_by_proximity(ConflictRecord & c) {
    // Placeholder — requires actual distance measurements from
    // geo_x/geo_y in perception reports. For now, this rule
    // doesn't fire unless we have that data.
    //
    // When implemented: closer node to the target wins.
    (void)c;
    return false;
}

bool ConflictDetector::resolve_by_consensus(ConflictRecord & c) {
    // Requires at least 3 peers with opinions on this target.
    // Count how many peers agree with us vs with the conflicting peer.
    std::lock_guard<std::mutex> lock(_mutex);

    int our_votes   = 1;  // us
    int peer_votes  = 1;  // the conflicting peer

    for (const auto & [pid, pmap] : _peer_cache) {
        if (pid == c.peer_node_id) continue;
        for (const auto & [tid, det] : pmap) {
            if (tid == c.target_id || is_same_target(
                _local_cache[c.target_id], det)) {
                if (det.classification ==
                    _local_cache[c.target_id].classification) ++our_votes;
                else ++peer_votes;
            }
        }
    }

    int total = our_votes + peer_votes;
    if (total >= 3) {
        if (our_votes > peer_votes) {
            c.final_decision   = c.our_decision;
            c.final_rationale = "Consensus: " + std::to_string(our_votes) +
                "/" + std::to_string(total) + " agree with us";
        } else {
            c.final_decision   = c.peer_decision;
            c.final_rationale = "Consensus: " + std::to_string(peer_votes) +
                "/" + std::to_string(total) + " agree with peer";
        }
        c.resolution_method = "consensus";
        return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════
// Cloud escalation
// ═══════════════════════════════════════════════════════════════

void ConflictDetector::escalate_to_cloud(const ConflictRecord & conflict) {
    ++_stats.escalated_to_cloud;

    if (_cloud && _cloud->is_reachable()) {
        // Build escalation payload
        nlohmann::json j;
        j["conflict_id"]   = conflict.conflict_id;
        j["target_id"]     = conflict.target_id;
        j["target_type"]   = conflict.target_type;
        j["our_node_id"]   = conflict.our_node_id;
        j["our_decision"]  = conflict.our_decision;
        j["our_confidence"] = conflict.our_confidence;
        j["peer_node_id"]  = conflict.peer_node_id;
        j["peer_decision"] = conflict.peer_decision;
        j["peer_confidence"] = conflict.peer_confidence;
        j["reason"]        = conflict.reason;
        j["severity"]      = (int)conflict.severity;
        j["timestamp_ms"]  = conflict.timestamp_ms;

        // Send via EdgeCloudClient — use query_offload as a simple
        // way to transmit JSON to a cloud endpoint.
        // In production, this would POST to /api/v1/edge/conflicts.
        InferenceContext ictx;
        ictx.temperature    = 0.0f;
        ictx.max_new_tokens = 128;

        _cloud->query_offload(
            j.dump(), ictx,
            [this, conflict](const std::string & text, bool finished,
                             const std::string & err) {
                if (finished && err.empty()) {
                    // Cloud arbitration received — mark resolved
                    ConflictRecord resolved = conflict;
                    resolved.resolved         = true;
                    resolved.resolved_locally = false;
                    resolved.resolution_method = "cloud";
                    resolved.final_decision    = text;
                    resolved.final_rationale   = "Cloud arbitration";
                    ++_stats.resolution_success;
                    if (on_conflict_resolved) on_conflict_resolved(resolved);
                }
            },
            nullptr);
    }

    if (on_conflict_escalated) on_conflict_escalated(conflict);
}

// ═══════════════════════════════════════════════════════════════
// Rule management
// ═══════════════════════════════════════════════════════════════

void ConflictDetector::add_rule(const ConflictRule & rule) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = std::find_if(_rules.begin(), _rules.end(),
        [&](const ConflictRule & r) { return r.name == rule.name; });
    if (it != _rules.end()) *it = rule;
    else _rules.push_back(rule);
    // Re-sort by priority
    std::sort(_rules.begin(), _rules.end(),
        [](const ConflictRule & a, const ConflictRule & b) {
            return a.priority < b.priority;
        });
}

void ConflictDetector::remove_rule(const std::string & name) {
    std::lock_guard<std::mutex> lock(_mutex);
    _rules.erase(std::remove_if(_rules.begin(), _rules.end(),
        [&](const ConflictRule & r) { return r.name == name; }),
        _rules.end());
}

void ConflictDetector::clear_rules() {
    std::lock_guard<std::mutex> lock(_mutex);
    _rules.clear();
}

void ConflictDetector::reset_stats() {
    std::lock_guard<std::mutex> lock(_mutex);
    _stats = ConflictStats{};
    _conflict_history.clear();
}
