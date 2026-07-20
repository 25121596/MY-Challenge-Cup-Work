#include "edge_local_decision.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

// ═══════════════════════════════════════════════════════════════
// Cloud parameter sync
// ═══════════════════════════════════════════════════════════════

void EdgeLocalDecision::update_cloud_params(float confidence_threshold,
                                             float anomaly_threshold,
                                             int   max_queue_len) {
    _cloud_confidence_threshold = confidence_threshold;
    _cloud_anomaly_threshold    = anomaly_threshold;
    _cloud_max_queue_len        = max_queue_len;
}

// ═══════════════════════════════════════════════════════════════
// Network assessment
// ═══════════════════════════════════════════════════════════════

NetworkCondition EdgeLocalDecision::assess_network(int  rtt_ms,
                                                    bool cloud_reachable,
                                                    int  consecutive_hb_fails) {
    // Offline: RTT > 500ms, RTT < 0 (unknown), too many heartbeat failures,
    //          or cloud not reachable
    if (rtt_ms > 500 || rtt_ms < 0 || consecutive_hb_fails >= 5 || !cloud_reachable) {
        _current_condition = NetworkCondition::Offline;
        return NetworkCondition::Offline;
    }

    // Degraded: RTT in 200-500ms range, or moderate heartbeat failures
    if ((rtt_ms >= 200 && rtt_ms <= 500) ||
        (consecutive_hb_fails >= 2 && consecutive_hb_fails <= 4)) {
        _current_condition = NetworkCondition::Degraded;
        return NetworkCondition::Degraded;
    }

    // Healthy: RTT < 200ms, cloud reachable, heartbeat ok
    _current_condition = NetworkCondition::Healthy;
    return NetworkCondition::Healthy;
}

// ═══════════════════════════════════════════════════════════════
// Decision engine
// ═══════════════════════════════════════════════════════════════

LocalAction EdgeLocalDecision::decide(float confidence,
                                       NetworkCondition network) {
    DecisionSnapshot snap;
    snap.condition              = network;
    snap.confidence_threshold_used = active_confidence_threshold();

    // Capture timestamp
    auto now = std::chrono::system_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()).count();
    snap.timestamp_ms = static_cast<int64_t>(ms);

    LocalAction action;

    switch (network) {
        case NetworkCondition::Healthy: {
            // Network is good — let cloud PPO handle decisions normally.
            action = LocalAction::FollowCloud;
            snap.reason = "Network healthy: delegating to cloud PPO";
            break;
        }

        case NetworkCondition::Degraded: {
            // Hybrid mode: high-confidence tasks run locally,
            // low-confidence tasks are queued for later cloud review.
            if (confidence >= _cloud_confidence_threshold) {
                action = LocalAction::LocalInfer;
                snap.reason = "Degraded network, high confidence ("
                              + std::to_string(confidence) + " >= "
                              + std::to_string(_cloud_confidence_threshold)
                              + "): local inference";
            } else {
                action = LocalAction::QueueForCloud;
                snap.reason = "Degraded network, low confidence ("
                              + std::to_string(confidence) + " < "
                              + std::to_string(_cloud_confidence_threshold)
                              + "): queue for cloud review";
            }
            break;
        }

        case NetworkCondition::Offline: {
            float lowered = active_confidence_threshold(); // cloud - 0.15, min 0.5

            if (confidence >= lowered) {
                // Confident enough to act autonomously
                action = LocalAction::LocalInfer;
                snap.reason = "Offline, sufficient confidence ("
                              + std::to_string(confidence) + " >= "
                              + std::to_string(lowered)
                              + "): local inference (best effort)";
            } else {
                // Low confidence — attempt P2P consensus if peers are available.
                // If no P2P peers, the caller should fall back to LocalInfer.
                action = LocalAction::P2PConsensus;
                snap.reason = "Offline, low confidence ("
                              + std::to_string(confidence) + " < "
                              + std::to_string(lowered)
                              + "): seeking P2P consensus";
            }
            break;
        }
    }

    snap.action = action;

    // Log the decision to the offline log (capped at MAX_OFFLINE_LOG)
    if (_offline_log.size() < MAX_OFFLINE_LOG) {
        _offline_log.push_back(snap);
    } else {
        // Drop oldest to make room
        _offline_log.erase(_offline_log.begin());
        _offline_log.push_back(snap);
    }

    fprintf(stderr, "[EdgeLocalDecision] condition=%d confidence=%.2f "
            "threshold=%.2f action=%d reason=%s\n",
            static_cast<int>(network), confidence,
            snap.confidence_threshold_used,
            static_cast<int>(action), snap.reason.c_str());

    return action;
}

// ═══════════════════════════════════════════════════════════════
// Offline decision log
// ═══════════════════════════════════════════════════════════════

std::vector<DecisionSnapshot> EdgeLocalDecision::flush_offline_decisions() {
    std::vector<DecisionSnapshot> result;
    result.swap(_offline_log);
    return result;
}

// ═══════════════════════════════════════════════════════════════
// Queries
// ═══════════════════════════════════════════════════════════════

float EdgeLocalDecision::active_confidence_threshold() const {
    switch (_current_condition) {
        case NetworkCondition::Healthy:
        case NetworkCondition::Degraded:
            return _cloud_confidence_threshold;

        case NetworkCondition::Offline:
            // Lower the bar by 15% when offline, but never below 0.5
            return std::max(0.5f, _cloud_confidence_threshold - 0.15f);
    }
    return _cloud_confidence_threshold;
}
