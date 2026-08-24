#include "edge_local_decision.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

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

LocalAction EdgeLocalDecision::decide(bool review_required,
                                       NetworkCondition network) {
    DecisionSnapshot snap;
    snap.condition              = network;
    snap.review_required        = review_required;

    // Capture timestamp
    auto now = std::chrono::system_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()).count();
    snap.timestamp_ms = static_cast<int64_t>(ms);

    LocalAction action;

    switch (network) {
        case NetworkCondition::Healthy: {
            // Network is good — let the cloud review path handle it normally.
            action = LocalAction::FollowCloud;
            snap.reason = review_required ? "Network healthy: cloud review required"
                                          : "Network healthy: local result may continue";
            break;
        }

        case NetworkCondition::Degraded: {
            if (!review_required) {
                action = LocalAction::LocalInfer;
                snap.reason = "Degraded network, review not required: local inference";
            } else {
                action = LocalAction::QueueForCloud;
                snap.reason = "Degraded network, review required: queue for cloud review";
            }
            break;
        }

        case NetworkCondition::Offline: {
            if (!review_required) {
                action = LocalAction::LocalInfer;
                snap.reason = "Offline, review not required: local inference (best effort)";
            } else {
                action = LocalAction::P2PConsensus;
                snap.reason = "Offline, review required: seeking P2P consensus";
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

    fprintf(stderr, "[EdgeLocalDecision] condition=%d review_required=%d "
            "action=%d reason=%s\n",
            static_cast<int>(network), review_required,
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

