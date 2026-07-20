#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

// ── Network condition assessment ──────────────────────────────

enum class NetworkCondition {
    Healthy,     // RTT < 200ms, cloud reachable → cloud PPO makes decisions
    Degraded,    // RTT 200-500ms → hybrid: high-conf local, low-conf queue
    Offline,     // RTT > 500ms or heartbeat failed → fully autonomous
};

// ── Local decision action ─────────────────────────────────────

enum class LocalAction {
    FollowCloud,       // Network good, wait for cloud PPO instructions
    LocalInfer,        // Run local inference immediately
    QueueForCloud,     // Queue task, wait for network recovery then cloud review
    P2PConsensus,      // Use P2P mesh consensus as substitute for cloud arbitration
};

// ── Decision record (for offline logging) ─────────────────────

struct DecisionSnapshot {
    NetworkCondition condition;
    LocalAction      action;
    float            confidence_threshold_used = 0.78f;
    std::string      reason;
    int64_t          timestamp_ms = 0;
};

// ═══════════════════════════════════════════════════════════════
// EdgeLocalDecision — local autonomous fallback decision engine
// ═══════════════════════════════════════════════════════════════

class EdgeLocalDecision {
public:
    EdgeLocalDecision() = default;
    ~EdgeLocalDecision() = default;

    // ── Cloud parameter sync (values to use when offline) ──────
    // Called by ModelSyncManager when cloud pushes rule updates.
    void update_cloud_params(float confidence_threshold,
                             float anomaly_threshold,
                             int   max_queue_len);

    // ── Core: decide action based on network + task confidence ──
    // confidence: local model's confidence in its output (0.0-1.0)
    // network: current network condition from assess_network()
    LocalAction decide(float confidence, NetworkCondition network);

    // ── Network assessment ─────────────────────────────────────
    // rtt_ms: measured RTT to cloud (-1 = unknown/unreachable)
    // cloud_reachable: result of health check
    // consecutive_hb_fails: number of consecutive heartbeat failures
    NetworkCondition assess_network(int rtt_ms, bool cloud_reachable,
                                     int consecutive_hb_fails);

    // ── Offline decision log ───────────────────────────────────
    // Returns and clears accumulated offline decisions.
    // Called when network recovers, to upload to cloud for PPO training.
    std::vector<DecisionSnapshot> flush_offline_decisions();

    // ── Queries ────────────────────────────────────────────────
    NetworkCondition current_condition() const { return _current_condition; }

    // Returns the currently active confidence threshold.
    // When offline, returns the lowered threshold (cloud value - 0.15).
    float active_confidence_threshold() const;

    // Number of decisions made while offline (not yet flushed)
    size_t offline_decision_count() const { return _offline_log.size(); }

private:
    // Cloud-synced parameters (the "survival kit" when network drops)
    float _cloud_confidence_threshold = 0.78f;
    float _cloud_anomaly_threshold   = 0.85f;
    int   _cloud_max_queue_len       = 10;

    NetworkCondition _current_condition = NetworkCondition::Healthy;

    // Offline decision log
    std::vector<DecisionSnapshot> _offline_log;
    static constexpr size_t MAX_OFFLINE_LOG = 1000;
};
