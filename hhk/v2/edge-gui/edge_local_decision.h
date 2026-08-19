#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

// ── Network condition assessment ──────────────────────────────

enum class NetworkCondition {
    Healthy,     // RTT < 200ms, cloud reachable → cloud path is available
    Degraded,    // RTT 200-500ms → review is queued when possible
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
    bool             review_required = true;
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

    // ── Core: decide transport handling, not model confidence ──
    // Qwen results are uncalibrated and must set review_required=true.
    LocalAction decide(bool review_required, NetworkCondition network);

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

    // Number of decisions made while offline (not yet flushed)
    size_t offline_decision_count() const { return _offline_log.size(); }

private:
    NetworkCondition _current_condition = NetworkCondition::Healthy;

    // Offline decision log
    std::vector<DecisionSnapshot> _offline_log;
    static constexpr size_t MAX_OFFLINE_LOG = 1000;
};
