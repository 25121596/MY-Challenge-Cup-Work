#pragma once

// ── P2P Mesh — edge-to-edge communication (Module 3.1) ──────────
//
// Enables direct peer-to-peer coordination between edge nodes
// for overlapping perception areas.  Uses:
//   - UDP broadcast for peer discovery
//   - TCP for reliable perception / decision-intent exchange
//   - Cloud-assisted discovery as fallback
//
// Platform: Winsock2 (Windows) / BSD sockets (Linux/macOS).

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ═══════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════

// A known peer node
struct PeerNode {
    std::string node_id;
    std::string ip_address;
    int         udp_port   = 0;
    int         tcp_port   = 0;
    std::string device_type;
    std::string monitored_asset;     // "production_line_A", "camera_intersection_3"
    std::string camera_fov;          // human-readable FOV description
    bool        online      = false;
    int64_t     last_seen_ms = 0;
};

// Perception report shared among peers (lightweight)
struct PerceptionReport {
    std::string report_id;
    std::string node_id;
    int64_t     timestamp_ms = 0;

    struct Detection {
        std::string target_id;       // unique object/target identifier
        std::string target_type;     // "vehicle", "person", "defect", "anomaly"
        std::string classification;  // "car", "pedestrian", "scratch"
        float       confidence  = 0.0f;
        std::string decision;        // "normal", "warning", "critical"
        std::string sensor_type;     // "camera", "lidar", "thermal"

        // Spatial info (optional)
        float geo_x = 0.0f, geo_y = 0.0f;
        struct { int x = 0, y = 0, w = 0, h = 0; } bbox;
    };
    std::vector<Detection> detections;
};

// Decision intent sent to specific peers (reliable, TCP)
struct DecisionIntent {
    std::string intent_id;
    std::string node_id;
    std::string target_id;
    std::string proposed_action;     // "stop_line", "alert_operator", etc.
    float       confidence   = 0.0f;
    std::string rationale;
    int64_t     timestamp_ms = 0;
};

// ═══════════════════════════════════════════════════════════════
// Callbacks
// ═══════════════════════════════════════════════════════════════

using peer_discovery_cb  = std::function<void(const std::string & peer_id, const std::string & ip, bool online)>;
using perception_cb      = std::function<void(const PerceptionReport &)>;
using decision_intent_cb = std::function<void(const DecisionIntent &)>;

// ═══════════════════════════════════════════════════════════════
// P2P Mesh
// ═══════════════════════════════════════════════════════════════

class P2PMesh {
public:
    P2PMesh();
    ~P2PMesh();

    // ── Configuration ──────────────────────────────────────────
    void set_node_id(const std::string & id)       { _node_id = id; }
    void set_listen_ports(int udp_port, int tcp_port);
    void set_cloud_discovery_url(const std::string & url) { _cloud_disc_url = url; }

    // ── Lifecycle ──────────────────────────────────────────────
    bool start();
    void stop();
    bool is_running() const { return _running.load(); }

    // ── Peer management ────────────────────────────────────────
    // Discover peers (cloud-assisted or broadcast)
    std::vector<PeerNode> discover_peers();

    // Manually add a known peer
    void add_peer(const PeerNode & peer);

    // Get all currently known peers
    std::vector<PeerNode> known_peers() const;

    // ── Communication ──────────────────────────────────────────
    // Broadcast a perception report to all online peers
    void broadcast_perception(const PerceptionReport & report);

    // Send a decision intent to a specific peer (TCP)
    void send_decision_intent(const std::string & peer_id,
                              const DecisionIntent & intent);

    // ── Callbacks ──────────────────────────────────────────────
    peer_discovery_cb   on_peer_discovered;
    perception_cb       on_perception;
    decision_intent_cb  on_decision_intent;

private:
    // UDP broadcast: "EDGE_DISCOVER|<node_id>|<tcp_port>|<device_type>"
    void udp_listen_loop();
    void udp_broadcast_presence();

    // TCP server: accept connections, receive JSON payloads
    void tcp_listen_loop();
    void handle_peer_connection(int client_fd);

    // TCP client: connect to a peer and send data
    bool send_tcp(const std::string & ip, int port, const std::string & payload);

    // Cloud-assisted discovery
    std::vector<PeerNode> cloud_discover();

    // Serialization
    std::string serialize_report(const PerceptionReport & r) const;
    PerceptionReport parse_report(const std::string & json) const;
    std::string serialize_intent(const DecisionIntent & i) const;

    // ── State ──────────────────────────────────────────────────
    std::string _node_id     = "edge-unknown";
    int         _udp_port    = 15555;
    int         _tcp_port    = 15556;
    std::string _cloud_disc_url;

    std::thread _udp_thread;
    std::thread _tcp_thread;
    std::atomic<bool> _running{false};

    // peer_id → PeerNode
    std::map<std::string, PeerNode> _peers;
    mutable std::mutex _mutex;

    // Platform-specific
#ifdef _WIN32
    bool _wsa_initialized = false;
#endif
};
