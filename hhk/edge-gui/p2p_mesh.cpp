#include "p2p_mesh.h"

#include "http.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   using socklen_t = int;
#  define SHUT_RDWR SD_BOTH
#else
#  include <sys/socket.h>
#  include <sys/select.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <fcntl.h>
#  include <errno.h>
#  define INVALID_SOCKET (-1)
#  define SOCKET_ERROR   (-1)
   using SOCKET = int;
#endif

// ═══════════════════════════════════════════════════════════════
// Platform helpers
// ═══════════════════════════════════════════════════════════════

static void socket_close(SOCKET s) {
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

static int socket_last_error() {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

static bool socket_would_block() {
#ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

static bool set_nonblocking(SOCKET s) {
#ifdef _WIN32
    u_long mode = 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
#else
    int flags = fcntl(s, F_GETFL, 0);
    return fcntl(s, F_SETFL, flags | O_NONBLOCK) >= 0;
#endif
}

// ═══════════════════════════════════════════════════════════════
// Lifecycle
// ═══════════════════════════════════════════════════════════════

P2PMesh::P2PMesh()  = default;
P2PMesh::~P2PMesh() { stop(); }

void P2PMesh::set_listen_ports(int udp, int tcp) {
    _udp_port = udp;
    _tcp_port = tcp;
}

bool P2PMesh::start() {
    if (_running.load()) return false;

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "P2PMesh: WSAStartup failed\n");
        return false;
    }
    _wsa_initialized = true;
#endif

    _running.store(true);
    _udp_thread = std::thread(&P2PMesh::udp_listen_loop, this);
    _tcp_thread = std::thread(&P2PMesh::tcp_listen_loop, this);
    return true;
}

void P2PMesh::stop() {
    _running.store(false);
    if (_udp_thread.joinable()) _udp_thread.join();
    if (_tcp_thread.joinable()) _tcp_thread.join();

#ifdef _WIN32
    if (_wsa_initialized) {
        WSACleanup();
        _wsa_initialized = false;
    }
#endif
}

// ═══════════════════════════════════════════════════════════════
// Peer discovery
// ═══════════════════════════════════════════════════════════════

std::vector<PeerNode> P2PMesh::discover_peers() {
    // Try cloud-assisted first
    auto peers = cloud_discover();
    if (!peers.empty()) {
        for (const auto & p : peers) add_peer(p);
    }
    // Also send a UDP broadcast to trigger local responses
    udp_broadcast_presence();

    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<PeerNode> result;
    for (const auto & [id, peer] : _peers) {
        result.push_back(peer);
    }
    return result;
}

std::vector<PeerNode> P2PMesh::cloud_discover() {
    std::vector<PeerNode> result;
    if (_cloud_disc_url.empty()) return result;

    try {
        std::string url = _cloud_disc_url +
            "/api/v1/edge/neighbors?device_id=" + _node_id;
        auto [cli, parts] = common_http_client(url);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(2, 0);
        auto res = cli.Get(parts.path.c_str());
        if (!res || res->status != 200) return result;

        auto j = nlohmann::json::parse(res->body);
        for (const auto & item : j) {
            PeerNode p;
            p.node_id         = item.value("node_id", "");
            p.ip_address      = item.value("ip", "");
            p.udp_port        = item.value("udp_port", 15555);
            p.tcp_port        = item.value("tcp_port", 15556);
            p.device_type     = item.value("device_type", "");
            p.monitored_asset = item.value("asset", "");
            p.camera_fov      = item.value("fov", "");
            p.online          = true;
            p.last_seen_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            result.push_back(p);
        }
    } catch (...) {}
    return result;
}

void P2PMesh::add_peer(const PeerNode & peer) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _peers.find(peer.node_id);
    if (it != _peers.end()) {
        it->second.last_seen_ms = peer.last_seen_ms;
        it->second.online = true;
    } else {
        _peers[peer.node_id] = peer;
        if (on_peer_discovered) {
            on_peer_discovered(peer.node_id, peer.ip_address, true);
        }
    }
}

std::vector<PeerNode> P2PMesh::known_peers() const {
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<PeerNode> result;
    for (const auto & [id, p] : _peers) result.push_back(p);
    return result;
}

// ═══════════════════════════════════════════════════════════════
// UDP — discovery + perception broadcast
// ═══════════════════════════════════════════════════════════════

void P2PMesh::udp_broadcast_presence() {
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) return;

    int broadcast = 1;
    setsockopt(s, SOL_SOCKET, SO_BROADCAST,
               (const char *)&broadcast, sizeof(broadcast));

    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((u_short)_udp_port);
    addr.sin_addr.s_addr = INADDR_BROADCAST;

    char buf[256];
    snprintf(buf, sizeof(buf), "EDGE_DISCOVER|%s|%d",
             _node_id.c_str(), _tcp_port);

    sendto(s, buf, (int)strlen(buf), 0,
           (const sockaddr *)&addr, sizeof(addr));
    socket_close(s);
}

void P2PMesh::udp_listen_loop() {
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) return;

    int reuse = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR,
               (const char *)&reuse, sizeof(reuse));

    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((u_short)_udp_port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(s, (const sockaddr *)&addr, sizeof(addr)) != 0) {
        socket_close(s);
        return;
    }
    set_nonblocking(s);

    char buf[2048];
    auto last_broadcast = std::chrono::steady_clock::now();

    while (_running.load()) {
        // Periodic presence broadcast (every 5 seconds)
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                now - last_broadcast).count() >= 5) {
            udp_broadcast_presence();
            last_broadcast = now;
        }

        // Check for incoming packets (with timeout)
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(s, &fds);
        timeval tv = {0, 500000};  // 500 ms
        int ret = select((int)(s + 1), &fds, nullptr, nullptr, &tv);
        if (ret <= 0) continue;

        sockaddr_in from = {};
        socklen_t from_len = sizeof(from);
        int n = recvfrom(s, buf, sizeof(buf) - 1, 0,
                         (sockaddr *)&from, &from_len);
        if (n <= 0) continue;
        buf[n] = '\0';

        std::string data(buf, n);

        // Parse discovery packet: "EDGE_DISCOVER|<node_id>|<tcp_port>"
        if (data.find("EDGE_DISCOVER|") == 0) {
            size_t p1 = data.find('|');
            size_t p2 = data.find('|', p1 + 1);
            if (p1 != std::string::npos && p2 != std::string::npos) {
                std::string peer_id = data.substr(p1 + 1, p2 - p1 - 1);
                int peer_tcp = atoi(data.substr(p2 + 1).c_str());
                char ip_str[64];
                inet_ntop(AF_INET, &from.sin_addr, ip_str, sizeof(ip_str));

                PeerNode p;
                p.node_id      = peer_id;
                p.ip_address   = ip_str;
                p.udp_port     = _udp_port;
                p.tcp_port     = peer_tcp;
                p.online       = true;
                p.last_seen_ms = std::chrono::duration_cast<
                    std::chrono::milliseconds>(now.time_since_epoch()).count();
                add_peer(p);
            }
        }
        // Perception broadcast: JSON with "report_id" field
        else if (data.find("{\"report_id\"") == 0 ||
                 data.find("{ \"report_id\"") == 0) {
            try {
                PerceptionReport r = parse_report(data);
                if (r.node_id != _node_id && on_perception) {
                    on_perception(r);
                }
            } catch (...) {}
        }
    }
    socket_close(s);
}

// ═══════════════════════════════════════════════════════════════
// TCP — reliable perception / decision transport
// ═══════════════════════════════════════════════════════════════

void P2PMesh::tcp_listen_loop() {
    SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock == INVALID_SOCKET) return;

    int reuse = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR,
               (const char *)&reuse, sizeof(reuse));

    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((u_short)_tcp_port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(listen_sock, (const sockaddr *)&addr, sizeof(addr)) != 0) {
        socket_close(listen_sock);
        return;
    }
    if (listen(listen_sock, 8) != 0) {
        socket_close(listen_sock);
        return;
    }
    set_nonblocking(listen_sock);

    while (_running.load()) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(listen_sock, &fds);
        timeval tv = {0, 200000};  // 200 ms
        int ret = select((int)(listen_sock + 1), &fds, nullptr, nullptr, &tv);
        if (ret <= 0) continue;

        sockaddr_in client = {};
        socklen_t client_len = sizeof(client);
        SOCKET client_sock = accept(listen_sock, (sockaddr *)&client, &client_len);
        if (client_sock == INVALID_SOCKET) continue;

        handle_peer_connection(client_sock);
        socket_close(client_sock);
    }
    socket_close(listen_sock);
}

void P2PMesh::handle_peer_connection(int client_fd) {
    // Set receive timeout to prevent indefinite blocking
#ifdef _WIN32
    DWORD timeout_ms = 5000;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO,
               (const char *)&timeout_ms, sizeof(timeout_ms));
#else
    struct timeval tv = {5, 0};
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    // Read until connection closes or newline delimiter
    std::string buffer;
    char buf[4096];
    int total_read = 0;

    while (total_read < 65536 && _running.load()) {  // 64 KB max
        int n = recv(client_fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;
        buf[n] = '\0';
        buffer.append(buf, n);
        total_read += n;

        // Check for complete JSON (newline-delimited)
        if (buffer.find('\n') != std::string::npos) break;
    }

    if (buffer.empty()) return;

    // Remove trailing newline
    if (buffer.back() == '\n') buffer.pop_back();

    try {
        auto j = nlohmann::json::parse(buffer);

        // Perception report
        if (j.contains("report_id")) {
            PerceptionReport r = parse_report(buffer);
            if (on_perception) on_perception(r);
        }
        // Decision intent
        else if (j.contains("intent_id")) {
            DecisionIntent di;
            di.intent_id       = j.value("intent_id", "");
            di.node_id         = j.value("node_id", "");
            di.target_id       = j.value("target_id", "");
            di.proposed_action = j.value("proposed_action", "");
            di.confidence      = j.value("confidence", 0.0f);
            di.rationale       = j.value("rationale", "");
            di.timestamp_ms    = j.value("timestamp_ms", 0);
            if (on_decision_intent) on_decision_intent(di);
        }
    } catch (...) {
        // Ignore malformed messages
    }
}

// ═══════════════════════════════════════════════════════════════
// Send methods
// ═══════════════════════════════════════════════════════════════

bool P2PMesh::send_tcp(const std::string & ip, int port,
                        const std::string & payload) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return false;

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((u_short)port);
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

    // Short timeout for connect
    set_nonblocking(s);
    int ret = connect(s, (const sockaddr *)&addr, sizeof(addr));
    if (ret == SOCKET_ERROR && !socket_would_block()) {
        socket_close(s);
        return false;
    }

    // Wait for connection with timeout
    fd_set wfds;
    FD_ZERO(&wfds);
    FD_SET(s, &wfds);
    timeval tv = {2, 0};  // 2 second timeout
    ret = select((int)(s + 1), nullptr, &wfds, nullptr, &tv);
    if (ret <= 0) { socket_close(s); return false; }

    // Send payload + newline (loop to handle partial sends)
    std::string data = payload + "\n";
    int total_sent = 0;
    int remaining = (int)data.size();
    while (remaining > 0) {
        int n = send(s, data.c_str() + total_sent, remaining, 0);
        if (n <= 0) { socket_close(s); return false; }
        total_sent += n;
        remaining  -= n;
    }

    socket_close(s);
    return true;
}

void P2PMesh::broadcast_perception(const PerceptionReport & report) {
    std::string json = serialize_report(report);

    // UDP broadcast (best-effort, fast)
    {
        SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (s != INVALID_SOCKET) {
            int broadcast = 1;
            setsockopt(s, SOL_SOCKET, SO_BROADCAST,
                       (const char *)&broadcast, sizeof(broadcast));
            sockaddr_in addr = {};
            addr.sin_family      = AF_INET;
            addr.sin_port        = htons((u_short)_udp_port);
            addr.sin_addr.s_addr = INADDR_BROADCAST;
            sendto(s, json.c_str(), (int)json.size(), 0,
                   (const sockaddr *)&addr, sizeof(addr));
            socket_close(s);
        }
    }

    // Also TCP to each known peer for reliability
    std::vector<PeerNode> peers;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto & [id, p] : _peers) {
            if (p.online) peers.push_back(p);
        }
    }
    for (const auto & p : peers) {
        send_tcp(p.ip_address, p.tcp_port, json);
    }
}

void P2PMesh::send_decision_intent(const std::string & peer_id,
                                    const DecisionIntent & intent) {
    std::string json = serialize_intent(intent);

    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _peers.find(peer_id);
    if (it == _peers.end() || !it->second.online) return;

    send_tcp(it->second.ip_address, it->second.tcp_port, json);
}

// ═══════════════════════════════════════════════════════════════
// Serialization
// ═══════════════════════════════════════════════════════════════

std::string P2PMesh::serialize_report(const PerceptionReport & r) const {
    nlohmann::json j;
    j["report_id"]    = r.report_id;
    j["node_id"]      = r.node_id;
    j["timestamp_ms"] = r.timestamp_ms;

    nlohmann::json dets = nlohmann::json::array();
    for (const auto & d : r.detections) {
        nlohmann::json jd;
        jd["target_id"]      = d.target_id;
        jd["target_type"]    = d.target_type;
        jd["classification"] = d.classification;
        jd["confidence"]     = d.confidence;
        jd["decision"]       = d.decision;
        jd["sensor_type"]    = d.sensor_type;
        jd["geo_x"]          = d.geo_x;
        jd["geo_y"]          = d.geo_y;
        jd["bbox"]           = {d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h};
        dets.push_back(jd);
    }
    j["detections"] = dets;
    return j.dump();
}

PerceptionReport P2PMesh::parse_report(const std::string & json_str) const {
    auto j = nlohmann::json::parse(json_str);
    PerceptionReport r;
    r.report_id    = j.value("report_id", "");
    r.node_id      = j.value("node_id", "");
    r.timestamp_ms = j.value("timestamp_ms", 0);

    for (const auto & jd : j.value("detections", nlohmann::json::array())) {
        PerceptionReport::Detection d;
        d.target_id      = jd.value("target_id", "");
        d.target_type    = jd.value("target_type", "");
        d.classification = jd.value("classification", "");
        d.confidence     = jd.value("confidence", 0.0f);
        d.decision       = jd.value("decision", "");
        d.sensor_type    = jd.value("sensor_type", "");
        d.geo_x = jd.value("geo_x", 0.0f);
        d.geo_y = jd.value("geo_y", 0.0f);
        auto b = jd.value("bbox", nlohmann::json::array());
        if (b.size() >= 4) {
            d.bbox = {b[0].get<int>(), b[1].get<int>(),
                      b[2].get<int>(), b[3].get<int>()};
        }
        r.detections.push_back(d);
    }
    return r;
}

std::string P2PMesh::serialize_intent(const DecisionIntent & i) const {
    nlohmann::json j;
    j["intent_id"]       = i.intent_id;
    j["node_id"]         = i.node_id;
    j["target_id"]       = i.target_id;
    j["proposed_action"] = i.proposed_action;
    j["confidence"]      = i.confidence;
    j["rationale"]       = i.rationale;
    j["timestamp_ms"]    = i.timestamp_ms;
    return j.dump();
}
