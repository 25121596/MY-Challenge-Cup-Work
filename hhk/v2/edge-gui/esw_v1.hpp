#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace edge_gui::esw_v1 {

constexpr uint8_t schema_version = 1;

enum class LocalPrediction : uint8_t { UNKNOWN = 0, NORMAL = 1, FAULT = 2, RESERVED = 3 };
enum class Review : uint8_t { NONE = 0, RECOMMENDED = 1, REQUIRED = 2, RESERVED = 3 };
enum class Safety : uint8_t { UNKNOWN = 0, P2 = 1, P1 = 2, P0 = 3 };
enum class EdgeAvailability : uint8_t { UNKNOWN = 0, AVAILABLE = 1, UNAVAILABLE = 2, DEGRADED = 3 };
enum class NextAction : uint8_t { UNKNOWN = 0, ACCEPT_LOCAL = 1, DIRECT_CLOUD = 2, CLOUD_REVIEW = 3 };
enum class DecodeStatus : uint8_t { OK, INVALID, UNSUPPORTED };

struct Fields {
    uint8_t risk_q8 = 0;
    bool risk_valid = false;
    bool calibrated = false;
    LocalPrediction local_prediction = LocalPrediction::UNKNOWN;
    Review review = Review::NONE;
    Safety safety = Safety::UNKNOWN;
    EdgeAvailability edge_availability = EdgeAvailability::UNKNOWN;
    NextAction next_action = NextAction::UNKNOWN;
};

constexpr Fields safe_fallback() {
    Fields fields;
    fields.next_action = NextAction::CLOUD_REVIEW;
    return fields;
}

struct DecodeResult {
    DecodeStatus status = DecodeStatus::INVALID;
    Fields fields = safe_fallback();
    uint32_t raw_word = 0;
    bool has_raw_word = false;
};

constexpr uint8_t quantize_risk(float probability) {
    const float clamped = probability < 0.0f ? 0.0f : (probability > 1.0f ? 1.0f : probability);
    return static_cast<uint8_t>(clamped * 255.0f + 0.5f);
}

constexpr float dequantize_risk(uint8_t risk_q8) {
    return static_cast<float>(risk_q8) / 255.0f;
}

constexpr bool valid(const Fields & fields) {
    if ((!fields.risk_valid && fields.risk_q8 != 0) || fields.risk_valid != fields.calibrated) return false;
    if (fields.local_prediction == LocalPrediction::RESERVED || fields.review == Review::RESERVED) return false;
    if (fields.next_action == NextAction::DIRECT_CLOUD &&
            (fields.risk_q8 != 0 || fields.risk_valid || fields.calibrated ||
             fields.local_prediction != LocalPrediction::UNKNOWN || fields.review != Review::NONE)) return false;
    if (fields.review == Review::REQUIRED && fields.next_action != NextAction::CLOUD_REVIEW) return false;
    if (fields.next_action == NextAction::ACCEPT_LOCAL &&
            (fields.edge_availability == EdgeAvailability::UNAVAILABLE ||
             fields.local_prediction == LocalPrediction::UNKNOWN)) return false;
    if (fields.safety == Safety::P0 && fields.next_action == NextAction::ACCEPT_LOCAL) return false;
    if (fields.edge_availability == EdgeAvailability::UNAVAILABLE &&
            (fields.local_prediction != LocalPrediction::UNKNOWN || fields.risk_valid)) return false;
    if (fields.risk_valid && fields.local_prediction == LocalPrediction::UNKNOWN) return false;
    return true;
}

constexpr bool encode(const Fields & fields, uint32_t & word) {
    if (!valid(fields)) return false;
    word = static_cast<uint32_t>(fields.risk_q8)
        | (static_cast<uint32_t>(fields.risk_valid) << 8)
        | (static_cast<uint32_t>(fields.calibrated) << 9)
        | (static_cast<uint32_t>(fields.local_prediction) << 10)
        | (static_cast<uint32_t>(fields.review) << 12)
        | (static_cast<uint32_t>(fields.safety) << 14)
        | (static_cast<uint32_t>(fields.edge_availability) << 16)
        | (static_cast<uint32_t>(fields.next_action) << 18)
        | (static_cast<uint32_t>(schema_version) << 24);
    return true;
}

constexpr std::array<uint8_t, 4> to_big_endian(uint32_t word) {
    return { static_cast<uint8_t>(word >> 24), static_cast<uint8_t>(word >> 16),
             static_cast<uint8_t>(word >> 8), static_cast<uint8_t>(word) };
}

constexpr uint32_t from_big_endian(const std::array<uint8_t, 4> & payload) {
    return (static_cast<uint32_t>(payload[0]) << 24) | (static_cast<uint32_t>(payload[1]) << 16)
        | (static_cast<uint32_t>(payload[2]) << 8) | payload[3];
}

constexpr DecodeResult decode(uint32_t word) {
    DecodeResult result;
    result.raw_word = word;
    result.has_raw_word = true;
    if (((word >> 24) & 0x0fu) != schema_version) {
        result.status = DecodeStatus::UNSUPPORTED;
        return result;
    }
    if ((word & 0xf0f00000u) != 0) return result;
    const uint8_t local = (word >> 10) & 0x03u;
    const uint8_t review = (word >> 12) & 0x03u;
    if (local == 3 || review == 3) return result;
    Fields fields;
    fields.risk_q8 = word & 0xffu;
    fields.risk_valid = (word >> 8) & 1u;
    fields.calibrated = (word >> 9) & 1u;
    fields.local_prediction = static_cast<LocalPrediction>(local);
    fields.review = static_cast<Review>(review);
    fields.safety = static_cast<Safety>((word >> 14) & 0x03u);
    fields.edge_availability = static_cast<EdgeAvailability>((word >> 16) & 0x03u);
    fields.next_action = static_cast<NextAction>((word >> 18) & 0x03u);
    if (!valid(fields)) return result;
    result.status = DecodeStatus::OK;
    result.fields = fields;
    return result;
}

constexpr DecodeResult decode_bytes(const uint8_t * payload, std::size_t length) {
    if (payload == nullptr || length != 4) return {};
    return decode((static_cast<uint32_t>(payload[0]) << 24) | (static_cast<uint32_t>(payload[1]) << 16)
        | (static_cast<uint32_t>(payload[2]) << 8) | payload[3]);
}

} // namespace edge_gui::esw_v1
