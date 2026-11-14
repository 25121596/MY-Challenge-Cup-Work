#include "esw_v1.hpp"
#include <cassert>

int main() {
    using namespace edge_gui::esw_v1;
    Fields fields;
    fields.risk_q8 = 5;
    fields.risk_valid = true;
    fields.calibrated = true;
    fields.local_prediction = LocalPrediction::NORMAL;
    fields.safety = Safety::P2;
    fields.edge_availability = EdgeAvailability::AVAILABLE;
    fields.next_action = NextAction::ACCEPT_LOCAL;
    uint32_t word = 0;
    assert(encode(fields, word));
    assert(word == 0x01054705u);
    const auto decoded = decode(word);
    assert(decoded.status == DecodeStatus::OK);
    assert(decoded.fields.local_prediction == LocalPrediction::NORMAL);
    const auto bytes = to_big_endian(word);
    assert(from_big_endian(bytes) == word);
    assert(decode_bytes(bytes.data(), bytes.size()).status == DecodeStatus::OK);
    assert(decode_bytes(bytes.data(), 3).status == DecodeStatus::INVALID);
    assert(decode(0x02000000u).status == DecodeStatus::UNSUPPORTED);
    assert(decode(0x01100000u).status == DecodeStatus::INVALID);
    assert(decode(0x010a0000u).status == DecodeStatus::OK);
    assert(decode(0x01000000u).status == DecodeStatus::OK);
    assert(decode(0x01090400u).status == DecodeStatus::INVALID);
    assert(decode(0x01000500u).status == DecodeStatus::INVALID);
    assert(decode(0x01052400u).status == DecodeStatus::INVALID);
    return 0;
}
