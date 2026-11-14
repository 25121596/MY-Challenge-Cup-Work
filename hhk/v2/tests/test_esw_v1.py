import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge-io-protocol"))

from esw_v1 import (  # noqa: E402
    DecodeStatus,
    EdgeAvailability,
    EswFields,
    LocalPrediction,
    NextAction,
    Review,
    Safety,
    decode_esw,
    decode_esw_bytes,
    dequantize_risk,
    encode_esw,
    encode_esw_bytes,
    quantize_risk,
    qwen_unscored_fields,
)


class EswV1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "edge-io-protocol" / "test_vectors" / "esw_v1_vectors.json"
        cls.vectors = json.loads(path.read_text(encoding="utf-8"))

    def test_valid_vectors_round_trip(self):
        enum_types = {
            "local_prediction": LocalPrediction,
            "review": Review,
            "safety": Safety,
            "edge_availability": EdgeAvailability,
            "next_action": NextAction,
        }
        for vector in self.vectors["valid"]:
            values = dict(vector["fields"])
            for key, enum_type in enum_types.items():
                values[key] = enum_type[values[key]]
            fields = EswFields(**values)
            word = encode_esw(fields)
            self.assertEqual(f"{word:08X}", vector["word_hex"])
            decoded = decode_esw_bytes(encode_esw_bytes(fields))
            self.assertEqual(decoded.status, DecodeStatus.OK)
            self.assertEqual(decoded.fields, fields)

    def test_quantization_is_round_half_up(self):
        for vector in self.vectors["quantization"]:
            q = quantize_risk(vector["probability"])
            self.assertEqual(q, vector["risk_q8"])
            self.assertAlmostEqual(dequantize_risk(q), q / 255.0)

    def test_invalid_vectors_fail_closed(self):
        for vector in self.vectors["invalid"]:
            result = decode_esw(int(vector["word_hex"], 16))
            self.assertEqual(result.status.value, vector["status"])
            self.assertEqual(result.reason, vector["reason"])
            self.assertEqual(result.fields.next_action.name, vector["fallback"])

    def test_packet_length_and_qwen_invariants(self):
        self.assertEqual(decode_esw_bytes(b"\x01").reason, "invalid_packet_length")
        qwen = qwen_unscored_fields(local_prediction=LocalPrediction.FAULT)
        self.assertEqual(encode_esw(qwen) & 0x3FF, 0)
        self.assertFalse(qwen.risk_valid)
        self.assertFalse(qwen.calibrated)


if __name__ == "__main__":
    unittest.main()
