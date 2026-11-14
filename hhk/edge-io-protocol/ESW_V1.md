# ESW-v1

ESW-v1 is a fixed 32-bit / 4-octet control-plane projection of an edge
decision. It complements, and never replaces, the complete `edge-output/v3`
JSON audit record. It contains no identifiers, evidence, timestamps, model
versions, or text diagnostics.

The wire order is big-endian. Bit 0 is the least-significant bit of the
decoded unsigned 32-bit word. Senders emit schema version 1 and zero all
reserved bits.

| Bits | Field | Values |
| --- | --- | --- |
| 0..7 | `risk_q8` | Quantized calibrated `P(failure)`: `floor(clamp(p, 0, 1) * 255 + 0.5)` |
| 8 | `risk_valid` | A calibrated probability is present |
| 9 | `calibrated` | The probability passed the versioned calibration artifact |
| 10..11 | `local_prediction` | unknown, normal, fault, reserved |
| 12..13 | `review` | none, recommended, required, reserved |
| 14..15 | `safety` | unknown, P2, P1, P0 |
| 16..17 | `edge_availability` | unknown, available, unavailable, degraded |
| 18..19 | `next_action` | unknown, accept_local, direct_cloud, cloud_review |
| 20..23 | reserved_v1 | Must be zero |
| 24..27 | `schema_version` | Always `1` for ESW-v1 |
| 28..31 | reserved | Must be zero |

The Qwen fallback is uncalibrated. Its canonical ESW representation always
sets `risk_q8=0`, `risk_valid=0`, and `calibrated=0`; it must not fabricate a
probability. `local_prediction`, review, safety, and next action may still
describe the categorical result and required workflow.

`next_action` describes the action after the represented edge state. It does
not encode the PPO pre-route action `LOCAL_FIRST`; that belongs in the JSON
audit trace. `DIRECT_CLOUD` applies only when a scheduler selected direct
cloud and skipped edge inference.

## Required validation

- `risk_valid` and `calibrated` must have the same value.
- `risk_valid=0` requires `risk_q8=0`; `risk_q8=0, risk_valid=1,
  calibrated=1` is valid and means calibrated zero risk.
- `risk_valid=1` requires a `normal` or `fault` local prediction.
- `review=required` requires `next_action=cloud_review`.
- `next_action=accept_local` requires a non-unknown local prediction and an
  edge availability other than `unavailable`.
- `next_action=direct_cloud` requires `risk_q8=0`, `risk_valid=0`,
  `calibrated=0`, `local_prediction=unknown`, and `review=none`. Edge
  availability remains meaningful because direct cloud can be selected for an
  available, degraded, or unavailable edge.
- `safety=P0` cannot use `accept_local`.
- `edge_availability=unavailable` requires `local_prediction=unknown` and
  `risk_valid=0`.
- Reserved enum values and non-zero reserved bits are invalid.

A decoder receiving a packet whose size is not 4 bytes, whose version is
unknown, whose reserved bits are non-zero, or whose field combination is
invalid must not infer intended semantics. It reports `ESW_INVALID` or
`ESW_UNSUPPORTED` and supplies `cloud_review` as the safe fallback.

The `unknown` enum values are valid sender statements, not malformed packets.
They retain `ESW_OK` status but downstream policy should fail closed where an
unknown action or result cannot be safely accepted.

## Trust boundary

Wire validation can verify only the 32-bit layout and the rules above. It
cannot prove provenance. Producers must ensure that P0 originates from a hard
safety trigger, calibrated risk originates from the frozen AI4I calibration,
and Qwen fallback never claims risk. Audit those claims against the paired
`edge-output/v3` JSON record.

ESW has no identity, CRC, MAC, or signature. The 4-octet payload must travel
inside an authenticated, integrity-protected envelope that binds it to the
device, request, and trace identity. "ESW payload is 4 octets" does not mean
the complete network message is 4 octets.
