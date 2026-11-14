# ESW-v1 Cloud Decoder Guide

This document is for cloud services that receive the ESW-v1 real-time control
word. ESW is a compact scheduling hint. It does not replace the corresponding
`edge-output/v3` JSON record, which remains the source for audit, training,
device identity, evidence, timestamps, and detailed diagnostics.

## Wire format

Each ESW packet is exactly 4 octets representing an unsigned 32-bit integer
in network byte order (big-endian). Bit 0 is the least-significant bit of that
integer.

For example, the bytes `01 05 47 05` decode to `0x01054705`.

```text
word = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]
```

Before acting on a word, validate its length, schema version, reserved bits,
field enums, and cross-field invariants. An invalid or unsupported word must
be routed to cloud review. Do not infer sender intent from an invalid packet.

## Bit layout and field codes

| Bits | Field | Codes |
| --- | --- | --- |
| 0..7 | `risk_q8` | 0..255 |
| 8 | `risk_valid` | 0 false, 1 true |
| 9 | `calibrated` | 0 false, 1 true |
| 10..11 | `local_prediction` | 0 unknown, 1 normal, 2 fault, 3 reserved |
| 12..13 | `review` | 0 none, 1 recommended, 2 required, 3 reserved |
| 14..15 | `safety` | 0 unknown, 1 P2, 2 P1, 3 P0 |
| 16..17 | `edge_availability` | 0 unknown, 1 available, 2 unavailable, 3 degraded |
| 18..19 | `next_action` | 0 unknown, 1 accept_local, 2 direct_cloud, 3 cloud_review |
| 20..23 | reserved_v1 | Must be 0 |
| 24..27 | `schema_version` | Must be 1 for this protocol version |
| 28..31 | reserved | Must be 0 |

Extract a field as `field = (word >> shift) & mask`. The sender always emits
schema version 1 and zeros both reserved regions.

## Risk semantics

When `risk_valid=1`, the cloud can recover the calibrated AI4I failure
probability as:

```text
failure_probability = risk_q8 / 255.0
```

The sender quantizes with round-half-up:

```text
risk_q8 = floor(clamp(failure_probability, 0.0, 1.0) * 255.0 + 0.5)
```

This value is a model risk probability, not a generic confidence or a
business priority. The maximum wire quantization error is half a Q8 step.

For an uncalibrated Qwen fallback, the following is mandatory:

```text
risk_q8 = 0
risk_valid = 0
calibrated = 0
```

The cloud must treat this as "no probability supplied", never as zero failure
risk. Categorical fields may still state the local observation and that cloud
review is required.

## Action semantics

`next_action` describes what to do after the state represented by this word:

- `accept_local`: the cloud need not review this result solely because of ESW.
- `direct_cloud`: the scheduler selected a cloud path and edge inference was
  skipped.
- `cloud_review`: enqueue or execute cloud review according to the cloud SLA.
- `unknown`: no action meaning was supplied; use the fail-closed path.

`unknown` is a valid enum value, distinct from `ESW_INVALID`. Retain a valid
decode status and record `ACTION_UNKNOWN` (or the receiver's equivalent
metric) before applying the same safe `cloud_review` routing fallback.

The PPO pre-route choice `LOCAL_FIRST` is intentionally absent from ESW. It
exists in the full JSON trace as `pre_route_action` and related requested /
effective routing fields. It must not be reconstructed from `next_action`.

`safety` is safety urgency rather than business importance. In particular,
`P0` is issued only from a hard safety trigger and never from a high model
probability alone. Asset criticality, production impact, and SLA order remain
cloud policy inputs outside this word.

## Required fail-closed rules

Treat the word as invalid and route it to cloud review when any of the
following applies:

- payload length is not 4 bytes;
- schema version is not 1 (`ESW_UNSUPPORTED`);
- bits 20..23 or 28..31 are non-zero;
- `local_prediction=3` or `review=3`;
- `risk_valid=0` with `risk_q8 != 0`;
- `risk_valid` and `calibrated` have different values;
- `risk_valid=1` with a local prediction other than `normal` or `fault`;
- `review=required` with a next action other than `cloud_review`;
- `next_action=accept_local` with an unavailable edge or unknown prediction;
- `next_action=direct_cloud` with any risk, calibration, local prediction, or
  review result present;
- `safety=P0` with `next_action=accept_local`;
- `edge_availability=unavailable` with a non-unknown local prediction or a
  claimed valid risk.

The receiver should retain the raw four bytes, decoded word, validation
status, and rejection reason alongside the JSON trace for diagnostics.

## Envelope and trust boundary

The ESW payload has no device ID, request ID, trace ID, CRC, MAC, or
signature. It is exactly 4 octets, but the transport message must include an
authenticated, integrity-protected envelope binding those identity fields to
the payload. TLS with authenticated peers, or an equivalently protected
message envelope, is required for deployment.

Cloud wire validation can establish only the bit-level rules in this document.
It cannot prove that a P0 value came from a hard safety trigger or that a
calibrated risk originated from the AI4I calibration artifact. Verify such
producer requirements from the corresponding `edge-output/v3` audit record.

## Reference packets

| Bytes | Meaning |
| --- | --- |
| `01 05 47 05` | Calibrated AI4I normal result, P2, edge available, accept locally, `risk_q8=5` |
| `01 0D AB 80` | Calibrated AI4I fault, review required, P1, edge available, cloud review, `risk_q8=128` |
| `01 0D A8 00` | Qwen categorical fault / review record; no calibrated probability |
| `01 0D EB FF` | Calibrated AI4I P0 state requiring cloud review, `risk_q8=255` |
| `01 0A 00 00` | Direct cloud path with no edge inference result |

The authoritative machine-readable vectors are in
`test_vectors/esw_v1_vectors.json`. Cloud implementations should test their
encoder and decoder against that file before connecting to edge devices.
