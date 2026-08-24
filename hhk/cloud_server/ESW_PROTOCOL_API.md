# ESW-v1 Cloud API

`POST /api/v1/protocol/esw` admits and decodes one ESW-v1 control-plane
payload. It does not replace `POST /api/v1/escalations`, which receives the
complete `edge-cloud-review/v3` JSON audit record and owns review admission.

The endpoint imports the canonical decoder from
`../edge-io-protocol/esw_v1.py`; the cloud service does not maintain a second
copy of the bit layout or validation rules.

## Envelope

```json
{
  "schema_version": "edge-esw-envelope/v1",
  "device_id": "edge-01",
  "request_id": "request-123",
  "trace_id": "trace-123",
  "audit_review_id": "review-123",
  "esw_payload_b64": "AQ2rgA=="
}
```

`esw_payload_b64` is only a JSON transport representation. Its decoded value
must be exactly four ESW wire octets; `AQ2rgA==` decodes to `01 0D AB 80`.
The envelope supplies identity and correlation because ESW itself contains no
device, request, or trace identifiers.

## Response and routing

The response returns `ESW_OK`, `ESW_INVALID`, or `ESW_UNSUPPORTED`, raw word
and rejection reason when available, decoded fields for valid packets, and a
safe `cloud_action`.

- Valid `accept_local`, `direct_cloud`, and `cloud_review` actions are passed
  through.
- A valid `unknown` action returns `ESW_OK` but maps safely to
  `cloud_review` with `policy_reason=action_unknown`.
- Invalid, unsupported, and wrong-length payloads map safely to
  `cloud_review`, preserving the decoder reason.

The endpoint does not enqueue a review or invoke a model. The paired full JSON
record must be submitted separately to `/api/v1/escalations`; a 4-byte hint
must never be mistaken for a durable audit record.

## Transport requirement

The endpoint parses protocol semantics only. It has no client-supplied flag
that can prove authentication. Production deployment must use an
integrity-protected, authenticated transport or trusted proxy that binds the
envelope identity to its sender.
