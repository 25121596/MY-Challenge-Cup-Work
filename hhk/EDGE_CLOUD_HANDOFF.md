# Edge-cloud handoff

This handoff is for the first shadow-mode integration. The edge decision stays
authoritative for local safety actions. A cloud Qwen answer is a reviewer or
teacher signal until a human or downstream equipment outcome is available.

## Endpoints

The edge submits the generated `cloud_payload` to:

```text
POST /api/v1/escalations
```

After a review or confirmed outcome, post the feedback record to:

```text
POST /api/v1/escalations/{review_id}/feedback
```

The current server stores both records in memory. HTTP 202 means admission only;
it does not mean that a cloud model ran or that storage is durable.

## Edge run

```bash
python edge_run.py \
  --data data/standard_inputs/uci_hydraulic_inputs.json \
  --max-samples 100 \
  --cloud-escalation-endpoint https://cloud.example/api/v1/escalations \
  --feedback-log results/shadow.jsonl
```

Every row in `results/shadow.jsonl` keeps the edge raw output, parsed result,
local latency, bounded admission acknowledgement, feature hash, and model,
prompt, calibration, and routing-policy versions. Failed rows are retained.

Summarize the log with:

```bash
python tools/summarize_feedback.py results/shadow.jsonl \
  --output results/shadow_summary.json
```

## Feedback semantics

`disagreement` and `cloud_changed_decision` are populated only when both sides
have comparable `normal`/`fault` labels. They remain `null` for an admission
acknowledgement or an unavailable cloud result. `review_useful` remains `null`
unless the payload includes `human_label`, `confirmed_outcome`, or an explicit
`label_source`. Do not convert a cloud answer into ground truth.

The AI4I path preserves `raw_logit` next to the calibrated
`failure_probability`; this allows a future Platt recalibration without
rerunning the model. Qwen fallback keeps probability and local prediction
`null` until an independent held-out calibration exists.

The protocol rejects legacy `confidence`, `priority`, and `device_label` keys.
`safety_priority` is limited to safety urgency: P0 requires a hard safety
trigger, while model disagreement can at most require P1 review.

## Scheduler alignment

The cloud scheduler has a separate pre-route contract: 24D observation and
`LOCAL_FIRST` or `DIRECT_CLOUD`. It does not receive an edge model result yet.
After `LOCAL_FIRST`, Integration's PostEdgePolicy consumes an
`edge-output/v3` envelope as `{"output": <edge result>}`. It keeps
`failure_probability` separate from legacy `confidence`:

- `review_state=required` requests `CLOUD_REVIEW` when the deadline and
  network permit it;
- calibrated AI4I `review_state=not_required` accepts the edge result without
  inventing a confidence value;
- Qwen fallback remains review-required because it is uncalibrated;
- a P0 edge action protects locally immediately. Cloud review is supplementary
  and cannot delay or reverse that action.

The generic `integration/model_client.py` still calls a llama-compatible chat
endpoint and cannot by itself manufacture an `edge-output/v3` envelope. A
deployment must therefore place the existing edge protocol runner/adapter in
front of Integration's edge client; raw Qwen text is retained only as a legacy
fallback and will be treated as invalid or confidence-unavailable by policy.
