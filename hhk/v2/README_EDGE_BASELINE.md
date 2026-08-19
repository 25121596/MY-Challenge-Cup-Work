# Edge decision contract and evidence status

Updated: 2026-08-07

This document records what is implemented and what the available measurements
actually cover.  It does not treat an in-process microbenchmark, mock-cloud
admission, or a 30-sample percentile as a production end-to-end SLO.

## Routes and frozen data contracts

There are two separate local routes.  Their measurements and semantics must
not be compared as if they were the same model or the same service boundary.

| Route | Input | Local model work | Output risk semantics |
| --- | --- | --- | --- |
| `ai4i_tabular_moe_v1` | exactly 8 frozen AI4I features | two-expert tabular MoE, then Platt calibration | calibrated `failure_probability` |
| `qwen_text_fallback_v1` | a prompt built from a non-AI4I task record | Qwen3-1.7B IQ4_XS, grammar-constrained one class token | no probability is claimed |

The online AI4I input is
`data/standard_inputs/ai4i_milling_inference_v1.json`.  It has 10,000 records,
contains no ground truth or duplicate `readings` object, and is bound to
`edge-io-protocol/feature_schemas/ai4i_8d_v1.json`.
Its manifest records source, schema, and dataset SHA-256 values.  The labelled
`ai4i_milling_features_inputs.json` is a training-only source and must not be
routed online.

The feature order is fixed:

1. `type_H`, `type_L`, `type_M` (one-hot exactly once)
2. z-scored air temperature in K
3. z-scored process temperature in K
4. z-scored rotational speed in rpm
5. z-scored torque in Nm
6. z-scored tool wear in min

`Ai4iMoeInference` rejects wrong schema IDs, model routes, preprocessing
hashes, non-finite values, non-one-hot type vectors, any feature count other
than eight, duplicate `readings`, and online records containing ground truth.

The refreshed project-task manifest is
`../../benchmark-results/edge_protocol_tasks.manifest.json`. It derives 960
items from all eight canonical types with deterministic 60/20/20 splits and
does not generate a business-priority field. Its companion protocol report is
`../../benchmark-results/protocol_consistency_report.json`: the canonical source
has eight types, the legacy compatibility input schema has six, and
`generic_anomaly` plus `video_anomaly` are intentionally canonical-only.

## Decision semantics

The normative human-readable specifications are
`edge-io-protocol/schemas/output.json` and
`edge-io-protocol/schemas/cloud_escalation.json`.

- There is no unified `confidence` field.
- AI4I performs `p_raw -> Platt scaling -> failure_probability`.
  `local_prediction` is `normal` only when the calibrated probability is at
  most the frozen normal-accept threshold (currently read from the model
  metrics artifact; the rounded protocol value is `0.02751`).
- `threshold_review_margin` is an explicit, versioned operational band around
  that cutoff. The current baseline artifact sets it to `0.0`, so it does not
  alter the frozen threshold. If a later approved policy makes it positive,
  samples within the band keep their normal/fault `local_prediction` but carry
  `review_state=required` and `near_normal_accept_threshold`; the exact margin
  is included in the cloud request and independently checked on admission.
- `suspected` is not a model class.  It is represented by
  `review_state=required` plus one or more `review_reason` values.
- Qwen class tokens are provisional observations.  Until a separately held-out
  calibration is installed, Qwen emits `failure_probability: null`,
  `local_prediction: null`, and `review_state: required`.
- `safety_priority` is safety urgency, not business importance.  Only a hard
  safety signal (`emergency_stop`, `interlock_trip`, `overtemperature_trip`,
  or `pressure_relief_trip`) can produce P0.  A model score or Qwen class can
  never produce P0. A hard signal immediately yields `action=protect` and
  `review_state=required`, even if the AI4I classifier locally predicts normal.
- An AI4I record that explicitly claims the frozen route but fails schema
  validation or local model execution is never downgraded to Qwen. It emits
  `failure_probability=null`, `local_prediction=null`, and exactly one closed
  review reason (`feature_schema_mismatch` or `model_exception`). This is a
  fail-closed availability signal, not a model score.

Example AI4I result:

```json
{
  "local_prediction": "normal",
  "failure_probability": 0.021,
  "calibration_version": "ai4i_platt:<artifact>",
  "review_state": "not_required",
  "safety_priority": "P2",
  "safety_trigger": null,
  "action": "continue"
}
```

Example Qwen fallback result:

```json
{
  "local_prediction": null,
  "failure_probability": null,
  "calibration_version": null,
  "review_state": "required",
  "review_reason": ["qwen_fallback_uncalibrated"],
  "safety_priority": "P1",
  "safety_trigger": null,
  "action": "continue_with_review"
}
```

## Cloud review boundary

`compact_cloud_payload()` produces a bounded, program-generated JSON body with
review/trace/idempotency IDs, timestamps, deadline, route and policy versions,
network state, and an evidence hash.  It deliberately omits `confidence`,
`device_label`, numeric business priority, and GUI placeholders.

The queue sorts by `safety_priority`, then deadline, then acceptance time.
That only schedules review urgency; it does not infer asset criticality,
production loss, or SLA value from AI4I data.

The admission model is closed to unknown fields. In particular, requests with
legacy `confidence`, `device_label`, or business `priority` fields receive HTTP
422 instead of having those fields silently ignored.

### Shadow feedback loop

The edge/cloud integration is prepared as an observation loop. Run the edge
path with `--feedback-log path/to/shadow.jsonl`; every successful or failed
record is appended with the edge raw output, parsed result, `raw_logit` (when
AI4I is used), latency, model/prompt/calibration/policy versions, and the
bounded cloud admission acknowledgement. A cloud response may be joined later
as `cloud_output` without changing the original edge decision.

`edge-io-protocol/feedback_loop.py` computes three separate fields:

- `disagreement`: both edge and cloud have comparable `normal`/`fault` labels
  and they differ;
- `cloud_changed_decision`: the same observable comparison, not a priority;
- `review_useful`: `null` until an explicit human or confirmed outcome label is
  supplied. A cloud disagreement alone is never treated as ground truth.

The cloud endpoint `POST /api/v1/escalations/{review_id}/feedback` accepts the
versioned `edge-cloud-feedback/v1` label record. It is still an in-memory mock;
HTTP 202 means admission only, not cloud inference completion or durable
storage. `tools/summarize_feedback.py` reports comparable, disagreement,
and explicitly labeled review-useful rates while retaining unlabeled rows.

For later calibration, do not discard `raw_logit`. The edge record binds it to
`calibration_version`, `edge_model_version`, `edge_prompt_version`,
`cloud_model_version`, and `routing_policy_version`. Cloud Qwen output is a
teacher/reviewer signal; human or downstream equipment confirmation remains a
separate label source.

`cloud_server` is currently an in-memory admission mock.  HTTP 202 proves only
that this process accepted a request.  It does not prove durable storage,
worker execution, cloud inference, remediation, or recovery after restart.

## Latency evidence and limits

`latency_ms` in an edge result means only the local decision path.  Network
transfer, cloud queue wait, cloud inference, and cloud response are separately
reported or unavailable; none is silently counted as zero.

The historical artifact
`../../benchmark-results/e2e-one-token-30/edge_e2e_summary.json` recorded a
pre-v3-contract Qwen run with IQ4_XS, 16 threads, mmap off, one generated
token, a 393-token fixed prompt, and 30 warmed samples: mean 72.68 ms, P95
107.63 ms, and P99 155.36 ms.  It excludes cloud work and has only 30 samples;
it is therefore a fixed-prompt warm reference, not a general P99 or production
end-to-end claim.  Re-run it after any protocol, model, build, CPU policy, or
input change.

After the v3 contract change, a fresh 5-warmup/30-measurement run is stored in
`../../benchmark-results/e2e-v3-qwen/edge_e2e_summary.json`: IQ4_XS, 16 threads,
mmap off, UCI hydraulic record index 1, **506 actual prompt tokens**, and one
generated grammar token. The local completed-decision path measured mean
**56.91 ms**, P95 **61.54 ms**, and P99 **64.21 ms**. This is still a
30-sample warm estimate and excludes HTTP upload, cloud queueing, cloud
inference, and cloud response.

A separate long-completion load is stored in
`../../benchmark-results/e2e-300in-10out/edge_e2e_summary.json`: the same IQ4_XS
CPU configuration, one fixed UCI hydraulic record rendered with the
`quality_inspection` prompt, server-tokenized to **exactly 300 input tokens**,
then run with grammar disabled and `n_predict=10`. Five warmups and 30 measured
samples produced exactly 10 tokens in every measured response. The local
completed-path mean was **348.97 ms** (P50 **347.70 ms**, P95 **368.48 ms**, P99
**375.09 ms**); prompt construction averaged 6.53 ms, model HTTP/inference
round-trip 342.31 ms, and parsing/output assembly 0.12 ms. The instruction and
output are intentionally non-semantic latency load (for example, `x x ...`),
not a valid class decision. It therefore proves neither a 10-token production
decision contract nor a 200-ms end-to-end SLA.

The historical AI4I artifact
`../../benchmark-results/ai4i_moe_e2e.json` was an in-process JSON
deserialize -> model -> JSON serialize microbenchmark.  It excluded GUI, IPC,
network, cloud queueing, and cloud inference.  After contract changes it must
also be regenerated before quoting its byte size or percentiles.

For a defensible latency report, use at least 1,000 warm samples for P99 and
report, separately: model-only, local completed-decision path, process-cold,
network/upload, queue wait, cloud inference, and failure/timeout rate.

## Verification and reproduction

From WSL in `competition/hhk`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-audit/bin/python -m pytest -q \
  cloud_server/tests/test_escalations.py tests/test_decision_contract.py

.venv-audit/bin/python data/validate_deployment.py
.venv-audit/bin/python edge_run.py \
  --data data/standard_inputs/ai4i_milling_inference_v1.json --max-samples 100

.venv-audit/bin/python ../../benchmark/measure_ai4i_moe_e2e.py \
  --warmups 100 --repetitions 1000
```

For Qwen, the benchmark must use a non-AI4I text fallback input and retain the
same chat template, grammar, system prompt, actual token count, `--n-predict 1`,
thread count, affinity, mmap mode, and CPU policy in every sample.  The cloud
endpoint remains unmeasured until an authenticated, durable review service is
configured.

```bash
python3 ../../benchmark/run_edge_e2e.py \
  --model iq4_xs --threads 16 --mmap off --ctx-size 512 \
  --input data/standard_inputs/uci_hydraulic_inputs.json --input-index 1 \
  --warmups 5 --repetitions 1000 --results ../../benchmark-results/e2e-v3-qwen
```

## GUI status

The native sensor-Qwen pipeline now displays review state and safety urgency;
it no longer derives `normal/suspected/fault`, P0/P1/P2, or a numeric
confidence from a class token.  It has not yet been connected to the Python
AI4I calibrated output path, so the GUI must not claim it displays a live
calibrated `failure_probability`. The changed GUI translation units pass
`g++ -fsyntax-only` in WSL; a graphical runtime smoke test still requires a
desktop-capable build and is not claimed here.
