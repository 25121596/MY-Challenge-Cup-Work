# Cloud-edge Integration

This module is the real-system consumer of policy bundles published by
`scheduler_training`. It owns measured state, non-preemptive EDF queues,
wall-clock lifecycle, model execution, profiling, and trace collection.

The active contract is:

```text
24D pre-route observation
Discrete(2): 0=LOCAL_FIRST, 1=DIRECT_CLOUD
Post-edge: ACCEPT_EDGE / CLOUD_REVIEW
trace schema: cloud-edge-real-trace/2.0
```

Old 25D/Discrete(3) PPO weights are intentionally incompatible. Historical
training/simulation copies and the old three-action bridge have been removed;
canonical simulation and PPO training live only in `scheduler_training`.

## Runtime flow

```text
task arrival
  -> RealStateProvider
  -> build 24D observation
  -> bundle PPO
       0 LOCAL_FIRST
         -> EDGE_WAITING -> EDGE_RUNNING -> EDGE_RESULT_READY
         -> bundle PostEdgePolicy
              ACCEPT_EDGE -> FINISHED
              CLOUD_REVIEW
                -> TRANSMITTING
                -> CLOUD_REVIEW_WAITING
                -> CLOUD_REVIEW_RUNNING
                -> FINISHED / TIMEOUT
       1 DIRECT_CLOUD
         -> CLOUD_WAITING -> CLOUD_RUNNING -> FINISHED / TIMEOUT
```

Waiting queues use non-preemptive EDF. Absolute deadline is arrival wall-clock
monotonic time plus the task's relative deadline. A running model call is not
preempted by a newly arrived task.

## Policy bundle

Integration loads the complete bundle manifest, not a standalone PPO zip. The
loader verifies paths, SHA-256 hashes, schema versions, field order, action
mapping, compatibility declaration, and Post-edge rule version.

Expected artifacts in one release directory:

```text
policy_bundle.json
pre_route_ppo.zip
observation_schema.json
action_schema.json
post_edge_rules.json
```

Default manifest:

```text
../scheduler_training/models/releases/pre_route_v2/policy_bundle.json
```

Override it with `--policy-bundle` or `SCHEDULER_POLICY_BUNDLE`.

## Real entry point

Edge and cloud test endpoints remain configurable and default to:

```text
edge  http://127.0.0.1:8080
cloud http://127.0.0.1:8081
```

Run:

```powershell
python run_runtime.py --policy-bundle <release>/policy_bundle.json
```

`check_setup.py` validates the bundle, Integration input loader, Python dependencies,
optional metrics sources, and model endpoints. `validate_bundle_runtime.py` is a
bundle/PPO loading smoke check; canonical simulation remains in
`scheduler_training`.

## Post-edge signals

The edge response adapter reads measured or model-returned fields only. Current
HHK V2 / `post-edge-rules/2.0` signals are:

```text
failure_probability, threshold_used, threshold_review_margin,
model_route, calibration_version, feature_schema_id,
review_state, local_prediction, safety_priority, safety_trigger
```

Legacy `post-edge-rules/1.0` bundles remain readable through:

```text
confidence, level, category, output_valid, ood
```

Missing probability or confidence is not fabricated, and one is not converted
into the other. Plain text or invalid structured output is marked invalid and
follows the versioned Post-edge rules. An HHK
`escalate_to_cloud` field is recorded only as a legacy hint; Integration's
bundle PostEdgePolicy is authoritative.

If the cloud is measured unreachable, a requested `DIRECT_CLOUD` is safely
overridden to `LOCAL_FIRST`. A Post-edge `CLOUD_REVIEW` is similarly reduced to
`ACCEPT_EDGE`. Both overrides are explicit in trace v2.

## Trace v2

Trace v2 separates:

* requested and effective pre-route action;
* policy version and bundle schema;
* Post-edge policy/effective decision, rule, reason and threshold;
* edge quality signals;
* final physical route (`EDGE`, `CLOUD`, `COLLABORATIVE`);
* runtime safety overrides;
* measured stage timings and available resource/network metrics.

Unavailable TTFT, TPS, token counts, GPU metrics, confidence, or OOD values are
stored as null/unavailable rather than invented.

## AI4I real cloud review

`run_ai4i_cloud_review.py` provides a resumable, deterministic Qwen3-8B shadow
review for the HHK V2 AI4I held-out test records. It joins edge predictions and
cloud prompts by `task_id`, calls the real cloud model only for the Post-edge
review subset by default, and publishes raw cloud results, HHK-compatible
`edge-cloud-feedback/v1` records, machine-readable metrics, and a Markdown
report. Cloud disagreement and label-confirmed correction remain separate.

See [AI4I_CLOUD_REVIEW.md](AI4I_CLOUD_REVIEW.md) for the Anaconda Prompt pilot,
full-run, resume, and report commands.

## Validation

```powershell
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -q
```

The tests cover 24D ordering, two-action routing, Post-edge collaborative
lifecycle, simulator/runtime rule conformance, EDF, wall-clock timeout, network
safety override, trace v2, profiling, and heartbeat ingestion.
