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

For local end-to-end validation in WSL, use the tracked launcher below. It
starts IQ4_XS as the edge endpoint and the same-source BF16 GGUF as a
**loopback-only cloud simulation**; it is not a substitute for a deployed
cloud endpoint, cloud credentials, or cloud scheduler.

```powershell
wsl.exe -d Ubuntu-22.04 -- bash competition/integration/scripts/start_local_model_services.sh
python validate_model_services.py
python check_setup.py
```

The launcher binds only `127.0.0.1`, uses eight threads and context 512 by
default, writes PID/log files under `integration/outputs/local_services/`, and
refuses to overwrite an occupied 8080 or 8081. `LLAMA_SERVER`, `EDGE_MODEL`,
`CLOUD_MODEL`, `LLAMA_THREADS`, and `LLAMA_CONTEXT_SIZE` can be overridden for
another local layout.

`check_setup.py` validates the bundle, Integration input loader, Python dependencies,
optional metrics sources, and model endpoints. `validate_bundle_runtime.py` is a
bundle/PPO loading smoke check; canonical simulation remains in
`scheduler_training`.

## Post-edge signals

The edge response adapter reads measured or model-returned fields only. For
the current HHK edge-output/v3 contract it preserves these fields separately:

```text
failure_probability, local_prediction, review_state, safety_priority,
safety_trigger, action, model_route, level, category, output_valid, ood
```

`failure_probability` is never converted to `confidence`. A v3 result with
`review_state=required` requests `CLOUD_REVIEW`; a calibrated AI4I result with
`review_state=not_required` can be accepted without a fabricated confidence.
Uncalibrated Qwen output already carries `review_state=required`. Plain text or
invalid structured output is marked invalid and follows the versioned Post-edge
rules. An HHK `escalate_to_cloud` field is recorded only as a legacy hint;
Integration's bundle PostEdgePolicy remains authoritative for queue lifecycle.

If the cloud is measured unreachable, a requested `DIRECT_CLOUD` is safely
overridden to `LOCAL_FIRST`. A Post-edge `CLOUD_REVIEW` is similarly reduced to
`ACCEPT_EDGE`. Both overrides are explicit in trace v2.

## Asset Routing Before Inference

`config/asset_routing_v1.json` is a separate, versioned pre-route policy. It
is intentionally empty until an authoritative asset registry is supplied.
When enabled, exact matches in `designated_cloud_device_ids` or
`designated_cloud_device_types` bypass edge inference and go directly to
cloud. Every non-matching asset is forced to `LOCAL_FIRST`; its calibrated
AI4I `failure_probability` is then evaluated by the normal post-edge review
policy.

Asset importance is neither `safety_priority` nor model confidence. The task
payload cannot self-declare importance, and this policy does not inspect a
model-generated field. Use `--asset-routing-policy <path>` to point the
runtime at a registry export. A designated asset falls back to edge only when
cloud reachability is measured false, and that exception is recorded in trace.

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

## Validation

```powershell
python -m compileall -q .
python -m unittest discover -s tests -p "test_*.py" -q
```

The tests cover 24D ordering, two-action routing, Post-edge collaborative
lifecycle, simulator/runtime rule conformance, EDF, wall-clock timeout, network
safety override, trace v2, profiling, and heartbeat ingestion.
