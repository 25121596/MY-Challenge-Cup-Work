# Real Trace Schema v2

Schema identifier: `cloud-edge-real-trace/2.0`.

v2 adds explicit two-stage scheduling fields:

* bundle, PPO policy, and Post-edge rule versions;
* requested/effective `LOCAL_FIRST` or `DIRECT_CLOUD` action;
* final `EDGE`, `CLOUD`, or `COLLABORATIVE` route;
* Post-edge policy decision, effective decision, rule ID, reason, feasibility,
  threshold, and recommendation;
* measured legacy confidence signals or HHK V2 failure-probability,
  model-route, calibration, feature-schema, local-prediction and producer-review signals;
* runtime safety override;
* ordered `observation_24d`.

Stage timings, resource snapshots, network observations, model identity, and
available llama.cpp profile fields remain available. Missing fields remain
null. This document describes the only active export schema.

For `post-edge-rules/2.0`, the trace also records the effective
failure-probability threshold, evidence source, producer review state, and
producer-rule consistency. Legacy confidence is not synthesized from HHK V2
failure probability.
