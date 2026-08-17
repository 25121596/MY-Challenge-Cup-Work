# Real Trace Schema v2

Schema identifier: `cloud-edge-real-trace/2.0`.

v2 adds explicit two-stage scheduling fields:

* bundle, PPO policy, and Post-edge rule versions;
* requested/effective `LOCAL_FIRST` or `DIRECT_CLOUD` action;
* final `EDGE`, `CLOUD`, or `COLLABORATIVE` route;
* Post-edge policy decision, effective decision, rule ID, reason, feasibility,
  threshold, and recommendation;
* measured edge output signals and legacy escalation hint;
* runtime safety override;
* ordered `observation_24d`.

Stage timings, resource snapshots, network observations, model identity, and
available llama.cpp profile fields remain available. Missing fields remain
null. This document describes the only active export schema.
