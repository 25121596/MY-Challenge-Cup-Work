# Real Trace Schema v2

Schema identifier: `cloud-edge-real-trace/2.0`.

v2 adds explicit two-stage scheduling fields:

* bundle, PPO policy, and Post-edge rule versions;
* requested/effective `LOCAL_FIRST` or `DIRECT_CLOUD` action;
* final `EDGE`, `CLOUD`, or `COLLABORATIVE` route;
* Post-edge policy decision, effective decision, rule ID, reason, feasibility,
  threshold, and recommendation;
* measured edge output signals, including edge-output/v3 failure probability,
  local prediction, review state, safety urgency, and legacy escalation hint;
* runtime safety override;
* ordered `observation_24d`.

Asset-routing fields record a versioned external registry decision before model
execution. `asset_routing_route=LOCAL_FIRST` means the asset was not
designated and must receive an edge result first. `DIRECT_CLOUD` means an exact
registry match bypassed edge inference. `ppo_action_*` retains the PPO proposal
for analysis; it is not effective when asset routing is enabled. Asset routing
is unrelated to `safety_priority` and never derives from `failure_probability`.

Stage timings, resource snapshots, network observations, model identity, and
available llama.cpp profile fields remain available. Missing fields remain
null. This document describes the only active export schema.
