# Routing Shadow Phase A

This implementation prepares unbiased data collection for a future learned
edge-to-cloud router. It does not train a router, select a production upload
threshold, or claim any cloud quality gain.

## Boundary

`routing_shadow.py` first excludes structural non-candidates: P0 hard safety,
explicitly unsupported cloud review, an unavailable network observation, a
known offline network, an expired deadline, and a remaining deadline lower
than the measured `minimum_cloud_completion_ms` supplied by the network
collector. P0 local protection remains
immediate; a parallel safety review is not learned-router exploration.

Among the remaining population, existing `review_state=required` records are
logged as `policy_uploaded`. Existing local records are selected by a stable
SHA-256 draw over `seed`, `policy_version`, and `task_id`; this makes the
selection reproducible and records its exact propensity. `upload_probability`
means the assignment probability under the active design: zero for structural
exclusions, one for deterministic baseline uploads, and the configured
exploration probability for eligible local records.

The resulting states are:

- `structurally_ineligible`
- `eligible_not_uploaded`
- `randomized_shadow`
- `policy_uploaded`

The current cloud endpoint is a review-only admission mock. A normal record
selected as `randomized_shadow` is recorded but not POSTed to that endpoint,
because it cannot represent real cloud inference or accept a non-action
shadow request. Once a real cloud dual-run endpoint exists, it must accept and
return an auditable cloud result without changing the local action.

## Collection

Supply a decision-time network observation, then append to a feedback log:

```powershell
Set-Content network.json '{"cloud_reachable":true,"rtt_ms":45}'
python edge_run.py --data data/standard_inputs/ai4i_milling_inference_v1.json `
  --max-samples 100 --feedback-log results/shadow.jsonl `
  --routing-shadow-probability 0.10 --routing-network-json network.json
python tools/summarize_routing_shadow.py results/shadow.jsonl `
  --output results/shadow.manifest.json
```

Every routing selection records a feature-availability timestamp contract.
Training code must use only features with
`available_at_decision_time=true`; future cloud output, outcome labels, and
post-decision rolling latency statistics are prohibited from the live feature
set.

## Next Gate

Before model training, connect a real cloud model in dual-run mode and retain
edge result, cloud result, cloud completion/failure state, and a confirmed
outcome. Only then can `delta_quality = loss(edge, y) - loss(cloud, y)` be
calculated. The frozen test set remains unavailable to calibration and policy
search.
