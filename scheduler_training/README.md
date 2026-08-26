# Cloud-Edge Scheduler Training

Scheduler Training is the canonical training-side design for the cloud-edge
scheduler. The current version is 4.5.0. It is a real-ready continuous-event
simulator, not a real-calibrated environment.

## Current scheduling contract

Pre-route PPO observation:

```text
24 dimensions
```

`task_confidence` was removed because edge inference confidence does not exist
before the pre-route decision. The remaining field order and numeric
normalization constants are retained.

Pre-route PPO action space:

```text
Discrete(2)
0 = LOCAL_FIRST
1 = DIRECT_CLOUD
```

`COLLABORATIVE` is no longer a pre-route PPO action. It is produced only when a
LOCAL_FIRST task completes its edge stage and `PostEdgePolicy` returns
`CLOUD_REVIEW`:

```text
LOCAL_FIRST -> ACCEPT_EDGE  = final EDGE route
LOCAL_FIRST -> CLOUD_REVIEW = final COLLABORATIVE route
DIRECT_CLOUD                = final CLOUD route
```

Legacy 25D / Discrete(3) PPO models are intentionally incompatible and must not
be loaded as current policies.

## Runtime flow

```text
Task arrival
    |
    v
SimStateProvider -> SystemState -> build_observation(24D)
    |
    v
Pre-route PPO
    |-----------------------------|
    v                             v
LOCAL_FIRST                   DIRECT_CLOUD
    |                             |
EDGE_WAITING                  CLOUD_WAITING
    |                             |
EDGE_RUNNING                  CLOUD_RUNNING
    |                             |
EDGE_RESULT_READY                FINISHED/TIMEOUT
    |
PostEdgePolicy
    |-----------------------------|
    v                             v
ACCEPT_EDGE                  CLOUD_REVIEW
    |                             |
 FINISHED                    TRANSMITTING
                                  |
                         CLOUD_REVIEW_WAITING
                                  |
                         CLOUD_REVIEW_RUNNING
                                  |
                            FINISHED/TIMEOUT
```

Waiting queues remain non-preemptive EDF. Simulation time is continuous seconds;
`remaining_slots` is not used for service completion.

## Versioned scenario experiments

The controlled experiment suite is stored under `config/scenarios/`:

```text
normal_load.json   steady workload and healthy network
edge_burst.json    periodic 15s workload with a 5s burst window
weak_network.json  steady workload under low-bandwidth unstable networking
```

These files instantiate replaceable arrival, resource, network and service-time
models. They do not alter the legacy simulator defaults, task profiles,
deadlines, reward weights, PPO hyperparameters, 24D field order or two-action
mapping. Their numbers are explicit experimental assumptions, not real
measurements or real-system calibration.

One cross-scenario PPO is trained by selecting a scenario at each episode with
the configured weights `40% normal / 40% burst / 20% weak network`. The choice
is seed-reproducible. Scenario identity is deliberately not added to the 24D
observation; the policy must act from the resource, queue, workload and network
state it can observe.

The formal ablation matrix contains four strategies:

```text
EDGE_ONLY       Always LOCAL_FIRST + forced ACCEPT_EDGE
PRE_ROUTE_ONLY  PPO LOCAL_FIRST/DIRECT_CLOUD + forced ACCEPT_EDGE
POST_EDGE_ONLY  Always LOCAL_FIRST + configured Post-edge rules
TWO_LAYER       PPO pre-route + configured Post-edge rules
```

`EDGE_ONLY` is therefore a true no-cloud baseline: it neither routes directly
to cloud nor escalates after edge inference. All strategies retain the original
hard task deadlines; an action does not erase or extend a deadline.

Scenario smoke evaluation without a PPO model:

```powershell
python evaluate.py --standard-scenarios --skip-ppo --steps 200
```

Full four-strategy evaluation uses a trained bundle model and five default
seeds:

```powershell
python evaluate.py --standard-scenarios --ppo-model models\releases\scenario_formal\pre_route_ppo.zip --steps 2000
```

The evaluator writes per-seed details, mean/std summaries, per-task-type rows,
collaborative-task records and deltas relative to `EDGE_ONLY`. Queue and
workload statistics are simulated system metrics. `average_expected_accuracy`
is a configured expected-accuracy prior; it must not be reported as measured
Accuracy or F1. Measured AI4I Accuracy/F1 remains an Integration-side result.
Within one scenario and seed, arrival, task, resource and network random streams
are isolated so all strategies receive the same workload sequence; this makes
the `EDGE_ONLY` deltas paired rather than comparisons of unrelated samples.

## Chinese experiment charts

After the formal training and four-strategy evaluation have produced the
`scenario_formal_*` CSV files, generate report-ready Chinese charts with:

```powershell
python plot_experiment_results.py
```

The command reads the formal summary, per-seed detail, per-task-type detail and
PPO monitor files from `results/`. It writes seven 300-DPI PNG figures plus
`chart_manifest.json` to `output/`: grouped completion-rate bars with seed
standard-deviation error bars, grouped P95-latency bars, final-route stacked
bars, grouped upload-volume bars, a PPO reward curve and a task-type cloud-route
heatmap, plus grouped expected-accuracy-prior bars with seed standard-deviation
error bars. Every visible title, axis, legend and value annotation is Chinese.
SVG/PDF can be requested for document editing, for example
`--formats png svg`.

The charts explicitly label Scheduler metrics as scenario-simulation results.
The expected-accuracy figure labels `average_expected_accuracy` as a configured
simulation prior, never as measured Accuracy/F1. Generated files under `output/`
are ignored by Git.

## Post-edge rules

The current HHK V3-compatible default rules are in:

```text
config/post_edge_rules_v1.json
```

For real HHK `edge-output/v3` integration, `failure_probability` remains a
risk probability and is not converted to confidence. The shared evaluator uses
`review_state=required` to request `CLOUD_REVIEW` and can accept a calibrated
AI4I result with `review_state=not_required`. Legacy simulated rows without a
producer review state continue to use the configured confidence threshold.

## HHK V2 AI4I replay

`hhk_v2_replay` joins an external standard-input JSON and HHK V2 edge-result
JSON by `task_id`. Only matched tasks are replayed. HHK files are read-only and
Scheduler does not import HHK implementation code.

```powershell
python inspect_hhk_v2_replay.py `
  --edge-input-path ..\hhk_v2_snapshot\data\standard_inputs\ai4i_milling_inference_v1.json `
  --edge-result-path ..\hhk_v2_snapshot\edge_run_ai4i_v3_results.json
```

If source timestamps are identical or non-increasing, Scheduler uses
`--edge-input-interval-s` (default 0.05s) and records this fallback in the replay
report. The measured HHK edge latency is used as an edge latency base; cloud,
network, resources, deadlines, accuracy priors and reward remain simulated or
profile-based. This mode is therefore partially evidence-replayed, not a fully
real-calibrated simulator. The pre-route observation remains 24D and never
contains `failure_probability`.

## Policy bundle

`train_ppo.py` publishes a complete release bundle containing:

```text
models/releases/pre_route_v2/
    pre_route_ppo.zip
    observation_schema.json
    action_schema.json
    post_edge_rules.json
    calibration_manifest.json
    policy_bundle.json
```

The manifest records schema versions, action mapping, compatibility status,
artifact hashes, rule version and training metadata. Validate a release with:

```powershell
python validate_policy_bundle.py models/releases/pre_route_v2/policy_bundle.json --load-ppo
```

Integration should later load this manifest through an explicit configurable
path. Integration and HHK are not modified by this Phase4.5 change.

## Key files

```text
observation.py             24D pre-route observation schema
ppo_env.py                 Gymnasium 24D / Discrete(2) wrapper
ppo_core.py                PPO decision-event core
simulator.py               continuous-event lifecycle and PostEdgePolicy call
post_edge_policy.py        ACCEPT_EDGE / CLOUD_REVIEW rule evaluator
config/post_edge_rules_v2.json
hhk_v2_replay.py           read-only AI4I input/result join and replay report
inspect_hhk_v2_replay.py   replay contract inspection CLI
policy_bundle.py           publish and validate policy bundles
validate_policy_bundle.py  bundle validation CLI
train_ppo.py               PPO training and release publishing
dynamics.py                replaceable arrival/service/resource/network models
experiment_scenarios.py    versioned scenario loading and experiment ablations
config/scenarios/          normal/burst/weak-network experiment assumptions
plot_experiment_results.py Chinese report-chart generation from formal results
edf_queue.py               non-preemptive EDF waiting queue
tests/test_scheduler_core.py
                           current simulator, scheduling and bundle tests
tests/test_distillation_pipeline.py
                           current distillation pipeline tests
legacy_reference/          retained legacy Phase4 source reference
```

## Validation

Unified validation without Gymnasium/SB3 training:

```powershell
python validate.py
```

This entry discovers all current unit tests, including edge-input simulation,
Post-edge rules, policy bundles and the distillation data pipeline. Historical
phase-specific validation entry points are no longer part of the active project.

Short training-path smoke test after installing the existing requirements:

```powershell
python train_ppo.py --timesteps 2048 --episode-steps 50 --eval-episodes 2
```

Cross-scenario smoke training:

```powershell
python train_ppo.py --standard-scenarios --timesteps 5120 --episode-steps 100 --eval-episodes 3 --output models\checkpoints\scenario_smoke\pre_route_ppo --release-dir models\releases\scenario_smoke --results-prefix scenario_smoke --policy-version pre-route-scenario-smoke-v1
```

Do not run long PPO training merely to validate a small code change.

## Calibration status

The following default implementations still use legacy/profile parameters:

- `LegacyTaskArrivalModel`
- `ProfileServiceTimeModel`
- `LegacyResourceDynamics`
- `LegacyNetworkDynamics`

Future Integration traces should be introduced through calibrated replacements
for these interfaces. Current simulation values are simulated or estimated, not
real measurements.
