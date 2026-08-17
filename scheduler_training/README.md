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

## Post-edge rules

The default versioned rules are in:

```text
config/post_edge_rules_v1.json
```

`post_edge_policy.py` is dependency-light and configuration-driven so the same
rule contract can later be consumed by Integration. In Phase4.5, edge output and
confidence remain simulated/profile-based values and must not be described as
real measurements.

## Policy bundle

`train_ppo.py` publishes a complete release bundle containing:

```text
models/releases/pre_route_v2/
    pre_route_ppo.zip
    observation_schema.json
    action_schema.json
    post_edge_rules.json
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
config/post_edge_rules_v1.json
policy_bundle.py           publish and validate policy bundles
validate_policy_bundle.py  bundle validation CLI
train_ppo.py               PPO training and release publishing
dynamics.py                replaceable arrival/service/resource/network models
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
