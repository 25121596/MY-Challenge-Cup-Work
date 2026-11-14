# Active Scheduler Bundle Runtime Architecture

`scheduler_training` owns PPO observation/action design and publishes the
deployable bundle. Integration owns real state, real queues, real execution,
profiling, and trace data.

```text
scheduler_training/train_ppo.py
  -> policy_bundle.json
     + 24D schema
     + Discrete(2) mapping
     + PPO zip
     + Post-edge JSON

Integration PolicyBundleLoader
  -> RealPPOPolicy
  -> RealPPORuntime + RealStateProvider
  -> RealTaskManager
     + edge EDF worker
     + transfer EDF worker
     + cloud EDF worker
     + wall-clock deadline monitor
  -> trace schema v2
```

The bundle contains policy data and schemas. Integration implements the pinned
`pre-route-observation/2.0` normalization and `post-edge-rules/1.0` evaluator.
Cross-module tests compare the Integration evaluator with Scheduler for the
same rule configuration and inputs.

Integration owns only its real-runtime task model, task-input loader, and
legacy profile priors used to seed estimates. Canonical simulation, training,
distillation, observation design, and bundle publication remain solely in
`scheduler_training`. Old 25D/Discrete(3) policies cannot be loaded.
