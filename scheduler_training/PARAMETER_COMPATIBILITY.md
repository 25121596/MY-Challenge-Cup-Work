# Phase4.5 Parameter Compatibility

Phase4.5 的目标是先改“接口与动力学结构”，不做真实参数校准。

| 项目 | Phase4 v1.8.5 | Phase4.5 | 状态 |
|---|---:|---:|---|
| Observation 维度 | 25 | 24 | 有意变更：删除pre-route不可用的task_confidence |
| PPO action | 3 | 2 | 有意变更：LOCAL_FIRST / DIRECT_CLOUD |
| `arrival_probability` 默认值 | 0.10 | 0.10 | 不变 |
| `slot_duration` 参考值 | 0.05s | 0.05s | 数值不变，语义改为到达分布/兼容参考 |
| Edge recovery/s | 0.045 | 0.045 | 不变 |
| Cloud recovery/s | 0.0375 | 0.0375 | 不变 |
| Edge consumption/s | 0.090 | 0.090 | 不变 |
| Cloud consumption/s | 0.075 | 0.075 | 不变 |
| GOOD→BAD / s | 0.10 | 0.10 | 不变 |
| BAD→GOOD / s | 0.30 | 0.30 | 不变 |
| Bandwidth range | 5–50 Mbps | 5–50 Mbps | 不变 |
| Reward latency weight | 2.0 | 2.0 | 不变 |
| Reward queue weight | 0.2 | 0.2 | 不变 |
| Timeout penalty | 20.0 | 20.0 | 不变 |
| Accuracy penalty | 3.0 | 3.0 | 不变 |
| Communication penalty | 0.25 | 0.25 | 不变 |
| Edge/Cloud compute penalty | 0.05 / 0.05 | 0.05 / 0.05 | 不变 |

`task_profiles.py` 在构建 Phase4.5 时与原 Phase4 文件 SHA-256 一致：

```text
7e04b5cde389759d8db2ff68e2c7f049059d041148481cb38399e97b5fc6995f
```

`observation.py` 删除了 `task_confidence`；其余字段保持相对顺序，原归一化数值不变。旧25D/Discrete(3) PPO不兼容，必须重新训练。

## 需要注意的“机制变化”

虽然数值参数不变，以下动力学语义有意变化：

1. service completion 不再量化到 50ms slot；
2. PPO transition 从“决策 + 多个空闲 slot”变为“决策事件 -> 下一决策事件”；
3. resource/network 按真实 `delta_t_s` 推进；
4. EDF 明确采用 absolute deadline 且 non-preemptive；
5. reward 仍使用相同公式/权重，但由 `StepOutcome` 驱动，计分时机与 slot 版不再完全相同。

因此旧 Phase4 PPO 可以做接口兼容测试，但 Phase4.5 正式使用应重新训练。
