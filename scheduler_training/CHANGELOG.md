# Scheduler Training Changelog

## Pre-route v2接口变更

- `task_profiles.py` 数值内容未修改。
- observation由25维变为24维，删除执行前不可获得的`task_confidence`。
- PPO Gym action mapping变为`0 LOCAL_FIRST / 1 DIRECT_CLOUD`。
- `COLLABORATIVE`不再是PPO动作，由PostEdgePolicy在边端结果产生后触发。
- 旧25D/Discrete(3) PPO明确不兼容。
- resource/network/reward 数值参数未修改。
- PPO 训练超参数未修改。

## 核心重构

1. 新增 `SystemState` 与 `StateProvider`。
2. `simulator.py` 从 service slot 递减改为 continuous-event。
3. 新增独立 `EDFTaskQueue`，明确 non-preemptive EDF。
4. task 时间字段改为连续秒。
5. 协作任务生命周期对齐 Integration v0.3。
6. arrival/service/resource/network 拆成可替换接口。
7. Reward 只消费 `StepOutcome`。
8. `ppo_core.py` 直接推进到下一 decision event，不再循环制造 IDLE slot。
9. 增加 Phase4.5 regression tests 与 edge_input rollout。
10. 增加版本化Post-edge JSON规则和独立规则执行器。
11. 训练脚本发布带schema、规则和哈希的完整policy bundle。

## 不应误解

Phase4.5 不是最终真实环境。Legacy 实现仍然使用原 Phase4 仿真参数，只是以后这些参数/模型可以通过 provider/dynamics 接口被真实 trace 替换。
