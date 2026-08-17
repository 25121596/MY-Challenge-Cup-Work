# Scheduler Training Pre-route Architecture

> Current contract: 24D observation and `Discrete(2)` with
> `0=LOCAL_FIRST`, `1=DIRECT_CLOUD`. Legacy 25D/Discrete(3) PPO models are
> intentionally incompatible.

## 目标

Phase4.5 将 Phase4 从“参数和动力学写死在 simulator 中的仿真程序”重构成“可替换真实数据动力学的训练框架”。

## 运行链

```text
TaskArrivalModel
      |
      v
Decision Event
      |
      v
SimStateProvider -> SystemState -> build_observation() -> 24D
                                             |
                                             v
                                      Pre-route PPO
                                          0 / 1
                                             |
                                             v
                                  non-preemptive EDF
                                             |
                 +---------------------------+---------------------------+
                 |                           |                           |
          LOCAL_FIRST               DIRECT_CLOUD
                 |                           |                           |
          EDGE_RUNNING               CLOUD_RUNNING
        EDGE_RESULT_READY                    |
                 |                           |
          PostEdgePolicy                     |
          /             \                    |
  ACCEPT_EDGE       CLOUD_REVIEW              |
       |                 |                    |
       |           TRANSMITTING               |
       |                 |                    |
       |       CLOUD_REVIEW_WAITING           |
       |                 |                    |
       |       CLOUD_REVIEW_RUNNING           |
       +-----------------+--------------------+
                                             |
                                             v
                                         StepOutcome
                                             |
                                             v
                                     RewardCalculator
```

## 连续时间事件

仿真器只在以下时刻推进：

- 新任务到达；
- Edge/Cloud service 完成；
- Collaborative transmission ready；
- deadline 到期。

两个事件之间：

- resource dynamics 按 `delta_t_s` 推进；
- network dynamics 按 `delta_t_s` 推进；
- active task 不使用 slot 递减。

## 四类可替换模型

```text
TaskArrivalModel
  LegacyTaskArrivalModel      <- 当前
  TraceTaskArrivalModel       <- Phase5

ServiceTimeModel
  ProfileServiceTimeModel     <- 当前
  CalibratedServiceTimeModel  <- Phase5

ResourceDynamics
  LegacyResourceDynamics      <- 当前
  Trace/Calibrated dynamics   <- Phase5

NetworkDynamics
  LegacyNetworkDynamics       <- 当前
  TraceNetworkDynamics        <- Phase5
```

## 训练/部署一致性

训练与部署都应保持以下状态语义：

```text
edge_queue
cloud_queue
edge_workload_s
cloud_workload_s
edge_resource
cloud_resource
network_quality
bandwidth_mbps
```

Phase4.5 由 `SimStateProvider` 提供；Integration 由 `RealStateProvider` 提供。
