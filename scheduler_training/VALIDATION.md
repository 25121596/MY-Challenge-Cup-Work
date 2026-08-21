# Scheduler Training Validation Report

验证日期：2026-08-20。

## 已通过

### `python validate.py`

- 24D pre-route observation，不包含 `task_confidence`；
- `Discrete(2)`：`LOCAL_FIRST / DIRECT_CLOUD`；
- non-preemptive EDF；
- continuous service time（不使用 `remaining_slots`）；
- `LOCAL_FIRST -> EDGE_RESULT_READY -> ACCEPT_EDGE/CLOUD_REVIEW` 生命周期；
- 仿真器与独立 `PostEdgePolicy` 读取同一份版本化 JSON 规则；
- policy bundle manifest、artifact 哈希和 schema 校验；
- 原 resource/network/reward 数值参数保持不变；
- `edge_input` 样例 96 条加载与事件式 rollout。
- HHK V2 AI4I input/result按task_id联结、覆盖率审计和相同时间戳回放间隔兜底；
- `failure_probability` Post-edge v2方向、阈值和生产端规则一致性检查；
- bundle中的`calibration_manifest.json`及其哈希检查。

### Python unittest

`python -m unittest discover -s tests -p 'test*.py' -v`：15项通过。

其中包括：

- 96条edge input的类型、加载和完整运行检查；
- EDF、24D observation和两动作契约；
- Post-edge规则与仿真生命周期；
- policy bundle发布与哈希校验；
- 教师输出协议、蒸馏数据集拆分、准确率和学生模型注册表检查。

### 兼容回归

- `python evaluate.py --skip-ppo --steps 30`：通过，可区分 pre-route 动作比例与最终 EDGE/CLOUD/COLLABORATIVE 路径比例；
- `python -m compileall -q .`：通过。

历史阶段命名的验证入口已被单元测试取代，不再作为当前运行入口。

## 旧 PPO 兼容性

旧 25D / `Discrete(3)` PPO 与当前契约明确不兼容，必须重新训练。加载器和 bundle 校验器都会拒绝空间不匹配的模型。

## 当前 AI4I 回放边界

工作区现有AI4I标准输入有10000条，HHK V2边端结果有100条；当前可联结回放100条，覆盖率1%。这100条的边端输出、failure probability和边端耗时来自HHK V2结果；网络、云端、资源、deadline、accuracy prior和reward仍为仿真/配置值。

未在本次代码验证中执行长时间PPO训练。可在已有Anaconda环境中执行：

```powershell
python train_ppo.py --timesteps 2048 --episode-steps 50 --eval-episodes 2
python validate_policy_bundle.py models/releases/pre_route_v2/policy_bundle.json --load-ppo
```

训练脚本将一次发布 PPO 模型、24D observation schema、两动作 schema、Post-edge JSON 规则和 `policy_bundle.json`。
