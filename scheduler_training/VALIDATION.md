# Scheduler Training Validation Report

最近一次既有验证日期：2026-08-20。多场景阶段2代码变更日期：2026-08-22。

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

## 阶段2新增验证入口

阶段2增加了版本化 normal/burst/weak-network 场景、跨场景 episode 采样和
四策略消融。旧默认参数、24D observation、两动作、deadline、reward 与 PPO
超参数没有改动。

在配置完整的 Anaconda 环境、`scheduler_training` 目录下依次运行：

```powershell
python validate.py
python evaluate.py --standard-scenarios --skip-ppo --steps 200 --seeds 42 43
python train_ppo.py --standard-scenarios --timesteps 5120 --episode-steps 100 --eval-episodes 3 --output models\checkpoints\scenario_smoke\pre_route_ppo --release-dir models\releases\scenario_smoke --results-prefix scenario_smoke --policy-version pre-route-scenario-smoke-v1
python validate_policy_bundle.py models\releases\scenario_smoke\policy_bundle.json --load-ppo
python evaluate.py --standard-scenarios --ppo-model models\releases\scenario_smoke\pre_route_ppo.zip --steps 200 --seeds 42 43
```

检查重点：

- `EDGE_ONLY` 的 `direct_cloud_ratio`、`cloud_review_ratio`、`upload_mb` 必须为0；
- `POST_EDGE_ONLY` 的 `direct_cloud_ratio` 必须为0；
- 汇总 CSV 同时包含均值和 `_std` 字段；
- `*_vs_edge_only.csv` 以相同场景、相同队列策略比较完成率、超时率、P95和上传量；
- source 字段必须明确这些结果来自 scenario simulation，而非真实测量；
- `average_expected_accuracy` 只能称为配置期望准确率先验，不能称为真实 Accuracy/F1。

### 2026-08-22阶段2实际验证

已使用现有 `cloudedge` Anaconda环境完成：

- `python -m compileall -q scheduler_training`：通过；
- `python validate.py`：22项单元测试通过，包括同seed跨策略工作负载一致性；
- 三场景 `EDGE_ONLY/POST_EDGE_ONLY` 200-step、2-seed评测：通过；
- 读取现有24D/两动作PPO的三场景四策略80-step smoke：通过；
- 跨场景512-step最小训练、1 episode评估、完整bundle发布：通过；
- 新bundle `--load-ppo` 校验：通过，calibration status为
  `versioned_experiment_scenarios_not_real_calibrated`。

512-step训练只证明代码路径可执行，不是正式实验结果。尚未执行5120-step smoke、
100000-step正式训练或五种子2000-step正式评估。

### 2026-08-22中文图表验证

已在现有 `cloudedge` Anaconda环境安装项目新增的 Matplotlib 依赖，并完成：

- `python plot_experiment_results.py`：从 `scenario_formal_*` 正式结果生成7张
  300 DPI中文PNG图和 `output/chart_manifest.json`；
- 逐张视觉检查标题、坐标轴、图例、数值标注和中文字体，无遮挡；
- 全量上云通信基准由同批 `EDGE_ONLY` 已完成任务的 payload 总量派生，图例
  明确标为全量上云基准；
- `python validate.py`：26项单元测试通过，其中4项覆盖绘图输入校验、通信量
  派生、任务类型加权及七图实际渲染；
- `python -m compileall -q .`：通过。

绘图入口只读取训练/评测结果，没有修改24D observation、两动作映射、奖励、
deadline、PPO超参数或场景仿真参数。输出目录内容已由项目 `.gitignore` 忽略。
