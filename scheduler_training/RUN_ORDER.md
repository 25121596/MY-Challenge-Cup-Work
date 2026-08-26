# Scheduler Training 推荐运行顺序

## 1. 统一验证

```powershell
python validate.py
```

该入口会同时执行当前的调度仿真、edge input、Post-edge、bundle和蒸馏管线单元测试。
也可以直接运行：

```powershell
python -m unittest discover -s tests -p "test*.py" -v
```

## 2. 安装训练依赖

```powershell
pip install -r requirements.txt
```

## 3. 最小 PPO smoke training

```powershell
python train_ppo.py --timesteps 2048 --episode-steps 50 --eval-episodes 2
```

## 4. 标准三场景 smoke training

从 `scheduler_training` 目录运行；每个 episode 按配置权重选择 normal、burst
或 weak-network 场景，场景编号不进入24D observation：

```powershell
python train_ppo.py --standard-scenarios --timesteps 5120 --episode-steps 100 --eval-episodes 3 --output models\checkpoints\scenario_smoke\pre_route_ppo --release-dir models\releases\scenario_smoke --results-prefix scenario_smoke --policy-version pre-route-scenario-smoke-v1
```

先验证 smoke bundle：

```powershell
python validate_policy_bundle.py models\releases\scenario_smoke\policy_bundle.json --load-ppo
```

## 5. 三场景四策略 smoke evaluation

```powershell
python evaluate.py --standard-scenarios --ppo-model models\releases\scenario_smoke\pre_route_ppo.zip --steps 200 --seeds 42 43
```

输出包含四种策略：`EDGE_ONLY`、`PRE_ROUTE_ONLY`、`POST_EDGE_ONLY`、
`TWO_LAYER`。其中 `EDGE_ONLY` 强制接受边端结果，不会调用云端。

## 6. 正式跨场景训练和五种子评估

smoke 全部通过后运行：

```powershell
python train_ppo.py --standard-scenarios --timesteps 100000 --episode-steps 300 --eval-episodes 20 --output models\checkpoints\scenario_formal\pre_route_ppo --release-dir models\releases\scenario_formal --results-prefix scenario_formal --policy-version pre-route-scenario-formal-v1
```

```powershell
python evaluate.py --standard-scenarios --ppo-model models\releases\scenario_formal\pre_route_ppo.zip --steps 2000 --seeds 42 43 44 45 46
```

## 7. 生成中文实验图表

正式评估完成后，从 `scheduler_training` 目录运行：

```powershell
python plot_experiment_results.py
```

默认读取 `results\scenario_formal_*`，在 `output\` 中生成7张300 DPI中文PNG
图和 `chart_manifest.json`。如需同时用于Word排版的矢量图：

```powershell
python plot_experiment_results.py --formats png svg
```

图中所有调度指标均为场景仿真结果，不是实测系统指标。

第7张图展示 `average_expected_accuracy`，图中明确标为“平均预期准确率（仿真
先验）”。不要把它写成真实 Accuracy/F1；真实 AI4I Accuracy/F1 由
Integration 的云端复核评估提供。

## 8. 检查 HHK V2 AI4I 数据契约

从 `scheduler_training` 目录运行：

```powershell
python inspect_hhk_v2_replay.py --edge-input-path ..\hhk_v2_snapshot\data\standard_inputs\ai4i_milling_inference_v1.json --edge-result-path ..\hhk_v2_snapshot\edge_run_ai4i_v3_results.json
```

## 9. 使用已匹配 AI4I 边端结果训练

```powershell
python train_ppo.py --task-source hhk_v2_replay --edge-input-path ..\hhk_v2_snapshot\data\standard_inputs\ai4i_milling_inference_v1.json --edge-result-path ..\hhk_v2_snapshot\edge_run_ai4i_v3_results.json --post-edge-rules config\post_edge_rules_v2.json --timesteps 20000 --episode-steps 100 --eval-episodes 5 --output models\checkpoints\ai4i_replay\pre_route_ppo --release-dir models\releases\ai4i_replay --policy-version pre-route-ai4i-replay-v1
```

该模式只使用输入/结果按 `task_id` 匹配的任务。现有文件若只有100条结果，不能宣称完成了10000条全量调度训练。

## 10. 旧单场景正式训练

确认 smoke training 正常后，再使用更大的 `--timesteps`。

> `slot_duration=0.05` 在 Phase4.5 仍用于旧 Bernoulli arrival 参数解释和日志兼容；任务服务生命周期本身已经是 continuous-event，不再以 50ms slot 递减。

## 11. 验证发布策略包

```powershell
python validate_policy_bundle.py models/releases/pre_route_v2/policy_bundle.json --load-ppo
```

旧25D/Discrete(3) PPO不兼容，不作为当前版本兼容校验对象。
