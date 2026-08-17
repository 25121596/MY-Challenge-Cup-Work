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

## 4. 正式训练

确认 smoke training 正常后，再使用更大的 `--timesteps`。

> `slot_duration=0.05` 在 Phase4.5 仍用于旧 Bernoulli arrival 参数解释和日志兼容；任务服务生命周期本身已经是 continuous-event，不再以 50ms slot 递减。

## 5. 验证发布策略包

```powershell
python validate_policy_bundle.py models/releases/pre_route_v2/policy_bundle.json --load-ppo
```

旧25D/Discrete(3) PPO不兼容，不作为当前版本兼容校验对象。
