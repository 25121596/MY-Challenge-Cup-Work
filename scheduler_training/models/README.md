# 模型目录

请使用 `train_ppo.py` 重新训练24D/Discrete(2) Pre-route PPO。

发布产物默认写入 `models/releases/pre_route_v2/`，并包含
`policy_bundle.json`。旧25D/Discrete(3)模型明确不兼容。

修正前动作三版本的PPO模型未被复制到本目录，因为其训练环境存在传输阶段、超时归类和奖励不一致问题。
