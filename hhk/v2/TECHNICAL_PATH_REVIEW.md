# HHK/v2 技术路径核查

本文以 `competition/hhk/v2` 为主要核查对象，描述当前系统已经实现的链路、测试边界和仍未闭合的部分。本文不把仿真结果、mock 云端结果或未校准的模型输出当成生产能力。

## 1. 总体链路

```text
传感器 / CSV / MAT
        |
        v
preprocessing + edge_preprocess
        |
        v
标准化 edge input JSON
        |
        v
边端专家路由
  +----------------------+----------------------+
  |                                             |
  v                                             v
AI4I 表格 MoE                              Qwen3 IQ4_XS
8 个数值特征                               短文本/单 token 决策
  |                                             |
  v                                             v
Platt 校准                                  未校准类别观察
failure_probability                         failure_probability=null
  +----------------------+----------------------+
                         |
                         v
                 edge-output/v3
                         |
                         v
             本地动作 / 云端复核请求
                         |
                         v
                 云端 API 与队列
                         |
                         v
              反馈记录 / shadow 数据
                         |
                         v
           后续 p_improve 路由模型训练
```

当前是**系统级专家结构**，不是 Qwen 内部 token-level MoE。Qwen3-1.7B 本身仍是稠密模型。

## 2. 数据与预处理

当前支持的数据入口包括：

- UCI Hydraulic 液压数据；
- Nature 三相电机数据；
- AI4I 铣削故障数据；
- 通用 CSV/JSON 数据流。

预处理脚本位于 `preprocessing/` 和 `edge_preprocess/`，最终统一为 `edge-io-protocol/schemas/input.json` 所定义的请求格式。AI4I 在线路径固定使用 `ai4i_8d_v1`，输入是 8 个特征；训练集保留标签，在线标准输入不携带 ground truth，避免数据泄漏。

Qwen 路径会移除大特征数组，只保留任务类型、必要读数和压缩后的上下文。这样可以降低 prompt token，但不会改变 AI4I 表格模型的 8 维输入契约。

## 3. 边端模型与决策语义

### 3.1 AI4I 表格专家

`edge-io-protocol/decision_contract.py` 定义了 AI4I 的结果构造：

- 模型路由：`ai4i_tabular_moe_v1`；
- 模型类型：`two_expert_sparse_tabular_moe`；
- 输出 `failure_probability`；
- 使用固定的正常接受阈值，当前约为 `0.0275137`；
- `p > threshold` 时 `local_prediction=fault` 并请求复核；
- 阈值附近可通过 `threshold_review_margin` 强制复核；
- `emergency_stop`、`interlock_trip`、`overtemperature_trip`、`pressure_relief_trip` 等硬触发器直接产生 P0 保护动作。

概率、分类、复核和安全等级是四个独立概念：

```text
failure_probability  模型风险概率
local_prediction     本地分类
review_state         是否需要云端复核
safety_priority      安全紧急程度
action               实际动作
```

P0 只能由硬安全触发器产生；模型概率不能直接升级为 P0。

### 3.2 Qwen 轻量专家

`edge-io-protocol/edge_inference.py` 调用本地 llama-server，使用 GBNF 将输出限制为一个类别数字。当前 Qwen 没有保留集校准，因此协议明确返回：

- `failure_probability: null`；
- `calibration_version: null`；
- `review_state: required`；
- `review_reason: ["qwen_fallback_uncalibrated"]`。

Qwen 的类别输出只能作为复核触发信号，不能被转换成伪造的 `confidence` 或业务优先级。

## 4. v3 协议与 ESW 压缩协议

完整审计结果使用 `edge-output/v3` JSON，主要字段由 `decision_contract.py` 生成。云端复核请求由 `compact_cloud_payload()` 构造，包含：

- `review_id`、`trace_id`、`idempotency_key`；
- 观测时间、决策时间和截止时间；
- 本地分类、故障概率、模型/校准/策略版本；
- 复核原因、安全等级和证据哈希；
- 网络状态。

`ESW_V1.md` 和 `ESW_V1_CLOUD_DECODER.md` 定义 32-bit/4-byte 的快速传输投影。ESW 只用于控制面和低延迟传输，不能替代完整 JSON 审计记录。

旧版协议文档仍可能出现 `confidence`、`priority` 等字段；v2 运行时的反馈校验会拒绝这些 legacy 字段。云端接入时应以 v2 的 machine-readable schema 和 decoder 为准，并将旧文档标为兼容参考。

## 5. 边云调度

当前调度分为两层：

```text
任务到达
    |
    v
24D pre-route observation
    |
    v
PPO: LOCAL_FIRST / DIRECT_CLOUD
    |
    +-- LOCAL_FIRST --> 边端推理 --> ACCEPT_EDGE
    |                              \-> CLOUD_REVIEW
    |
    \-- DIRECT_CLOUD -------------> 云端执行
```

`LOCAL_FIRST + CLOUD_REVIEW` 才是 collaborative 路径。等待队列采用非抢占 EDF，运行中的模型调用不会被新任务抢占。

资产路由配置 `config/asset_routing_v1.json` 默认没有指定设备，因此默认走 `LOCAL_FIRST`。资产重要度不能由输入任务或模型输出自行声明，必须来自外部权威资产清单。

需要特别区分：

- PPO 解决“推理前是否直接上云”；
- 未来的 `p_improve` 模型解决“边端结果出来后是否值得复核”。

两者不是同一个模型，也不能把现有 PPO smoke bundle 当成最终上云阈值模型。

## 6. 云端服务现状

`cloud_server/` 提供 FastAPI 接口，包括 chat、inference、heartbeat、tasks、escalations、ESW 解码、model updates 和 feedback。

模型后端支持 `mock`、`llama` 和 `vllm`，但默认配置仍为 `mock`。因此当前云端已验证的是：

```text
请求接收 -> 协议校验 -> 入队/流式接口 -> 反馈回写
```

尚未验证：真实云模型质量、真实云端排队和推理耗时、网络上传耗时、持久化队列可靠性，以及真正的 hidden-state split inference。`generate_from_hidden()` 在真实后端仍使用 prompt fallback，并未把 hidden state 注入模型中间层。

## 7. Shadow 与反馈闭环

`edge-io-protocol/routing_shadow.py` 只做 Phase A 数据采集，不训练生产路由器。它先排除 P0、网络不可达、任务不支持、deadline 过期或预算不足的样本，再对结构上可上云的样本做稳定随机 shadow。

每条记录包含：

- `shadow_eligible`；
- `upload_probability`；
- `selection_reason`；
- `policy_version`；
- `population_state`；
- 决策时特征可用性时间戳。

`feedback_loop.py` 保存边端和云端原始输出、解析结果、版本、耗时和确认标签。云边不一致只能作为弱信号；只有人工或其他明确的 `confirmed_outcome` 才能定义真实质量增益。

目标训练标签应为：

```text
DeltaQ = loss(edge, y) - loss(cloud, y)
p_improve = P(DeltaQ > 0 | decision-time features)
```

当前尚未接入真实云端 dual-run，因此还没有足够的真实增益标签，也没有可信的生产上云阈值。

## 8. 已有性能证据与边界

AI4I 表格 MoE 的进程内测试约为：

- 平均 `0.81 ms`；
- P50 `0.71 ms`；
- P95 `1.69 ms`；
- P99 `2.50 ms`。

Qwen IQ4_XS、约 393 输入 token、生成 1 token 的本地闭环测试约为：

- 平均 `72.68 ms`；
- P50 `66.54 ms`；
- P95 `107.63 ms`；
- P99 `155.36 ms`。

约 300 输入、10 输出时平均约 `348.97 ms`。这些结果只覆盖本地标准 JSON 到协议输出，不包括传感器采集、网络上传、云端排队和真实云模型。

因此不能把当前结果表述为完整云边端到端 SLA，也不能据此宣称云端复核后的总时延已经满足目标。

## 9. 当前结论

v2 已经具备：

1. 多数据集标准化输入；
2. AI4I 校准表格专家；
3. Qwen 短输出 fallback 专家；
4. 不混淆概率、复核、安全和动作的 v3 协议；
5. ESW 快速传输投影；
6. LOCAL_FIRST/DIRECT_CLOUD 两阶段调度；
7. 云端复核接口和审计反馈记录；
8. shadow 数据采集框架。

尚未闭合的关键链路是：

```text
真实云模型
    -> dual-run
    -> confirmed outcome
    -> DeltaQ
    -> p_improve 小模型
    -> 弱网自治率 >= 90% 的策略搜索
    -> frozen test 最终验证
```

在这条链路完成前，项目可以称为“可运行的云边协同原型”，不能称为“已经获得最终可信上云阈值的生产系统”。

