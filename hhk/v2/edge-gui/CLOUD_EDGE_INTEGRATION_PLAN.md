# 边端-云端协同对接实施计划

## 核心架构原则：双层决策 — 云端 PPO 主导，边端自主兜底

```
                      ┌──────────────────────────────────────────┐
                      │              云端（主决策大脑）            │
                      │                                           │
                      │  ┌───────────────────────────────────┐    │
                      │  │  PPO 调度引擎                      │    │
                      │  │  ├─ 全局状态感知（所有边端心跳汇总）  │    │
                      │  │  ├─ PPO 模型推理 → 最优路由决策     │    │
                      │  │  └─ 动态下发: 规则/阈值/LoRA/任务   │    │
                      │  └───────────────┬───────────────────┘    │
                      │                  │                        │
                      │  ┌───────────────▼───────────────────┐    │
                      │  │  全量大模型推理                     │    │
                      │  │  ├─ 云端复核（低置信度样本）         │    │
                      │  │  ├─ 冲突最终仲裁                    │    │
                      │  │  └─ 全局态势分析                    │    │
                      │  └───────────────────────────────────┘    │
                      └────────────────┬─────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │         网络良好              │         网络弱/断             │
        │   (云端 PPO 做主)             │    (边端自主决策兜底)         │
        │                              │                               │
        ▼                              ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│  边端节点 A       │          │  边端节点 A       │
│                  │          │                  │
│ 决策层:           │          │ 决策层:           │
│ ├─ 跟随云端PPO决策 │  断网    │ ├─ ★ 本地规则引擎  │
│ ├─ 拉取云端任务    │ ──────► │ │   自主决策       │
│ └─ 执行云端指令    │          │ └─ 使用云端上次同步 │
│                  │          │    的阈值/规则参数   │
│ 执行层:           │          │                  │
│ ├─ 本地推理(毫秒)  │          │ 执行层:           │
│ ├─ 心跳上报       │          │ ├─ 本地推理(纯本地) │
│ ├─ P2P 协同       │          │ ├─ P2P 协同(活跃)  │
│ └─ 结果融合       │          │ └─ 冲突本地消解    │
└──────────────────┘          └──────────────────┘
```

**双层决策模型**：

| 层级 | 条件 | 决策者 | 能力 | 说明 |
|------|------|--------|------|------|
| **L1 云端决策** | 网络正常 (RTT<200ms) | 云端 PPO 引擎 | 全局最优路由、动态阈值调整、模型更新分发 | 边端通过 TaskScheduler 拉取云端决策并执行 |
| **L2 边端自主** | 网络弱/断 (RTT>200ms 或心跳失败) | 边端本地规则引擎 | 基于云端上次同步的规则参数自主决策、纯本地推理 | 保证离线基本业务可用 |

**关键设计**：
- **网络好时**：边端不做路由决策，TaskScheduler 拉取云端 PPO 的任务指令并执行。边端的角色是"执行者"。
- **网络差时**：边端自动检测网络恶化 → 激活本地规则引擎 → 使用**云端断网前最后一次同步的阈值/规则参数**进行自主决策。这是"兜底"能力，不是替代云端。
- **网络恢复时**：自动切回 L1，边端重新拉取云端最新决策。本地产生的决策记录上传云端用于 PPO 模型持续训练。

---

## 当前状态总览

### 已完成（边端）
| 模块 | 状态 | 说明 |
|------|------|------|
| EdgeEngine 本地推理 | ✅ 完成 | Qwen3-1.7B IQ4_XS 量化，KV Cache 管理，Jinja 模板 |
| EdgeCloud 简单卸载 | ✅ 完成 | OpenAI 兼容 /v1/chat/completions SSE 流式 |
| EdgeCloudClient 协同客户端 | ✅ 完成 | Split/Query/Feature 三种推理模式 + Health Check |
| DeviceMonitor 资源采集 | ✅ 完成 | CPU/GPU/RAM/Network 全平台 |
| HeartbeatManager 心跳上报 | ✅ 完成 | 自适应频率（稳定30s/波动5s）|
| TaskScheduler 任务调度 | ✅ 完成 | 拉取云端任务→分发执行→结果上报，4 种任务类型 |
| ModelSyncManager 模型同步 | ✅ 完成 | 接收云端 LoRA + 决策规则，热加载即时生效 |
| P2PMesh 边边通信 | ✅ 完成 | UDP 广播发现 + TCP 可靠传输 |
| ConflictDetector 冲突消解 | ✅ 完成 | 四级本地消解 + 升级云端仲裁 |
| CLOUD_EDGE_INTERFACE.md | ✅ 完成 | 9 个 API 端点完整定义 + Flask 参考实现 |

### ⚠️ 边端缺失：本地自主决策引擎
| 模块 | 状态 | 说明 |
|------|------|------|
| **EdgeLocalDecision** | ✅ 完成 | `edge_local_decision.h/cpp` — 三级网络评估 + 四级本地决策 |

### 已完成（云端 Demo）
| 模块 | 状态 | 说明 |
|------|------|------|
| CloudEdgeSimulator 仿真器 | ✅ 完成 | 离散时隙仿真，FIFO/EDF 队列 |
| RuleScheduler 规则调度 | ✅ 完成 | 基于置信度+资源+网络的规则路由（**移植为云端在线调度引擎**）|
| PPOScheduler PPO 调度 | ✅ 完成 | Stable-Baselines3 PPO 模型推理（**移植为云端在线调度引擎**）|
| EnvironmentDynamics 环境 | ✅ 完成 | 资源均值回归/消耗恢复，网络随机游走/Markov |
| RewardCalculator 奖励 | ✅ 完成 | 延迟+队列+超时+准确率四项惩罚 |
| Streamlit 可视化 | ✅ 完成 | 调度过程实时展示 |

### 缺失（需要新建）
| 模块 | 状态 | 说明 |
|------|------|------|
| **云端 HTTP 服务** | ❌ 未建 | 需要实现边端期望的 9 个 API 端点 |
| **云端大模型推理** | ❌ 未建 | 需要部署全量模型 |
| **云端在线调度引擎** | ❌ 未建 | 将 Demo 的 RuleScheduler/PPOScheduler 移植为在线版本 |
| **多场景测试环境** | ❌ 未建 | 工业检测 + 交通监控双场景 |
| **端到端指标测试** | ❌ 未建 | TTFT/时延/冲突率等指标自动测试 |

---

## Phase 1：云端服务基础架构（预计 5-7 天）

### 1.1 云端 FastAPI 服务搭建 — 调度决策引擎是核心

**目标**：将 CLOUD_EDGE_INTERFACE.md 第七章的 Flask 参考实现升级为生产级 FastAPI 服务。**核心目标是让云端成为调度大脑，边端无条件跟随**。

**新建文件**：`cloud_server/` 目录（独立于 llama.cpp 项目）

```
cloud_server/
├── main.py                  # FastAPI 应用入口，注册所有路由
├── config.py                # 配置管理（模型路径、端口等）
├── api/
│   ├── __init__.py
│   ├── health.py            # GET /api/v1/health
│   ├── heartbeat.py         # POST /api/v1/edge/heartbeat （接收边端状态）
│   ├── tasks.py             # GET /api/v1/edge/tasks （边端拉取云端决策）
│   ├── inference.py         # POST /api/v1/infer/split + /api/v1/infer/features
│   ├── chat.py              # POST /v1/chat/completions （代理转发到大模型）
│   ├── model_updates.py     # GET /api/v1/edge/model-updates （云端推送更新）
│   ├── neighbors.py         # GET /api/v1/edge/neighbors
│   └── conflicts.py         # POST /api/v1/edge/conflicts （云端最终仲裁）
├── scheduler/
│   ├── __init__.py
│   ├── rule_scheduler.py    # 从 cloud_edge_demo_phase1 移植 RuleScheduler
│   ├── ppo_scheduler.py     # 从 cloud_edge_demo_phase1 移植 PPOScheduler
│   └── decision_engine.py   # ★ 云端总调度大脑 — 这是 Phase 1 的核心
├── inference/
│   ├── __init__.py
│   ├── model_manager.py     # 大模型后端统一管理（支持多种后端切换）
│   ├── backend_llama.py     # llama-server 后端适配器
│   ├── backend_vllm.py      # vLLM 后端适配器（GPU 环境可选）
│   └── streaming.py         # SSE 流式输出工具
├── storage/
│   ├── __init__.py
│   ├── device_state.py      # 全局设备状态存储（所有边端心跳汇总）
│   ├── task_queue.py        # 任务队列管理
│   └── models.py            # Pydantic 数据模型
├── requirements.txt         # FastAPI, uvicorn, httpx 等
└── tests/
    └── test_api.py          # API 端点测试
```

**关键实现**：

---

**1. 云端调度决策引擎**（`scheduler/decision_engine.py`）— **这是整个云边协同系统的核心大脑**：

从云端 Demo 移植并在线化的调度逻辑：

```python
class CloudDecisionEngine:
    """
    云端统一调度大脑。
    
    职责：
    1. 接收所有边端心跳 → 维护全局状态视图
    2. 对每个推理任务做出"边端/云端"路由决策
    3. 根据全局状态动态调整边端运行参数（阈值、规则）
    4. 管理模型版本分发（LoRA 更新）
    """

    def __init__(self):
        self.rule_scheduler = RuleScheduler(confidence_threshold=0.78)
        self.ppo_scheduler = None  # 延迟加载 PPO 模型
        self.device_states = {}    # device_id → 最新心跳数据
        self.use_ppo = False       # 是否启用 PPO 调度

    # ── 心跳触发：每次收到边端心跳后执行 ──
    def on_heartbeat(self, device_id: str, heartbeat: dict):
        """收到边端心跳 → 更新状态 → 生成调度决策 → 放入该设备的任务队列"""
        
        # 1. 更新全局设备状态
        self.device_states[device_id] = {
            'cpu_pct': heartbeat['cpu']['usage_pct'],
            'mem_avail_mb': heartbeat['memory']['avail_mb'],
            'tps': heartbeat['inference']['current_tps'],
            'rtt_ms': heartbeat['network']['rtt_ms'],
            'queue_len': heartbeat['inference']['task_queue_len'],
            'is_fluctuating': heartbeat.get('is_fluctuating', False),
            'last_seen': time.time(),
            'online': True,
        }
        
        # 2. 调度决策：云端决定这个边端该如何运行
        decisions = self._make_decisions(device_id)
        
        # 3. 将决策转化为任务，放入该设备的待拉取队列
        for d in decisions:
            self.task_queue.enqueue(device_id, d)
    
    # ── 核心决策逻辑 ──
    def _make_decisions(self, device_id: str) -> list[dict]:
        """云端根据全局状态做出所有调度决策"""
        state = self.device_states[device_id]
        decisions = []
        
        # 决策 1: 网络恶化 → 命令边端切换到纯本地模式
        if state['rtt_ms'] > 500:
            decisions.append({
                'type': 'rule_sync',
                'priority': 5,  # 最高优先级
                'description': '网络延迟过高，切换纯本地推理模式',
                'payload': {
                    'rule_id': 'network_degraded_mode',
                    'rule_type': 'override',
                    'target': 'inference_mode',
                    'value': {'mode': 'local_only', 'cloud_fallback': False},
                }
            })
        
        # 决策 2: 边端内存不足 → 命令卸载推理到云端
        elif state['mem_avail_mb'] < 1024:
            decisions.append({
                'type': 'cloud_review',
                'priority': 4,
                'description': '边端内存不足，卸载推理到云端',
                'payload': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'route': 'cloud',  # 云端明确指令：走云端
                }
            })
        
        # 决策 3: 边端 TPS 过低 → 降低本地置信度阈值
        if state['tps'] < 10.0:
            decisions.append({
                'type': 'rule_sync',
                'priority': 3,
                'description': '边端推理速度过低，降低决策阈值',
                'payload': {
                    'rule_id': 'low_tps_threshold_adjust',
                    'rule_type': 'threshold',
                    'target': 'confidence_threshold',
                    'value': {'min': 0.65},  # 从默认 0.78 降到 0.65
                }
            })
        
        # 决策 4: 周期性特征采集（用于云端持续优化模型）
        if self._should_collect_features(device_id):
            decisions.append({
                'type': 'upload_features',
                'priority': 1,
                'description': '周期性特征采集',
                'payload': {'layer_idx': -1, 'targets': ['all']},
            })
        
        return decisions

    # ── 边端轮询：边端来拉任务时调用 ──
    def get_pending_tasks(self, device_id: str) -> list[dict]:
        """边端 GET /api/v1/edge/tasks 时返回云端为该设备准备的任务"""
        return self.task_queue.pop_all(device_id)
    
    # ── PPO 决策（可选，需要 GPU/足够算力） ──
    def decide_route_ppo(self, task: Task, device_state: dict) -> Action:
        """使用训练好的 PPO 模型做路由决策"""
        if self.ppo_scheduler is None:
            self.ppo_scheduler = PPOScheduler(model_path="models/ppo_cloud_edge_fifo.zip")
        obs = build_observation(task, device_state)
        action, _ = self.ppo_scheduler.model.predict(obs, deterministic=True)
        return Action.EDGE if action == 0 else Action.CLOUD
```

**关键理念**：
- 云端不是被动等边端上报，而是**主动根据全局状态做出决策并下发**
- 边端的 `TaskScheduler` 只负责"拉取→执行→上报"，**不做自主路由决策**
- 决策规则、阈值、推理模式均由云端统一控制，边端即时响应

---

2. **设备状态管理**（`storage/device_state.py`）：
   - 维护每个边端节点的最新心跳快照
   - 在线/离线状态跟踪（心跳超时 60s 判离线）
   - 历史时序数据保留（用于调度决策的趋势分析）
   - 全局视图：云端可以看到所有边端节点的实时状态

### 1.2 云端大模型推理部署

**目标**：在云端搭建全量模型推理能力。

**设计原则**：后端可替换 — 不锁定具体推理框架，通过适配器模式支持多种后端。

```
FastAPI (cloud_server)
    │
    ├── 后端适配器 A: llama-server (CPU/GGUF)
    │   └── 与边端同源，开箱即用，适合没有 GPU 的环境
    │
    └── 后端适配器 B: vLLM (GPU)
        └── 高性能，适合有 GPU 服务器的环境
```

**默认方案**（不依赖 GPU）：**llama.cpp server**

```bash
# 启动云端大模型（具体模型待定，可用 DeepSeek/Qwen/其它全量模型）
./llama-server \
  -m models/<full-model>.gguf \
  --host 0.0.0.0 --port 8081 \
  -c 32768 \
  --parallel 4 \
  --api-key sk-cloud-edge-demo
```

**GPU 方案**（如果环境支持）：**vLLM**

```bash
# 如果有 GPU，切换到 vLLM 获得更高吞吐
python -m vllm.entrypoints.openai.api_server \
  --model <model-name> \
  --port 8081
```

FastAPI 服务通过 HTTP 调用推理后端的 `/v1/chat/completions`，两个后端的 API 格式完全一致（OpenAI 兼容），切换只需改配置。

**关键工作**：
- 准备全量模型的 GGUF 量化版本（或使用 vLLM 直接加载原始模型）
- 配置推理后端
- 在 FastAPI 中通过适配器层统一调用

### 1.3 推理协同策略：边端先跑、云端复核

**策略定位**：不追求复杂的 Split Inference（需要修改 llama.cpp 内部管线），采用实用的 **边端先跑 + 云端后台复核** 策略。

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────┐
│ 边端本地推理（毫秒级）                          │
│ ├─ Qwen3-1.7B / IQ4_XS                      │
│ ├─ TTFT ~50-200ms                            │
│ └─ 即时返回结果给用户                          │
│                                               │
│ 同时后台判断:                                  │
│ ├─ 置信度 < 云端设定的阈值？                     │
│ │   └─ YES → 触发云端复核（异步，不阻塞用户）      │
│ └─ 云端返回后 → 如果结果不一致 → 更新显示         │
└──────────────────────────────────────────────┘
    │  (异步)
    ▼
┌──────────────────────────────────────────────┐
│ 云端全量模型复核（百毫秒~秒级）                   │
│ ├─ 全量模型独立推理                             │
│ ├─ 更高准确率                                  │
│ └─ 结果回传边端 → TaskScheduler.fuse() 融合     │
└──────────────────────────────────────────────┘
```

**优势**：
- 用户始终感受毫秒级响应（边端先返回）
- 准确率接近全量模型（云端后台复核纠正）
- 不需要修改 llama.cpp 源码
- 网络断开时降级为纯边端推理，基本功能完全不中断

**Split Inference 预留**：如果后续需要做真正的模型层切分（边端跑前 K 层、云端跑剩余层），边端的 `EdgeCloudClient::split_infer()` 代码已经就绪，等云端条件成熟后直接对接即可。

### 1.4 边端本地自主决策引擎（新增模块 ★）

**目标**：在网络弱/断时，边端不是"傻等"，而是有一套自己的本地决策逻辑来兜底。

**设计原则**：
- **网络好时** → 边端本地引擎**静默**，TaskScheduler 拉取云端 PPO 决策并执行
- **网络恶化时** → 边端本地引擎**自动激活**，用云端上次同步的参数自主决策
- **网络恢复时** → 切回云端 PPO，上传离线期间的决策记录供云端分析

**新建文件**：`edge-gui/edge_local_decision.h` + `edge-gui/edge_local_decision.cpp`

```
边端本地决策引擎 EdgeLocalDecision
│
├── 输入: 当前网络状态 + 任务特征 + 云端上次同步的规则参数
├── 输出: LocalAction (本地推理 / 排队等云端 / P2P共识)
│
└── 决策逻辑:
    ┌─────────────────────────────────────────────────────┐
    │ 判断网络状态                                         │
    │                                                     │
    │ RTT < 200ms 且 云端可达                              │
    │   → FollowCloud: 不介入，等云端 PPO 决策              │
    │                                                     │
    │ RTT 200-500ms 或 心跳连续失败 2-4 次                  │
    │   → 弱网模式:                                        │
    │     ├─ 高置信度任务 (conf > 同步阈值) → LocalInfer    │
    │     ├─ 低置信度任务 → QueueForCloud (网络恢复后复核)   │
    │     └─ P2P 活跃 → 邻居节点可互相校验                  │
    │                                                     │
    │ RTT > 500ms 或 心跳连续失败 >= 5 次                   │
    │   → 离线模式:                                        │
    │     ├─ 所有任务纯本地推理 (LocalInfer)                │
    │     ├─ 使用云端断网前最后同步的阈值参数               │
    │     ├─ 降低决策门槛 (confidence_threshold - 0.15)    │
    │     ├─ P2P 仍活跃 → P2PConsensus 替代云端仲裁        │
    │     └─ 决策记录写入离线日志 (网络恢复后上传云端)      │
    └─────────────────────────────────────────────────────┘
```

**数据结构**：

```cpp
// edge-gui/edge_local_decision.h

enum class NetworkCondition {
    Healthy,     // RTT < 200ms, cloud reachable → 云端 PPO 做主
    Degraded,    // RTT 200-500ms → 混合模式，高置信度本地、低置信度排队
    Offline,     // RTT > 500ms or heartbeat failed → 纯本地自主
};

enum class LocalAction {
    FollowCloud,       // 网络好，等云端指令（边端仅做本地推理，路由由云端决定）
    LocalInfer,        // 本地推理
    QueueForCloud,     // 排队等待云端恢复后复核（弱网时对低置信度任务的策略）
    P2PConsensus,      // P2P 多节点共识（离线时替代云端仲裁）
};

struct DecisionSnapshot {
    NetworkCondition condition;
    LocalAction      action;
    float            confidence_threshold_used;  // 实际使用的阈值
    std::string      reason;
    int64_t          timestamp_ms;
};

class EdgeLocalDecision {
public:
    // ── 从云端同步参数（断网后继续使用这些值） ──
    void update_cloud_params(float confidence_threshold,
                             float anomaly_threshold,
                             int   max_queue_len);

    // ── 核心: 根据网络状态和任务特征做本地决策 ──
    LocalAction decide(float local_confidence,
                       NetworkCondition network);

    // ── 评估当前网络状况 ──
    NetworkCondition assess_network(int rtt_ms, bool cloud_reachable,
                                     int consecutive_heartbeat_failures);

    // ── 离线决策记录（用于网络恢复后上传云端） ──
    std::vector<DecisionSnapshot> flush_offline_decisions();

    // ── 查询 ──
    NetworkCondition current_condition() const { return _current_condition; }
    float active_confidence_threshold() const;  // 返回当前生效的阈值

private:
    // 云端上次同步的参数（断网后的"锦囊"）
    float _cloud_confidence_threshold = 0.78f;
    float _cloud_anomaly_threshold   = 0.85f;
    int   _cloud_max_queue_len       = 10;

    NetworkCondition _current_condition = NetworkCondition::Healthy;

    // 离线决策日志（网络恢复后上传）
    std::vector<DecisionSnapshot> _offline_log;
};
```

**与现有模块的集成关系**：

```
EdgeApp (app.cpp) — submit_message() 中的决策分流
    │
    ├── _local_decision.assess_network(rtt, reachable, hb_fails)
    │     │
    │     ├── Healthy  → TaskScheduler 正常工作，拉取云端 PPO 决策
    │     ├── Degraded → _local_decision.decide() 判断每条任务
    │     └── Offline  → _local_decision.decide() 纯本地自主
    │
    ├── _local_decision 依赖 _model_sync.active_rules()
    │   (获取云端断网前最后同步的阈值/规则)
    │
    └── 网络恢复时:
        ├── flush_offline_decisions() → 上传云端供 PPO 分析
        └── update_cloud_params(...)  → 重新同步最新参数
```

**集成伪代码**（在 `app.cpp:submit_message()` 中）：

```cpp
void EdgeApp::submit_message(const std::string & text) {
    // ... 构造消息 ...

    // 1. 评估网络状态
    int rtt = _device_monitor.measure_rtt_ms(_cloud_endpoint);
    bool reachable = _cloud_client.is_reachable();
    int hb_fails = _heartbeat.consecutive_failures();  // 需在 HeartbeatManager 中新增此方法
    auto net_cond = _local_decision.assess_network(rtt, reachable, hb_fails);

    if (net_cond == NetworkCondition::Healthy) {
        // ── 网络好: 云端 PPO 做主，边端跟随 ──
        // 边端先本地推理给用户快速响应
        _engine.generate(prompt, on_token, on_perf);
        // 后台: TaskScheduler 拉取云端任务，如果云端下发 cloud_review 则异步复核

    } else if (net_cond == NetworkCondition::Degraded) {
        // ── 弱网: 边端自主初筛 ──
        auto action = _local_decision.decide(local_conf, net_cond);
        if (action == LocalAction::LocalInfer) {
            _engine.generate(prompt, on_token, on_perf);  // 纯本地
        } else {
            _engine.generate(prompt, on_token, on_perf);  // 本地先跑
            _pending_cloud_queue.push_back(prompt);       // 低置信度 → 排队等网络恢复
        }

    } else {
        // ── 离线: 完全自主 ──
        auto action = _local_decision.decide(local_conf, net_cond);
        if (action == LocalAction::P2PConsensus && _p2p.is_running()) {
            // P2P 共识模式: 广播给邻居节点一起判断
            PerceptionReport report = build_report(text, local_result);
            _p2p.broadcast_perception(report);
        }
        _engine.generate(prompt, on_token, on_perf);  // 纯本地推理
    }
}
```

**需要伴随修改的现有文件**：

| 文件 | 修改内容 |
|------|---------|
| `heartbeat.h/cpp` | 新增 `consecutive_failures()` 方法，统计连续心跳失败次数 |
| `app.h` | 新增 `EdgeLocalDecision _local_decision` 成员 |
| `app.cpp` | `submit_message()` 增加网络状态判断和决策分流逻辑 |
| `CMakeLists.txt` | 新增 `edge_local_decision.cpp` 到编译 |

---

## Phase 2：云边联调 — 云端决策闭环验证（预计 3-5 天）

### 2.1 边端对接配置

**目标**：让 edge-gui.exe 成功连接到云端 FastAPI 服务。

**步骤**：
1. 启动云端服务：`python cloud_server/main.py`（端口 8080）
2. 启动推理后端（llama-server 或 vLLM，端口 8081）
3. 在 edge-gui "云端配置"面板中填入对应地址
4. 点击"测试连接" → 验证 `GET /api/v1/health` 返回 200
5. 发送测试消息 → 验证 Query Offload 流程端到端通

### 2.2 云端主导的调度流程验证

**核心验证：边端上报 → 云端决策 → 边端执行 的闭环**

```
边端                         云端
 │                            │
 │── ① 心跳上报 ──────────────▶│  收到设备状态，更新全局视图
 │   (CPU/RAM/TPS/RTT)        │  决策引擎分析 → 生成任务列表
 │                            │
 │◀── ② 任务下发 ──────────────│  边端 GET /tasks 拉取任务
 │   [cloud_review,            │  （云端已提前放入该设备的任务队列）
 │    rule_sync,               │
 │    model_update, ...]       │
 │                            │
 │── ③ 执行任务 ──────────────▶│
 │   执行 cloud_review:        │  云端大模型推理
 │   POST /v1/chat/completions │  逐 token SSE 流式返回
 │                            │
 │── ④ 上报结果 ──────────────▶│  云端记录，用于后续优化
 │   (fused result)            │
```

**验证流程**：
1. 启动 edge-gui → HeartbeatManager 自动开始 30s 间隔上报
2. 云端收到心跳 → 决策引擎分析 → 如果满足条件（如内存不足），下发 `cloud_review` 任务
3. 边端 TaskScheduler 每 2s 拉取 → 收到任务 → 自动执行
4. 验证边端 UI "任务面板" 显示完整的"收到任务→执行→完成"流程
5. 验证 `rule_sync` 任务：云端下发新阈值 → 边端 ModelSyncManager.apply_rule() → 立即生效

### 2.3 边端 "跟随云端" 行为验证

**验证边端不自主决策，完全由云端指挥**：

| 云端指令 | 边端行为 | 验证方法 |
|---------|---------|---------|
| `rule_sync: mode=local_only` | 边端停止所有云端卸载，纯本地推理 | 查看 UI 云端状态变为"已禁用" |
| `cloud_review: route=cloud` | 边端将指定任务发送云端 | 查看任务面板显示"cloud_review" |
| `rule_sync: confidence_threshold=0.65` | 边端降低本地决策阈值 | 查看 ConflictDetector 规则更新 |
| `model_update: lora_v3` | 边端下载并热加载新 LoRA | 查看模型信息面板版本号变化 |

### 2.4 异常场景测试

| 测试场景 | 云端行为 | 边端行为 | 验证指标 |
|---------|---------|---------|---------|
| 边端断网 | 心跳超时 → 标记离线 → 通知邻居节点 | 自动切换到纯本地模式 | 基本推理功能 100% 保持 |
| 网络恢复 | 收到心跳 → 标记上线 → 恢复任务下发 | 自动重连，恢复心跳 | 恢复时间 < 30s |
| 云端不可达 | — | 边端降级运行 + 等待云端恢复 | 本地推理不受影响 |
| 边端内存不足 | 下发 cloud_review 卸载推理 | 执行云端卸载任务 | 不因内存不足崩溃 |

---

## Phase 3：技术指标达成（预计 5-7 天）

### 3.1 TTFT 减少 75%

**当前基线测量**：
- 纯云端推理 TTFT: ~500-2000ms（取决于网络和模型大小）
- 纯边端推理 TTFT: ~50-200ms（1.7B IQ4_XS 量化模型）
- **当前 TTFT 减少**: 边端相比云端已经减少 75%-90%

**进一步优化**：
1. **边端 prompt processing 优化**：
   - 使用 Flash Attention（llama.cpp 已自动启用）
   - 减小 prompt batch size → 降低首 token 延迟
2. **预测性 prefetch**：
   - 用户输入过程中，边端对已输入的部分文字进行预编码
   - 用户按下"发送"时，prompt 已部分 prefill
3. **云端并行预热**：
   - 边端判断可能需要云端协助时，提前发送上下文到云端预热

### 3.2 单次推理内存占用 ≤ 1.5GB

**当前状态**（已验证）：
- Qwen3-1.7B IQ4_XS：模型 ~1.1GB + KV Cache ~224MB + 计算缓冲 ~305MB = **~1.6GB**
- 略超 1.5GB 目标

**优化方案**：
1. **更小 KV Cache**：将 `n_ctx` 从 2048 降到 1536 → KV Cache 降至 ~168MB
2. **更激进的量化**：使用 IQ2_XS 或 Q2_K 量化 → 模型降至 ~0.7GB
3. **更小的模型**：使用 Qwen3-1.0B 或 SmolLM2-0.5B → 总内存 < 1GB
4. **mmap 模式**：使用内存映射加载模型，OS 自动管理物理内存

**推荐组合**：Qwen3-1.7B IQ3_XS + n_ctx=1536 → 预计总内存 ~1.3GB

### 3.3 网络波动期间基本业务功能保持率 ≥ 90%

**实现策略**：

1. **离线模式自动切换**（代码中已有基础）：
   ```
   心跳连续失败 3 次 → 进入离线模式
   ├── 所有推理走本地 EdgeEngine
   ├── 降低异常检测阈值（减少误报）
   ├── P2P 网格保持活跃（与其他边端节点协作）
   └── UI 显示"离线模式"状态
   ```

2. **网络状态感知的任务路由**：
   ```python
   # 云端调度引擎
   def route_task(device_metrics):
       rtt = device_metrics["network"]["rtt_ms"]
       if rtt > 500:  # 高延迟
           return "local_only"  # 不下发云端依赖任务
       elif rtt > 200:  # 中延迟
           return "local_with_async_cloud"  # 本地优先+异步云端
       else:  # 低延迟
           return "adaptive"  # 动态选择最优路径
   ```

3. **优雅降级策略**：
   | 网络状态 | 推理策略 | 准确率 | 延迟 |
   |---------|---------|--------|------|
   | 正常 (RTT<50ms) | 本地+云端融合 | 95%+ | <100ms |
   | 波动 (RTT 50-200ms) | 本地优先+异步云端 | 85-90% | <100ms |
   | 恶化 (RTT 200-500ms) | 纯本地+降低阈值 | 75-85% | <100ms |
   | 断开 (RTT>500ms) | 纯本地+P2P协作 | 70-80% | <100ms |

4. **基本业务可用性定义**：
   - 核心推理功能（问答/分类）正常 → 100% 保持
   - 云端复核功能暂停 → 不影响基本使用
   - P2P 冲突消解降级为本地规则 → 仍可运行

### 3.4 决策冲突比例 ≤ 5%，冲突解决成功率 ≥ 90%

**当前实现状态**（ConflictDetector 已实现）：
- 四级消解规则：置信度差距 → 传感器权威 → 邻近优先 → 多数共识
- 冲突检测逻辑：分类冲突、置信度冲突、动作冲突
- 云端仲裁升级机制

**需要补充**：
1. **冲突统计面板**：在 edge-gui 中增加 `ConflictStats` 实时展示
2. **规则参数优化**：基于仿真数据调整规则权重（如 confidence_margin 从 0.3 调为 0.25）
3. **冲突日志记录**：将所有冲突记录写入本地文件，用于离线分析

**验证方法**：
- 启动 3 个 edge-gui 实例（模拟 3 个边缘节点）
- 使用相同测试数据（模拟重叠感知区域）
- 统计冲突比例和解决成功率

---

## Phase 4：多场景部署验证（预计 5-7 天）

### 4.1 场景一：工业质检（高实时性）

**场景描述**：
- 产线摄像头实时采集产品图像
- 边端轻量模型进行初步缺陷检测（毫秒级）
- 低置信度样本自动上传云端复核（**云端决策阈值**）
- 多工位边缘节点 P2P 协同（同一产线不同角度）

**部署拓扑**：
```
┌──────────────────────────────────────────────┐
│                    云端                       │
│  FastAPI Server + 全量模型                    │
│  + 云端调度引擎（全局决策大脑）                 │
└────┬──────────────┬──────────────┬───────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ edge-01 │◄─►│ edge-02 │◄─►│ edge-03 │  P2P Mesh
│ 工位A   │   │ 工位B   │   │ 工位C   │
│ 正面拍摄 │   │ 侧面拍摄 │   │ 顶部拍摄 │
└─────────┘   └─────────┘   └─────────┘
```

**测试指标**：
| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| 边端推理延迟 | < 100ms | EnginePerf.ttft_ms |
| 云复核端到端延迟 | < 500ms | 心跳 RTT + 云端推理时间 |
| 缺陷检测准确率 | > 90%（边端）/ > 98%（云端融合）| 用户已有标注测试集 |
| P2P 冲突比例 | ≤ 5% | ConflictDetector stats |
| 网络断开后可用性 | 100%（纯本地检测继续）| 拔网线测试 |

**数据准备**：用户已有工业缺陷数据集，编写测试脚本批量验证。

### 4.2 场景二：交通监控（广域覆盖）

**场景描述**：
- 多个路口摄像头边缘节点
- 每个节点独立检测车辆/行人
- 相邻路口节点 P2P 协同（同一车辆跨路口追踪）
- **云端全局交通态势分析 + 统一调度各节点参数**

**部署拓扑**：
```
┌──────────────────────────────────────────────┐
│                    云端                       │
│  全局交通态势分析 + 统一调度                    │
│  下发各节点检测阈值 + 跨区域协同参数            │
└────┬──────────────┬──────────────┬───────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ 路口A    │◄─►│ 路口B    │◄─►│ 路口C    │
│ 南北方向  │   │ 东西方向  │   │ 环岛入口  │
└─────────┘   └─────────┘   └─────────┘
     │              │              │
     └──── 重叠感知区域（车辆跨路口）──┘
```

**测试指标**：
| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| 边端检测延迟 | < 50ms | 单帧推理时间 |
| 跨节点目标追踪一致性 | > 95% | ConflictDetector 同一目标分类一致率 |
| 云端全局分析延迟 | < 2s | HTTP 请求往返时间 |
| 平均端到端时延 | < 0.2s | 从采集到决策全链路计时 |

**数据准备**：用户已有交通监控数据集（如 VisDrone/COCO），准备连续帧序列模拟视频流。

### 4.3 端到端时延测试

**测试框架**（`tests/e2e_latency_test.py`）：

```python
class E2ELatencyTest:
    def test_local_inference_latency(self):
        """测试纯边端推理延迟 → 期望 < 200ms"""
        
    def test_cloud_offload_latency(self):
        """测试云端卸载延迟 → 期望 < 2s"""
        
    def test_adaptive_routing_latency(self):
        """云端决策 → 边端执行全链路 → 期望 < 0.2s"""
        
    def test_e2e_perception_latency(self):
        """摄像头帧 → 边端检测 → 云端复核 → 决策输出 → 期望 < 0.2s"""
```

---

## Phase 5：最终集成与展示准备（预计 3-5 天）

### 5.1 系统集成

1. **一键启动脚本**：
   ```bash
   # start_cloud_edge_system.sh
   # 1. 启动云端 FastAPI 服务
   # 2. 启动 llama-server（云端大模型）
   # 3. 启动 N 个 edge-gui 实例（模拟多边端）
   ```

2. **Docker 化部署**（可选）：
   ```dockerfile
   # Dockerfile.cloud
   FROM python:3.11
   COPY cloud_server/ /app
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
   ```

### 5.2 演示脚本准备

**场景一演示脚本**（工业质检）：
1. 展示边端实时检测 → 低延迟（< 100ms）
2. 故意制造低置信度场景 → 触发云复核
3. 展示云端结果融合 → 准确率提升
4. 断开网络 → 展示离线继续工作
5. 恢复网络 → 展示自动重连

**场景二演示脚本**（交通监控）：
1. 展示 3 个边端节点各自检测
2. 故意制造重叠检测 → 触发冲突检测
3. 展示 P2P 消解过程 → 冲突成功解决
4. 展示云端全局态势图

### 5.3 指标收集与报告

**自动化指标采集**：
```python
# metrics_collector.py
class CompetitionMetricsCollector:
    def collect_all(self):
        return {
            "ttft_reduction": self.measure_ttft_reduction(),     # 目标: ≥75%
            "memory_usage": self.measure_peak_memory(),          # 目标: ≤1.5GB
            "offline_availability": self.measure_offline_rate(), # 目标: ≥90%
            "conflict_ratio": self.measure_conflict_ratio(),     # 目标: ≤5%
            "resolution_rate": self.measure_resolution_rate(),   # 目标: ≥90%
            "avg_e2e_latency": self.measure_avg_latency(),       # 目标: ≤0.2s
        }
```

---

## 时间线与里程碑

```
Week 1 (Day 1-7):  Phase 1 — 云端服务基础架构
  ├── Day 1-2: FastAPI 服务框架 + 全部 9 个端点
  ├── Day 3-4: 大模型部署（llama-server）+ 调度决策引擎
  ├── Day 5-6: Query Offload 端到端打通 + Split Inference 预留
  └── Day 7:   基础功能测试

Week 2 (Day 8-12): Phase 2 — 云边联调
  ├── Day 8-9: 边端配置对接 + 心跳验证
  ├── Day 10-11: 任务调度闭环 + 全流程测试
  └── Day 12:   异常场景测试（断网、高负载等）

Week 3 (Day 13-19): Phase 3 — 技术指标达成
  ├── Day 13-14: TTFT 优化 + 内存优化
  ├── Day 15-16: 网络波动韧性 + 冲突消解调优
  └── Day 17-19: 指标测量 + 调优迭代

Week 4 (Day 20-26): Phase 4 — 多场景部署
  ├── Day 20-22: 工业质检场景搭建 + 测试
  ├── Day 23-25: 交通监控场景搭建 + 测试
  └── Day 26:   端到端时延测试

Week 5 (Day 27-31): Phase 5 — 集成与展示
  ├── Day 27-28: 系统集成 + 一键启动
  ├── Day 29-30: 演示脚本 + 指标报告
  └── Day 31:   最终验收
```

---

## 关键技术风险与对策

| 风险 | 影响 | 概率 | 对策 |
|------|------|------|------|
| 云端大模型 GPU/内存需求过大 | 高 | 中 | 使用 GGUF 量化版本，支持 llama.cpp server CPU 模式作为 fallback |
| 边端 1.5GB 内存无法同时运行模型+GUI | 中 | 低 | 已实测 ~1.6GB，调整 n_ctx+量化等级可降至 1.3GB |
| P2P 在跨子网环境下无法发现 | 中 | 高 | 云端辅助发现为主要方式，UDP 广播为同子网补充 |
| 比赛现场网络条件不确定 | 高 | 中 | 所有核心功能均支持离线运行，云端能力为增量提升 |

---

## 云端部署：方案灵活适配

由于云端具体环境（GPU 有无、模型选择）待定，方案设计了**双后端可切换**架构：

```
cloud_server/
├── inference/
│   ├── backend_llama.py    # 后端A: llama-server (CPU/GGUF)
│   └── backend_vllm.py     # 后端B: vLLM (GPU)
│
# 切换方式：修改 config.py 中的一行配置
# INFERENCE_BACKEND = "llama"  # 或 "vllm"
```

| 选项 | 适用条件 | 模型格式 | 启动方式 |
|------|---------|---------|---------|
| llama-server | 有/无 GPU 均可 | GGUF 量化 | `./llama-server -m model.gguf` |
| vLLM | 需要 NVIDIA GPU | 原始模型 / AWQ | `vllm serve model-name` |

两种后端的 API 完全一致（OpenAI 兼容 `/v1/chat/completions`），FastAPI 通过适配器层调用，切换后端**无需修改边端代码**。

**模型选择灵活**：不锁定 DeepSeek，支持任何 OpenAI 兼容的推理后端（Qwen/GLM/Llama 等），最终模型待确定云端环境后配置。

---

## 边端待适配项（根据云端最终 API 地址微调）

边端代码中需要根据云端实际部署地址修改的配置点（均在 `app.h` 默认值中，运行时可通过 UI 修改）：

| 配置项 | 当前默认值 | 说明 |
|--------|-----------|------|
| `_cloud_endpoint` | `http://localhost:8080/v1/chat/completions` | EdgeCloud 文本卸载地址 |
| `_heartbeat_endpoint` | `http://localhost:8080/api/v1/edge/heartbeat` | 心跳上报地址 |
| `EdgeCloudClient._base_url` | 通过 `set_endpoint()` 设置 | 协同推理基地址 |
| `P2PMesh._cloud_disc_url` | 通过 `set_cloud_discovery_url()` 设置 | 邻居发现地址 |

所有地址均可在 edge-gui 的"云端配置"面板中实时修改，无需重新编译。

---

> **本方案基于对以下代码库的完整分析**：
> - 云端 Demo: `D:\大学\揭榜挂帅\cloud_edge_demo_phase1`（Streamlit 仿真，Python，RuleScheduler + PPOScheduler）
> - 边端项目: `D:\llama.cpp\edge-gui`（C++ 桌面应用，已完成 7 个云边协同子系统）
> - 接口文档: `D:\llama.cpp\edge-gui\CLOUD_EDGE_INTERFACE.md`（9 个 API 端点完整定义 + Flask 参考实现）
> - 更新日期: 2026-07-20

---

## 实施进度记录

### 2026-07-20：边端侧完成

| 文件 | 操作 | 说明 |
|------|------|------|
| `edge-gui/edge_local_decision.h` | **新建** | EdgeLocalDecision 类声明：NetworkCondition 三级评估 + LocalAction 四级决策 + DecisionSnapshot 离线日志 |
| `edge-gui/edge_local_decision.cpp` | **新建** | 完整实现：assess_network() 三级判断、decide() 决策分流、flush_offline_decisions() 日志导出、active_confidence_threshold() 动态阈值 |
| `edge-gui/heartbeat.h` | 修改 | 新增 `consecutive_failures()` 方法 + `_consecutive_failures` 计数器 |
| `edge-gui/heartbeat.cpp` | 修改 | send_heartbeat() / send_now() 中成功重置、失败递增计数器 |
| `edge-gui/app.h` | 修改 | 新增 `#include "edge_local_decision.h"` + `EdgeLocalDecision _local_decision` 成员 |
| `edge-gui/app.cpp` | 修改 | submit_message() 增加网络感知决策分流；render_top_bar() 显示 [net:ok/degraded/offline] 状态；update_device_metrics() 中检测网络恢复并 flush 离线日志 |
| `edge-gui/CMakeLists.txt` | 修改 | 新增 `edge_local_decision.cpp` + `edge_local_decision.h` 到编译 |

### 2026-07-20：云端侧（完成）

| 文件 | 操作 | 说明 |
|------|------|------|
| `cloud_server/main.py` | **新建** | FastAPI 应用入口 + lifespan 事件 + CORS + 8 个路由注册 |
| `cloud_server/config.py` | **新建** | 全环境变量配置（CLOUD_* 前缀），支持 backend/model/scheduler 切换 |
| `cloud_server/api/health.py` | **新建** | `GET /api/v1/health` — 健康检查 |
| `cloud_server/api/heartbeat.py` | **新建** | `POST /api/v1/edge/heartbeat` — 心跳接收 + 触发调度引擎 |
| `cloud_server/api/tasks.py` | **新建** | `GET /api/v1/edge/tasks` — 边端拉取任务（POP 语义） |
| `cloud_server/api/inference.py` | **新建** | `POST /api/v1/infer/split` + `/features` — Split/Feature 推理 |
| `cloud_server/api/chat.py` | **新建** | `POST /v1/chat/completions` — OpenAI 兼容 SSE 流式 |
| `cloud_server/api/model_updates.py` | **新建** | `GET /api/v1/edge/model-updates` — LoRA+规则更新查询 |
| `cloud_server/api/neighbors.py` | **新建** | `GET /api/v1/edge/neighbors` — P2P 邻居发现 |
| `cloud_server/api/conflicts.py` | **新建** | `POST /api/v1/edge/conflicts` — 三级冲突仲裁 |
| `cloud_server/scheduler/decision_engine.py` | **新建** | CloudDecisionEngine — 6 维度任务生成决策 |
| `cloud_server/scheduler/ppo_adapter.py` | **新建** | PPO 模型加载适配器（对接 cloud_edge_demo_phase1） |
| `cloud_server/inference/model_manager.py` | **新建** | 统一后端管理器（llama/vllm/mock 可切换） |
| `cloud_server/inference/backend_llama.py` | **新建** | llama-server HTTP 代理适配器 |
| `cloud_server/inference/backend_vllm.py` | **新建** | vLLM HTTP 代理适配器 |
| `cloud_server/inference/streaming.py` | **新建** | SSE 流式输出工具函数 |
| `cloud_server/storage/models.py` | **新建** | Pydantic 数据模型（Heartbeat/Task/Conflict 等） |
| `cloud_server/storage/device_state.py` | **新建** | 线程安全全局设备状态存储 |
| `cloud_server/storage/task_queue.py` | **新建** | 每设备任务队列管理（POP 语义） |
| `cloud_server/requirements.txt` | **新建** | fastapi, uvicorn, pydantic, httpx, pytest, httpx |
| `cloud_server/tests/test_api.py` | **新建** | **17 个测试用例**，覆盖全部 9 端点 + 全流程集成测试 |

**启动方式**：
```bash
cd D:\llama.cpp\cloud_server
pip install -r requirements.txt
python main.py
# 服务启动在 http://0.0.0.0:8080，API 文档在 http://localhost:8080/docs
```

**配置方式**（环境变量）：
```bash
# 切换推理后端
set CLOUD_MODEL_BACKEND=mock    # mock / llama / vllm

# 启用 PPO 调度
set CLOUD_USE_PPO_SCHEDULER=true
set CLOUD_PPO_MODEL_PATH=D:\...\models\ppo_cloud_edge_fifo.zip

# 连接真实大模型
set CLOUD_MODEL_BACKEND=llama
set CLOUD_LLAMA_SERVER_URL=http://localhost:8081
```
