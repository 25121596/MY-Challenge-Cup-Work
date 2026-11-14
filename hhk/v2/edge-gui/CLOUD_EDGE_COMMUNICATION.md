# 边端-云端通信逻辑详解

> 日期: 2026-07-20  
> 基于: `D:\llama.cpp\edge-gui\` 源码分析

---

## 一、总体架构

边端和云端之间有 **7 条独立通信通道**，各自由不同的模块管理，互不干扰：

```
                        ┌──────────────────────────┐
                        │         云端              │
                        │  FastAPI Server           │
                        │  + 大模型推理后端          │
                        │  + 调度决策引擎            │
                        └──────────┬───────────────┘
                                   │
        ┌──────────────┬───────────┼───────────┬──────────────┬──────────────┐
        │              │           │           │              │              │
        ▼              ▼           ▼           ▼              ▼              ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │通道一    │  │通道二    │  │通道三    │  │通道四    │  │通道五    │  │通道六+七 │
   │消息卸载  │  │心跳上报  │  │任务拉取  │  │模型同步  │  │邻居发现  │  │冲突+协同 │
   └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
   手动触发      后台自动      后台自动      后台自动      后台自动      自动触发
   (用户勾选)   (面板启用)   (随心跳)     (云端下发)   (随P2P)      (条件触发)
```

---

## 二、七条通道详解

### 通道一：消息卸载（用户手动触发）

**这是最直观的一条——把用户消息发给云端大模型，代替本地推理。**

```
触发条件: 用户勾选"启用云端推理" + 填入接口地址 + 点击"测试连接"成功 + 发送消息
通信协议: HTTP POST (OpenAI 兼容格式)
API 端点: POST /v1/chat/completions
请求方向: 边端 → 云端
响应方式: SSE 流式 (text/event-stream)，逐 token 返回
管理模块: cloud.cpp — EdgeCloud 类

代码位置:
  app.cpp:640   — ImGui::Checkbox("启用云端推理", &_cloud_enabled)
  app.cpp:833   — if (_cloud_enabled && _cloud.is_reachable()) { _cloud.send(...) }
  cloud.cpp:59  — EdgeCloud::run_thread() 构造 OpenAI JSON + HTTP POST
```

**请求内容**：
```json
{
  "model": "deepseek-v2",
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "用户的问题"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": true
}
```

**边端发送流程**：
```
EdgeApp::submit_message()
  │
  ├── _cloud_enabled == false → 走本地 EdgeEngine::generate()
  │
  ├── _cloud_enabled == true && _cloud.is_reachable() == true
  │     └── _cloud.send(json, on_token)
  │           └── 后台线程: HTTP POST → 接收 SSE → 逐 token 回调 on_token
  │
  └── _cloud_enabled == true && _cloud.is_reachable() == false
        └── 降级走本地，状态栏显示橙色 [本地]
```

**关键点**：
- 这个勾选框**只控制发消息走本地还是云端**
- 不影响心跳、任务拉取、P2P 等其他功能
- 云端不可达时自动降级，不阻塞用户

---

### 通道二：心跳上报（后台自动）

**边端定期向云端报告自己的"体检数据"，云端据此做全局调度决策。**

```
触发条件: 用户在"心跳"面板勾选启用
发送频率: 稳定时 30s / 波动时 5s（自适应）
通信协议: HTTP POST (JSON)
API 端点: POST /api/v1/edge/heartbeat
请求方向: 边端 → 云端
响应方式: 200 OK 或 503
管理模块: heartbeat.cpp — HeartbeatManager 类（独立后台线程）

代码位置:
  heartbeat.cpp:53  — run_loop() 主循环，自适应间隔
  heartbeat.cpp:153 — build_report_json() 构造上报 JSON
  app.cpp:952       — 心跳面板 Checkbox 控制 _heartbeat.start()/stop()
```

**上报内容**（每次心跳发送的 JSON）：
```json
{
  "device_id": "edge-node-01",
  "device_type": "jetson-orin-nano",
  "timestamp_ms": 1721200000000,
  "is_fluctuating": false,
  "cpu": { "usage_pct": 45.2, "core_count": 8, "freq_mhz": 0 },
  "memory": { "total_mb": 8192, "avail_mb": 3200 },
  "gpus": [{ "name": "NVIDIA Orin", "total_mb": 4096, "free_mb": 2100, "used_mb": 1996 }],
  "network": { "rtt_ms": 12, "packet_loss_pct": 0.0, "cloud_reachable": true },
  "inference": { "current_tps": 45.3, "task_queue_len": 2 }
}
```

**云端收到心跳后做什么**（`decision_engine.on_heartbeat()`）：

```
云端收到心跳
  │
  ├── 更新全局设备状态表
  │     └── device_state.update_from_heartbeat(body)
  │
  ├── 分析设备状态 → 生成调度决策
  │     ├── 网络 RTT > 500ms → 下发 rule_sync: 切换到纯本地模式
  │     ├── 内存 < 1GB     → 下发 cloud_review: 卸载推理到云端
  │     ├── TPS < 10       → 下发 rule_sync: 降低置信度阈值
  │     ├── 设备波动中      → 下发 rule_sync: 降低异常检测阈值
  │     ├── GPU 显存不足    → 下发 upload_features: 上传特征到云端
  │     └── LoRA 有新版本   → 下发 model_update: 下载新权重
  │
  └── 将决策转化为任务 → 放入该设备的任务队列（等待边端拉取）
```

**自适应频率逻辑**：
```
稳定状态（CPU/内存/网络变化 < 20%）→ 每 30 秒上报一次
波动状态（CPU 变化 > 20% / 内存变化 > 10% / RTT > 500ms / 网络可达性变化）
  → 每 5 秒上报一次（让云端更快感知变化）
```

---

### 通道三：任务拉取（后台自动）

**边端定期轮询云端，看有没有给自己下发的任务指令。**

```
触发条件: 心跳启用后自动关联（TaskScheduler 后台线程）
轮询频率: 每 2 秒
通信协议: HTTP GET (JSON)
API 端点: GET /api/v1/edge/tasks?device_id=X
请求方向: 边端 → 云端（拉取）
管理模块: task_scheduler.cpp — TaskScheduler 类（独立后台线程）

代码位置:
  task_scheduler.cpp:25  — start() 启动后台轮询线程
  task_scheduler.cpp:85  — poll_cloud_tasks() HTTP GET 拉取
  task_scheduler.cpp:53  — enqueue() 按优先级排序入队
  task_scheduler.cpp:68  — process_queue() 逐任务分发执行
```

**云端返回格式**：
```json
[
  {
    "type": "cloud_review",
    "task_id": "task-20260720-001",
    "description": "边端内存不足，卸载推理到云端",
    "priority": 5,
    "payload": { "max_tokens": 512, "temperature": 0.7, "route": "cloud" }
  },
  {
    "type": "rule_sync",
    "task_id": "task-20260720-002",
    "description": "网络波动，降低异常检测置信度阈值",
    "priority": 4,
    "payload": {
      "rule_id": "fluctuation_adjust",
      "rule_type": "threshold",
      "target": "anomaly_score",
      "value": { "min": 0.75, "max": 1.0 }
    }
  }
]
```

**四种任务类型及边端处理方式**：

| 任务类型 | 含义 | 边端收到后做什么 | 处理模块 |
|---------|------|-----------------|---------|
| `cloud_review` | 云端复核 | 把当前对话上下文发给云端全量模型推理，结果与本地融合 | EdgeCloudClient::query_offload() |
| `upload_features` | 上传特征 | 提取本地模型的中间层特征向量，上传云端做二次分析 | EdgeCloudClient::feature_offload() |
| `model_update` | 模型更新 | 下载新 LoRA 文件 → `llama_adapter_lora_init()` → 热生效，无需重启 | ModelSyncManager::download_and_apply_lora() |
| `rule_sync` | 规则同步 | 更新本地决策规则/阈值 → ConflictDetector 和 EdgeLocalDecision 即时读取新值 | ModelSyncManager::apply_rule() |

**任务执行与结果融合**（`TaskScheduler::fuse()`）：

```
本地推理结果 (confidence=0.72)  +  云端复核结果 (confidence=0.91)
        │                                    │
        └──────────── 加权融合 ───────────────┘
                     │
        ┌────────────▼────────────┐
        │ 云端置信度 > 本地 × 1.5  │
        │ → 采用云端结果           │
        │ → 最终输出: 云端答案     │
        └─────────────────────────┘
```

---

### 通道四：模型/规则同步（后台按需）

**云端有新的模型权重或决策规则时，边端自动下载并热加载。**

```
触发条件: 云端通过任务下发 model_update 或 rule_sync
通信协议: HTTP GET (JSON)
API 端点: GET /api/v1/edge/model-updates?device_id=X&current_lora_version=N
请求方向: 边端 → 云端（查询）
管理模块: model_sync.cpp — ModelSyncManager 类

代码位置:
  model_sync.cpp:158 — poll_updates() 查询最新 LoRA/规则
  model_sync.cpp:56  — download_and_apply_lora() 下载+热加载
  model_sync.cpp:137 — apply_rule() 即时更新本地规则
```

**云端返回格式**：
```json
{
  "lora_updates": [
    {
      "update_id": "lora-quality-v3",
      "url": "http://cloud/models/loras/quality_v3.gguf",
      "scale": 1.0,
      "version": 3,
      "checksum": "a1b2c3d4"
    }
  ],
  "rule_updates": [
    {
      "rule_id": "anomaly_threshold_v2",
      "rule_type": "threshold",
      "target": "anomaly_score",
      "value": { "min": 0.85, "max": 1.0 },
      "version": 2
    }
  ]
}
```

**LoRA 热加载流程**（不重启模型）：
```
云端下发 model_update 任务
  │
  ├── 边端 GET /api/v1/edge/model-updates 发现新版本
  │
  ├── HTTP 下载 .gguf LoRA 文件到本地
  │
  ├── llama_adapter_lora_init(model, lora_path)
  │     └── 创建 LoRA adapter 对象
  │
  ├── llama_set_adapters_lora(ctx, adapters, n, scales)
  │     └── 注入到推理上下文
  │
  └── 下一次 llama_decode() 自动应用 LoRA 权重
        └── 零停机，用户无感知
```

**规则即时生效流程**：
```
云端下发 rule_sync 任务
  │
  ├── ModelSyncManager::apply_rule(rule)
  │     └── 写入 _active_rules 列表
  │
  ├── ConflictDetector 读取 _active_rules
  │     └── 四级冲突消解使用最新权重/阈值
  │
  └── EdgeLocalDecision::update_cloud_params()
        └── 离线时使用的"锦囊"参数同步更新
```

---

### 通道五：P2P 邻居发现（云端辅助）

**边端节点之间通过 P2P 直接通信，云端辅助跨子网节点发现。**

```
触发条件: 用户在 P2P 面板启用 + 配置节点 ID 和端口
通信协议: UDP 广播（同子网发现）+ HTTP GET（云端辅助发现）
API 端点: GET /api/v1/edge/neighbors?device_id=X
请求方向: 边端 → 云端（查询邻居列表）
管理模块: p2p_mesh.cpp — P2PMesh 类

代码位置:
  p2p_mesh.cpp:191  — udp_broadcast_presence() UDP 广播 "EDGE_DISCOVER|node_id|port"
  p2p_mesh.cpp:134  — cloud_discover() HTTP GET 云端邻居查询
  p2p_mesh.cpp:213  — udp_listen_loop() 接收其他节点的广播
  p2p_mesh.cpp:300  — tcp_listen_loop() 接收 TCP JSON 感知报告
```

**两种发现方式**：

| 方式 | 适用场景 | 格式 | 端口 |
|------|---------|------|------|
| UDP 广播 | 同一子网内 | `EDGE_DISCOVER\|node_id\|tcp_port` 纯文本 | 15555 |
| 云端查询 | 跨子网 / UDP 不可达 | `GET /api/v1/edge/neighbors?device_id=X` → JSON | HTTP |

**云端返回邻居列表**：
```json
[
  {
    "node_id": "edge-02",
    "ip": "192.168.1.102",
    "udp_port": 15555,
    "tcp_port": 15556,
    "device_type": "jetson-orin",
    "monitored_asset": "production_line_A",
    "camera_fov": "camera_3_conveyor_belt"
  }
]
```

**P2P 通信内容**：
- **感知报告** (PerceptionReport): UDP 广播 + TCP 可靠传输，含检测目标、分类、置信度、决策
- **决策意图** (DecisionIntent): TCP 单播给特定节点，含提议动作、置信度、推理依据

---

### 通道六：冲突升级（仅本地消解失败时）

**多个边端节点对同一目标判断不一致时，先本地四级消解，失败后提交云端仲裁。**

```
触发条件: ConflictDetector 检测到冲突 + 四级本地规则全部失败
通信协议: HTTP POST (JSON)
API 端点: POST /api/v1/edge/conflicts
请求方向: 边端 → 云端（升级仲裁）
管理模块: conflict_detector.cpp — ConflictDetector 类

代码位置:
  conflict_detector.cpp:140 — check_all_conflicts() 遍历本地+邻居检测冲突
  conflict_detector.cpp:204 — resolve_locally() 四级规则逐一尝试
  conflict_detector.cpp:335 — escalate_to_cloud() 升级云端仲裁
```

**四级本地消解**（按优先级尝试）：

| 规则 | 逻辑 | 示例 |
|------|------|------|
| 1. 置信度差距 | 两边置信度差距 > 30%，高者胜 | 我方 0.90 vs 对方 0.50 → 我方胜 |
| 2. 传感器权威 | 传感器类型权重差距 > 20%，高者胜 | LiDAR(1.0) vs Camera(0.7) → LiDAR 胜 |
| 3. 邻近优先 | 距离目标更近的节点胜 | 待实现（需要空间位置数据） |
| 4. 多数共识 | ≥3 个节点，多数意见胜 | 3 个节点 2:1 → 多数胜 |

**全部失败后升级云端**：
```json
POST /api/v1/edge/conflicts
{
  "conflict_id": "target_001_vs_edge-02",
  "target_id": "target_001",
  "target_type": "defect",
  "our_decision": "critical",   "our_confidence": 0.72,
  "peer_decision": "normal",    "peer_confidence": 0.85,
  "reason": "class_mismatch",
  "severity": 3
}
```

**云端仲裁逻辑**（三级级联）：
```
1. 置信度差距 > 40% → 高置信度方胜
2. 任一方判定 "critical" → 安全优先，判定为 critical  
3. 其他情况 → 高置信度方胜
```

---

### 通道七：协同推理（按需触发）

**边端先跑本地轻量模型、后台异步送云端全量模型复核。Split Inference（层切分）代码已就绪但作为预留。**

```
触发条件: 云端通过任务下发 cloud_review 或 upload_features
通信协议: HTTP POST + SSE 流式
API 端点: POST /api/v1/infer/split（隐藏状态）或 /api/v1/infer/features（特征向量）
请求方向: 边端 → 云端（发送中间结果）
响应方式: SSE 流式返回推理结果
管理模块: edge_cloud.cpp — EdgeCloudClient 类

代码位置:
  edge_cloud.cpp:154 — split_infer() 发送隐藏状态
  edge_cloud.cpp:165 — query_offload() 发送文本上下文
  edge_cloud.cpp:196 — feature_offload() 发送多模态特征
```

**三种协同模式**：

| 模式 | 边端发送什么 | 数据量 | 云端做什么 |
|------|------------|--------|-----------|
| Query Offload (主力) | 对话文本 JSON（OpenAI 格式） | < 10 KB | 全量模型独立推理 |
| Feature Offload | 视觉特征向量 (Base64) | ~260 KB | 用特征继续视觉分析 |
| Split Inference (预留) | 中间层 hidden states (Base64 FP16) | ~2.7 MB | 从第 K+1 层继续推理 |

**当前策略**：
```
用户输入
  │
  ├── 1. 边端本地推理（毫秒级，即时返回给用户）
  │
  ├── 2. 后台异步判断：置信度 < 云端阈值？
  │     └── YES → 触发 Query Offload → 云端大模型推理
  │
  └── 3. 云端返回后 TaskScheduler.fuse() 融合
        └── 云端结果更可信 → 静默更新显示
        └── 本地结果更可信 → 保持不变
```

---

## 三、全部 API 端点速查表

| # | 方法 | 端点 | 通道 | 方向 | 频率 | 触发方式 |
|---|------|------|------|------|------|---------|
| ① | GET | `/api/v1/health` | — | 边→云 | 手动 | "测试连接"按钮 |
| ② | POST | `/api/v1/edge/heartbeat` | 二 | 边→云 | 5~30s | 心跳面板启用后自动 |
| ③ | GET | `/api/v1/edge/tasks?device_id=X` | 三 | 边→云 | 2s | 心跳启用后自动轮询 |
| ④ | POST | `/api/v1/infer/split` | 七 | 边→云 | 按需 | 云端下发任务后执行 |
| ⑤ | POST | `/api/v1/infer/features` | 七 | 边→云 | 按需 | 云端下发任务后执行 |
| ⑥ | POST | `/v1/chat/completions` | 一/七 | 边→云 | 按需 | 用户发消息或云端下发任务 |
| ⑦ | GET | `/api/v1/edge/model-updates` | 四 | 边→云 | 按需 | 云端下发 model_update 任务后 |
| ⑧ | GET | `/api/v1/edge/neighbors?device_id=X` | 五 | 边→云 | 按需 | P2P 启用后自动 |
| ⑨ | POST | `/api/v1/edge/conflicts` | 六 | 边→云 | 仅冲突时 | 四级本地消解全部失败后 |

---

## 四、"启用云端推理" 勾选框 vs 其他功能

**一个常见的误解**：以为"启用云端推理"是云边协同的总开关。

**实际情况**：

```
"启用云端推理" 勾选框 (_cloud_enabled)
  │
  └── 只控制通道一: 用户发消息时走本地还是走云端
       └── 不影响: 心跳、任务拉取、模型同步、P2P、冲突消解

心跳面板 (_show_heartbeat)
  │
  └── 只控制通道二: 是否向云端上报设备状态
       └── 启动后自动连带激活通道三（任务拉取）

P2P 面板 (_show_peers)
  │
  └── 只控制通道五: 是否与其他边端节点通信
       └── 启动后自动连带激活通道五的云端邻居发现

冲突检测
  │
  └── 始终在后台运行
       └── 检测到冲突 + 本地无法消解 → 自动触发通道六
```

**每条通道都是独立的**，在各自的 UI 面板中启停，互不依赖。

---

## 五、网络断开时的降级逻辑

```
网络正常时:
  ├── 通道一: 用户可选本地或云端
  ├── 通道二: 心跳正常上报 (30s)
  ├── 通道三: 任务正常拉取 (2s)
  ├── 通道四: 模型更新正常查询
  ├── 通道五: P2P 正常发现+通信
  └── 通道六: 冲突可升级云端

网络弱化时 (RTT 200-500ms, [net:degraded]):
  ├── 通道一: 边端自主决策 → 高置信度本地 / 低置信度排队
  ├── 通道二: 心跳频率自动提升到 5s
  └── 其他通道: 继续尝试，超时自动跳过

网络断开时 (RTT > 500ms, [net:offline]):
  ├── 通道一: 强制纯本地推理
  ├── 通道二: 心跳自动停止（连续失败 5 次后）
  ├── 通道三: 任务拉取自动跳过
  ├── 通道五: P2P 继续工作（UDP+TCP 不依赖云端）
  ├── 通道六: 冲突降级为仅本地消解（P2P 共识替代云端仲裁）
  └── 所有决策记录写入离线日志，网络恢复后上传云端

网络恢复时:
  ├── 自动重新连接
  ├── flush_offline_decisions() → 上传离线期间的决策日志
  └── 恢复云端参数同步 → 切回正常模式
```

---

## 六、代码模块对照表

| 模块 | 文件 | 管理哪条通道 | 线程 |
|------|------|-------------|------|
| EdgeCloud | `cloud.h/cpp` | 通道一（消息卸载） | 每请求一线程 |
| HeartbeatManager | `heartbeat.h/cpp` | 通道二（心跳上报） | 持久后台线程 |
| TaskScheduler | `task_scheduler.h/cpp` | 通道三（任务拉取） | 持久后台线程 |
| ModelSyncManager | `model_sync.h/cpp` | 通道四（模型同步） | 无独立线程（由任务触发） |
| P2PMesh | `p2p_mesh.h/cpp` | 通道五（邻居发现+P2P通信） | UDP + TCP 两个持久线程 |
| ConflictDetector | `conflict_detector.h/cpp` | 通道六（冲突升级） | 无独立线程（由回调触发） |
| EdgeCloudClient | `edge_cloud.h/cpp` | 通道七（协同推理） | 每请求一线程 |
| EdgeLocalDecision | `edge_local_decision.h/cpp` | 网络状态判断 + 离线决策 | 无独立线程（在 submit_message 中同步调用） |
| DeviceMonitor | `device_monitor.h/cpp` | 资源数据采集（供心跳使用） | 无独立线程（被心跳线程调用） |
