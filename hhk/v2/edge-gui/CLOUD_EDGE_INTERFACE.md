# 边端云端协作 — 实现总结与接口文档

## 一、完成了什么

基于 `D:\llama.cpp\edge-gui\` 现有代码，新增 **11 个文件**（约 2700 行 C++17 代码），修改 4 个文件，实现了比赛方案中**模块二（云边协同通信协议与调度接口）**和**模块三（边缘冲突检测与局部一致性维护）**的全部核心能力。

### 新增模块一览

```
edge-gui/
├── edge_cloud.cpp          # EdgeCloudClient 实现（Split/Query/Feature 三种推理模式）
├── device_monitor.h/cpp    # 设备资源采集（CPU/GPU/Memory/Network）
├── heartbeat.h/cpp         # 自适应心跳上报（稳定30s / 波动5s）
├── task_scheduler.h/cpp    # 任务调度器（拉取→分发→融合）
├── model_sync.h/cpp        # LoRA 热加载 + 决策规则同步
├── p2p_mesh.h/cpp          # P2P 网格通信（UDP 发现 + TCP 交换）
├── conflict_detector.h/cpp # 冲突检测与四级消解
├── engine.h/cpp  (修改)     # + LoRA 热加载 + 模型层数查询
├── app.h/cpp     (修改)     # + 7 个子系统集成 + 4 个新 UI 面板
└── CMakeLists.txt(修改)     # + 11 个源文件
```

### 模块能力清单

| 模块 | 能力 | 实现方式 |
|------|------|---------|
| **心跳上报** | CPU/GPU/Memory/Network/TPS 定期上报 | `DeviceMonitor` 采集 → `HeartbeatManager` JSON POST |
| **任务分拆** | 接收云端指令，上传特征向量，融合结果 | `TaskScheduler` 拉取任务 → `EdgeCloudClient` 执行 → 加权融合 |
| **模型同步** | LoRA 热加载 + 决策规则即时生效 | `llama_adapter_lora_init` + `llama_set_adapters_lora` |
| **P2P 协调** | 边端节点发现 + 感知报告交换 + 决策意图传输 | UDP 广播发现 + TCP 可靠传输 |
| **冲突检测** | 分类/置信度/动作三维冲突检测 | 逐 target_id 比对 → 4 级规则消解 → 云端仲裁 |

---

## 二、边端→云端 传输的数据类型

边端向云端发送 **5 类** 数据，每一类对应不同的上传目的和带宽特征：

### 类型 1：心跳状态上报 (Heartbeat)

**频率**：稳定 30s / 波动 5s（自适应）  
**方向**：边端 → 云端  
**用途**：云端调度器据此感知各边缘节点实时状态，做出任务路由决策

```json
{
  "device_id": "edge-node-01",
  "device_type": "jetson-orin-nano",
  "timestamp_ms": 1721200000000,
  "is_fluctuating": false,
  "cpu": {
    "usage_pct": 45.2,
    "core_count": 8,
    "freq_mhz": 0
  },
  "memory": {
    "total_mb": 8192,
    "avail_mb": 3200
  },
  "gpus": [{
    "name": "NVIDIA Orin",
    "total_mb": 4096,
    "free_mb": 2100,
    "used_mb": 1996,
    "device_type": 1
  }],
  "network": {
    "rtt_ms": 12,
    "packet_loss_pct": 0.0,
    "cloud_reachable": true
  },
  "inference": {
    "current_tps": 45.3,
    "task_queue_len": 2
  }
}
```

| 字段 | 说明 | 采集方式 |
|------|------|---------|
| `cpu.usage_pct` | CPU 使用率 (0-100) | Windows: `GetSystemTimes()` |
| `memory.*` | 系统内存 | `GlobalMemoryStatusEx()` |
| `gpus[*].*` | GPU 显存 | `ggml_backend_dev_memory()` |
| `network.rtt_ms` | 到云端的往返延迟 | HTTP GET 测速 |
| `inference.current_tps` | 当前推理速度 (token/s) | `llama_perf_context()` |

### 类型 2：Split Inference — 中间层隐藏状态

**频率**：按需（云端决策需要时才发送）  
**方向**：边端 → 云端  
**用途**：边端跑前 K 层后，将中间 hidden states 发给云端继续推理  
**特点**：这是**带宽最大的上传类型**，hidden states 体积 = `batch × seq_len × hidden_dim × dtype_bytes`

```json
{
  "model": "qwen3-1.7b",
  "hidden": {
    "split_layer": 12,
    "current_heads": 16,
    "head_dim": 128,
    "batch_size": 1,
    "seq_len": 512,
    "dtype": "float16",
    "data_b64": "AAAA...base64编码的连续内存块...",
    "data_bytes": 2097152,
    "checksum": "a1b2c3d4e5f6a7b8"
  },
  "context": {
    "n_past": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_new_tokens": 256,
    "prompt_tokens_b64": "...",
    "n_prompt_tokens": 512
  },
  "device": {
    "device_id": "edge-01",
    "avail_mem_mb": 3200,
    "current_tps": 45.3,
    "network_rtt_ms": 12
  }
}
```

**数据量估算**（Qwen3-1.7B, seq_len=512, FP16）：
- `hidden_dim = 2048`, `dtype_bytes = 2`
- 单次上传 = `1 × 512 × 2048 × 2` = **2 MB**（base64 编码后约 **2.7 MB**）

### 类型 3：Feature Offload — 多模态特征向量

**频率**：按需（工业视觉检测等场景）  
**方向**：边端 → 云端  
**用途**：将边端提取的视觉特征（如 Vision Encoder 输出）发送给云端做二次分析  
**特点**：**轻量级**——只传特征，不传原始图片/视频帧

```json
{
  "prompt": "分析这张产线图片中的缺陷类型",
  "model": "qwen2-vl-7b",
  "temperature": 0.7,
  "max_tokens": 256,
  "stream": true,
  "media": {
    "type": "feature_vector",
    "width": 224,
    "height": 224,
    "channels": 3,
    "mime": "image/jpeg",
    "feature_shape": [49, 1024],
    "features_b64": "AAAA...base64编码的float32数组...",
    "timestamp_us": 1721200000000
  }
}
```

**数据量估算**（ViT patch features, 49 patches × 1024 dim, FP32）：
- 特征体积 = `49 × 1024 × 4` = **196 KB**（base64 后约 **262 KB**）
- 对比直接上传 224×224 JPEG（~50 KB），特征向量虽然更大，但**保护了原始数据隐私**，且云端可直接推理

### 类型 4：Query Offload — 纯文本卸载

**频率**：按需（边端算力不足或云端模型更强时）  
**方向**：边端 → 云端  
**用途**：将对话上下文发给云端全量大模型处理（OpenAI 兼容格式）  
**特点**：**复用已有 `EdgeCloud` 逻辑**，增加推理上下文保持采样一致性

```json
{
  "model": "deepseek-v2",
  "messages": [
    {"role": "system", "content": "你是一个工业质检助手"},
    {"role": "user", "content": "这台设备的振动频率异常，判断是否需要停机检修？"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 512,
  "seed": 42,
  "stream": true
}
```

### 类型 5：冲突升级 (Conflict Escalation)

**频率**：仅当本地四级消解规则均无法解决时  
**方向**：边端 → 云端  
**用途**：请求云端全局仲裁  
**特点**：**极小体积**，仅包含冲突关键字段

```json
{
  "conflict_id": "target_001_vs_edge-02",
  "target_id": "target_001",
  "target_type": "defect",
  "our_node_id": "edge-01",
  "our_decision": "critical",
  "our_confidence": 0.72,
  "peer_node_id": "edge-02",
  "peer_decision": "normal",
  "peer_confidence": 0.85,
  "reason": "class_mismatch",
  "severity": 3,
  "timestamp_ms": 1721200000000
}
```

---

## 三、云端 API 接口定义

边端作为 HTTP 客户端，调用云端以下接口：

### 3.1 健康检查

```
GET {base_url}/api/v1/health
```

**响应**：
```json
{
  "reachable": true,
  "latency_ms": 8,
  "load": 35,
  "version": "1.2.0",
  "max_split_layer": 28
}
```

### 3.2 心跳上报

```
POST {base_url}/api/v1/edge/heartbeat
Content-Type: application/json
Body: HeartbeatPacket (见 2.1)
```

**响应**：`200 OK` 或 `503 Service Unavailable`

### 3.3 任务拉取

```
GET {base_url}/api/v1/edge/tasks?device_id=edge-01
```

**响应**：
```json
[
  {
    "type": "cloud_review",
    "task_id": "task-20260717-001",
    "description": "二次分析产线图像特征",
    "priority": 5,
    "payload": {
      "messages": [{"role": "user", "content": "..."}],
      "temperature": 0.7,
      "max_tokens": 512
    }
  },
  {
    "type": "model_update",
    "task_id": "task-20260717-002",
    "description": "更新质检LoRA权重",
    "priority": 3,
    "payload": {
      "url": "http://cloud/models/lora/quality_v3.gguf",
      "version": 3,
      "scale": 1.0
    }
  }
]
```

**task_type 枚举**：

| 值 | 含义 | 边端处理方式 |
|----|------|------------|
| `cloud_review` | 云端复核 | `EdgeCloudClient::query_offload()` |
| `upload_features` | 上传特征 | `EdgeCloudClient::feature_offload()` |
| `model_update` | 模型更新 | `ModelSyncManager::download_and_apply_lora()` |
| `rule_sync` | 规则同步 | `ModelSyncManager::apply_rule()` |

### 3.4 Split Inference

```
POST {base_url}/api/v1/infer/split
Content-Type: application/json
Body: HiddenStates + InferenceContext (见 2.2)
Accept: text/event-stream
```

**响应**：SSE 流式，与 OpenAI Chat Completions 格式兼容：
```
data: {"choices":[{"delta":{"content":"异"}, "finish_reason":null}]}
data: {"choices":[{"delta":{"content":"常"}, "finish_reason":null}]}
data: {"choices":[{"delta":{"content":""}, "finish_reason":"stop"}], "usage":{"total_tokens":128}}
data: [DONE]
```

### 3.5 特征上传

```
POST {base_url}/api/v1/infer/features
Content-Type: application/json
Body: MultimodalFeatures (见 2.3)
Accept: text/event-stream
```

**响应**：同 SSE 流式

### 3.6 模型更新拉取

```
GET {base_url}/api/v1/edge/model-updates?device_id=edge-01&current_lora_version=2
```

**响应**：
```json
{
  "lora_updates": [
    {
      "update_id": "lora-quality-v3",
      "url": "http://cloud/models/loras/quality_v3.gguf",
      "local_path": "",
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
      "value": {"min": 0.85, "max": 1.0},
      "version": 2,
      "effective_at_ms": 0
    }
  ]
}
```

### 3.7 邻居发现

```
GET {base_url}/api/v1/edge/neighbors?device_id=edge-01
```

**响应**：
```json
[
  {
    "node_id": "edge-02",
    "ip": "192.168.1.102",
    "udp_port": 15555,
    "tcp_port": 15556,
    "device_type": "jetson-orin",
    "asset": "production_line_A",
    "fov": "camera_3_conveyor_belt"
  }
]
```

---

## 四、传输逻辑详解

### 4.1 心跳上报流程

```
┌──────────┐                    ┌──────────┐
│ 边端节点  │                    │ 云端调度器 │
└────┬─────┘                    └────┬─────┘
     │                               │
     │  DeviceMonitor.snapshot()     │
     │  ├─ CPU: GetSystemTimes()     │
     │  ├─ GPU: ggml_backend_dev_    │
     │  │       memory()             │
     │  ├─ RAM: GlobalMemoryStatus   │
     │  └─ NET: HTTP RTT probe       │
     │                               │
     │  is_fluctuating(当前, 上次)?   │
     │  ├─ 变化>20% → 间隔=5s        │
     │  └─ 稳定     → 间隔=30s       │
     │                               │
     │  POST /api/v1/edge/heartbeat  │
     │  ─────────────────────────────▶
     │                               
     │                    200 OK / 503
     │  ◀─────────────────────────────
     │                               │
     │  sleep(interval_sec)          │
     │  ──→ 循环                      │
```

### 4.2 云边协同推理流程（以"工业质检云复核"为例）

```
┌──────────┐         ┌──────────┐         ┌──────────┐
│ 边端推理  │         │ TaskSched │         │ 云端大模型 │
│ (Qwen3   │         │ uler      │         │ (全量)    │
│  1.7B)   │         │           │         │           │
└────┬─────┘         └────┬─────┘         └────┬─────┘
     │                    │                    │
     │ 1. 本地推理完成      │                    │
     │ confidence=0.65    │                    │
     │ (低于阈值0.85)      │                    │
     │                    │                    │
     │ 2. request_cloud_  │                    │
     │    review() ──────▶│                    │
     │                    │                    │
     │                    │ 3. query_offload() │
     │                    │   带上下文+温度参数  │
     │                    │ ──────────────────▶│
     │                    │                    │ 4. 全量大模型推理
     │                    │                    │ confidence=0.92
     │                    │                    │ "疑似裂纹缺陷"
     │                    │                    │
     │                    │ 5. SSE 流式返回     │
     │                    │ ◀──────────────────│
     │                    │                    │
     │ 6. fuse_results()  │                    │
     │    local(0.65) +   │                    │
     │    cloud(0.92)     │                    │
     │    → 采用云端结果   │                    │
     │    final="裂纹缺陷" │                    │
```

### 4.3 冲突检测与消解流程

```
         ┌─────────────────┐
         │   本地感知结果    │
         │  target_001:     │
         │  class="缺陷"     │
         │  decision=critical│
         │  conf=0.72       │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ P2P 接收对等感知  │
         │ edge-02 报告:    │
         │ target_001:      │
         │ class="正常"      │
         │ decision=normal  │
         │ conf=0.85        │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ 冲突检测          │
         │ class不匹配 →     │
         │ 创建ConflictRecord│
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ 规则1: 置信度差距  │  gap=|0.72-0.85|=0.13
         │ margin=0.3       │  0.13 < 0.3 → 无法解决 ✗
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ 规则2: 传感器权威  │  双方都是camera
         │ camera=0.7       │  权重相同 → 无法解决 ✗
         │ camera=0.7       │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ 规则3: 邻近优先   │  无位置数据 → 跳过 ✗
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ 规则4: 多数共识   │  仅2个节点 → 无法投票 ✗
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  升级云端仲裁     │
         │  POST 冲突上下文  │ ────▶ 云端全局仲裁
         │  接收仲裁结果     │ ◀──── "确认为缺陷，置信度0.91"
         │  resolution=cloud│
         └─────────────────┘
```

### 4.4 P2P 发现与通信机制

```
时间轴 →

边端节点 A (edge-01)                    边端节点 B (edge-02)
      │                                        │
      │  UDP广播: "EDGE_DISCOVER|edge-01|15556"│
      │  255.255.255.255:15555                 │
      │ ──────────────────────────────────────▶│
      │                                        │ 解析: 发现 edge-01
      │                                        │ 加入已知对等列表
      │                                        │
      │  UDP广播: "EDGE_DISCOVER|edge-02|15556"│
      │ ◀──────────────────────────────────────│
      │ 解析: 发现 edge-02                      │
      │ 加入已知对等列表                         │
      │                                        │
      │  TCP: connect edge-02:15556             │
      │ ──────────────────────────────────────▶│
      │  发送: PerceptionReport JSON +\n        │
      │ ──────────────────────────────────────▶│
      │                                        │ 解析 → conflict detector
      │                                        │
      │  TCP: connect edge-01:15556             │
      │ ◀──────────────────────────────────────│
      │  接收: PerceptionReport JSON +\n        │
      │ ◀──────────────────────────────────────│
      │ 解析 → conflict detector                │
      │                                        │
      │  [每5秒重复UDP广播保活]                   │
```

### 4.5 数据流向总览

```
边端节点
├── 心跳上报 ────────────────────────▶ 云端 /api/v1/edge/heartbeat
│   (30s稳定/5s波动, JSON, <1KB)
│
├── 任务拉取 ◀──────────────────────── 云端 /api/v1/edge/tasks
│   (2s轮询, 返回待执行任务列表)
│
├── Split推理 ───────────────────────▶ 云端 /api/v1/infer/split
│   (按需, 发送隐藏状态 ~2.7MB)
│
├── 特征上传 ───────────────────────▶ 云端 /api/v1/infer/features
│   (按需, 发送特征向量 ~200KB)
│
├── 文本卸载 ───────────────────────▶ 云端 /v1/chat/completions
│   (按需, 发送对话上下文 ~几KB)
│
├── 冲突升级 ───────────────────────▶ 云端 /api/v1/edge/conflicts
│   (仅冲突无法本地消解时, <1KB)
│
├── 模型更新拉取 ◀──────────────────── 云端 /api/v1/edge/model-updates
│   (定期检查LoRA/规则更新)
│
├── 邻居发现 ◀─────────────────────── 云端 /api/v1/edge/neighbors
│   (P2P补充, 获取对等节点列表)
│
└── P2P 对等通信 ◀─────────────────▶ 其他边端节点
    (UDP广播发现 + TCP可靠传输, 局域网)
```

---

## 五、关键设计决策

| 决策 | 理由 |
|------|------|
| 零外部依赖 | 所有库均由 llama.cpp 项目提供（cpp-httplib, nlohmann/json, ggml, stb_image） |
| P2P 使用原生 Socket | UDP + TCP ~200 行即可实现，无需引入 libp2p/ZeroMQ 等重量级框架 |
| LoRA 热加载而非全量替换 | `llama_set_adapters_lora()` 在下次 `llama_decode` 时自动注入，无需重启 |
| 冲突消解用规则而非 ML | 确定性、可验证、云端可动态更新规则参数 |
| 心跳自适应频率 | 避免波动期间信息滞后，同时减少稳态带宽浪费 |
| Split Inference 数据用 Base64 | JSON 兼容所有 HTTP 中间件，虽增大约 33% 体积但保证了通用性 |

---

## 六、数据传输的格式与形式（本项目实现细节）

本章说明项目中数据从 C++ 结构体到 TCP 字节流的完整转换链路，以及三种传输通道在代码层面的具体格式。

### 6.1 通道一：HTTP JSON（边端 → 云端），短连接 request/response

这是边端主动向云端发起的请求，使用 `cpp-httplib` 库。每条消息的转换链为：

```
C++ 结构体 ──(nlohmann::json 逐字段赋值)──▶ JSON 对象 ──(.dump())──▶ std::string ──(cli.Post)──▶ HTTP Body ──▶ TCP 字节流
```

#### 6.1.1 心跳上报的完整转换

**代码位置**：[heartbeat.cpp](heartbeat.cpp) — `build_report_json()` + `send_heartbeat()`

**第 1 步 — C++ 结构体采集**（`heartbeat.cpp` line 67-76）：
```cpp
DeviceSnapshot snap = _monitor.snapshot();
// snap.cpu.usage_pct  →  45.2
// snap.cpu.core_count →  8
// snap.total_ram_mb   →  8192
// snap.avail_ram_mb   →  3200
// snap.gpus[0].name   →  "NVIDIA Orin"
// snap.gpus[0].free_mem_mb → 2100
```

**第 2 步 — JSON 构造**（`heartbeat.cpp:build_report_json()`）：
```cpp
nlohmann::json j;
j["device_id"]      = _cfg.device_id;
j["device_type"]    = _cfg.device_type;
j["timestamp_ms"]   = snap.timestamp_us / 1000;
j["cpu"] = {
    {"usage_pct",  snap.cpu.usage_pct},
    {"core_count", snap.cpu.core_count},
};
j["memory"] = {
    {"total_mb", snap.total_ram_mb},
    {"avail_mb", snap.avail_ram_mb},
};
// ... GPU、Network、Inference 同理逐字段填充
std::string body = j.dump();   // 序列化为字符串
```

`j.dump()` 生成的字符串就是标准的 JSON 文本，全为可读 ASCII 字符，无二进制字节。

**第 3 步 — HTTP POST 发送**（`heartbeat.cpp:send_heartbeat()`）：
```cpp
auto [cli, parts] = common_http_client(_cfg.cloud_endpoint);
// common_http_client 来自 common/http.h，内部调用 httplib::Client 构造函数

httplib::Headers headers = {{"Content-Type", "application/json"}};
auto res = cli.Post(parts.path, headers, body, "application/json");
```

**第 4 步 — 线路上实际传输的字节**：

```http
POST /api/v1/edge/heartbeat HTTP/1.1\r\n
Host: 192.168.1.100:8080\r\n
Content-Type: application/json\r\n
Content-Length: 387\r\n
\r\n
{"cpu":{"core_count":8,"usage_pct":45.2},"device_id":"edge-node-01","device_type":"jetson-orin-nano","gpus":[{"device_type":1,"free_mb":2100,"name":"NVIDIA Orin","total_mb":4096,"used_mb":1996}],"inference":{"current_tps":45.3,"task_queue_len":2},"is_fluctuating":false,"memory":{"avail_mb":3200,"total_mb":8192},"network":{"cloud_reachable":true,"rtt_ms":12},"timestamp_ms":1721200000000}
```

- HTTP 头部 + 空行 (`\r\n\r\n`) 是 HTTP/1.1 协议规定的分隔
- Body 部分就是紧凑 JSON（无缩进，无换行）
- 整个请求在 TCP 层面是一个连续的字节流
- 云端收到后，`Content-Length: 387` 告诉它 Body 刚好 387 字节，读完即止

#### 6.1.2 包含二进制数据时的处理：Base64 嵌入

当 JSON 中需要携带**非文本**数据（如 FP16 隐藏状态）时，不能直接放入 JSON 字符串（JSON 禁止未转义的控制字符和二进制字节）。本项目使用 **Base64 编码** 将任意二进制 blob 转为 ASCII 安全字符。

**代码位置**：[edge_cloud.cpp](edge_cloud.cpp) — `base64_encode()` + `build_split_request_json()`

**Base64 编码函数**（`edge_cloud.cpp` 内部静态函数）：
```cpp
static std::string base64_encode(const uint8_t * data, size_t len) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    // 核心规则：每 3 个原始字节 → 4 个 Base64 字符
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= (uint32_t)data[i + 2];
        out.push_back(table[(v >> 18) & 0x3f]);  // 取高 6 位
        out.push_back(table[(v >> 12) & 0x3f]);  // 取次高 6 位
        out.push_back(table[(v >> 6)  & 0x3f]);  // 取次低 6 位（可能为 '='）
        out.push_back(table[v & 0x3f]);           // 取低 6 位（可能为 '='）
    }
    return out;
}
```

**具体例子**——假设 hidden states 的前 3 个字节是 `0x48 0x65 0x6C`（"Hel"）：

```
原始:   01001000 01100101 01101100
分组:   010010 000110 010101 101100
索引:     18      6      21     44
字符:     S      G      V      s
```

结果是 `"SGVs"` 四个 ASCII 字符。

**在 JSON 中的最终形态**（`build_split_request_json()`）：
```cpp
body["hidden"] = {
    {"split_layer",   hidden.split_layer},
    {"data_b64",      base64_encode(hidden.data.data(), hidden.data.size())},
    {"data_bytes",    hidden.data.size()},
    {"dtype",         dtype_str(hidden.dtype)},
    {"checksum",      fnv_checksum(hidden.data.data(), hidden.data.size())},
};
```

生成的 JSON：
```json
{
  "hidden": {
    "split_layer": 12,
    "data_b64": "SGVsbG8gV29ybGQAAP8B/..."
  }
}
```

`data_b64` 字段是一个纯 ASCII 字符串，可以安全地嵌入 JSON、通过 HTTP 传输、被任何 JSON 解析器处理。

**体积变化**：

| 原始数据 | 原始体积 | Base64 后 | 膨胀 |
|---------|---------|----------|------|
| Hidden States (1×512×2048×FP16) | 2,097,152 B (2 MB) | ~2,796,203 B (~2.67 MB) | +33% |
| 特征向量 (49×1024×FP32) | 200,704 B (~196 KB) | ~267,605 B (~261 KB) | +33% |
| Prompt Token IDs (512×int32) | 2,048 B | ~2,732 B | +33% |

#### 6.1.3 每条 HTTP 请求在代码中的调用链

以"任务拉取"为例，展示从 UI 按钮到 TCP 字节流的完整调用链：

```
用户点击 UI "查询云端任务"
  └─ app.cpp: render_tasks_panel()
       └─ _task_scheduler.poll_cloud_tasks()
            └─ task_scheduler.cpp: poll_cloud_tasks()
                 │
                 │  1. 构造 URL: _cloud_endpoint + "/api/v1/edge/tasks?device_id=" + _device_id
                 │
                 │  2. 调用 common_http_client(url)
                 │     └─ common/http.h  解析 URL → {scheme, host, port, path}
                 │     └─ vendor/cpp-httplib/httplib.h  构造 httplib::Client 对象
                 │
                 │  3. cli.Get(parts.path)
                 │     └─ cpp-httplib 内部:
                 │          socket() → connect() → 发送 HTTP 请求文本 → recv() 读取响应
                 │
                 │  4. 收到 HTTP 响应:
                 │     HTTP/1.1 200 OK
                 │     Content-Type: application/json
                 │     Content-Length: 456
                 │
                 │     [{"type":"cloud_review","task_id":"task-001",...}]
                 │
                 │  5. nlohmann::json::parse(res->body) → CloudTask 结构体
                 │
                 └─ 6. enqueue(task) → 放入 _queue → process_queue() 逐个执行
```

---

### 6.2 通道二：SSE 流式传输（云端 → 边端），长连接分块传输

云端返回推理结果时，不是一次性返回全部文本，而是**逐 token 流式推送**。HTTP 协议使用 `Transfer-Encoding: chunked` + SSE (`text/event-stream`) 实现。

**代码位置**：[edge_cloud.cpp](edge_cloud.cpp) — `run_sse_request()` + `parse_sse_chunk()`

#### 6.2.1 SSE 线路上实际传输的字节

```
HTTP/1.1 200 OK\r\n
Content-Type: text/event-stream\r\n
Transfer-Encoding: chunked\r\n
\r\n
34\r\n                                     ← 块1: 0x34 = 52 字节
data: {"choices":[{"delta":{"content":"异"},"finish_reason":null}]}\n
\n
3A\r\n                                     ← 块2: 0x3A = 58 字节
data: {"choices":[{"delta":{"content":"常"},"finish_reason":null}]}\n
\n
3C\r\n                                     ← 块3: 0x3C = 60 字节
data: {"choices":[{"delta":{"content":"，建议停机"},"finish_reason":null}]}\n
\n
...\r\n
47\r\n
data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"total_tokens":15}}\n
\n
C\r\n                                      ← 块N: 12 字节
data: [DONE]\n
\n
0\r\n\r\n                                  ← 结束标记
```

说明：
- `\r\n` 是 HTTP chunked encoding 的分隔符
- `34`、`3A` 等是十六进制的块长度
- 每个 SSE 事件由 `data: ` 前缀 + JSON + `\n\n` 组成
- `[DONE]` 是 OpenAI 兼容的流结束标记
- `Transfer-Encoding: chunked` 允许 HTTP 服务器一边生成一边发送，不需要提前知道 `Content-Length`

#### 6.2.2 边端解析代码

**`cpp-httplib` 的 ContentReceiver 回调**（`edge_cloud.cpp:run_sse_request()`）：

```cpp
// ContentReceiver: httplib 每收到一块 TCP 数据就回调一次
httplib::ContentReceiver receiver = [&](const char * data, size_t len) -> bool {
    // data 是原始字节块，可能是多个 SSE 事件粘在一起
    // len  是这个块的字节数
    parse_sse_chunk(std::string(data, len), on_token, perf);
    return true;  // 返回 true 继续接收，false 中止连接
};

cli.Post(parts.path, headers, body_json, "application/json", receiver);
```

**SSE 分块解析**（`edge_cloud.cpp:parse_sse_chunk()`）：

```cpp
void EdgeCloudClient::parse_sse_chunk(const std::string & chunk, ...) {
    static std::string leftover;   // 上次未处理完的残留字节
    std::string buf = leftover + chunk;

    size_t pos = 0;
    while (pos < buf.size()) {
        // 1. 查找 "data: " 标记
        auto data_start = buf.find("data: ", pos);
        if (data_start == std::string::npos) {
            leftover = buf.substr(pos);  // 不完整，留给下次
            break;
        }
        data_start += 6;

        // 2. 查找换行符确定本条事件结束
        auto data_end = buf.find('\n', data_start);
        if (data_end == std::string::npos) {
            leftover = buf.substr(pos);
            break;
        }

        // 3. 提取内容
        std::string line = buf.substr(data_start, data_end - data_start);
        pos = data_end + 1;

        // 4. 判断消息类型
        if (line == "[DONE]") {
            on_token("", true, "");   // 通知上层：流结束
            return;
        }

        // 5. 解析 JSON
        auto j = nlohmann::json::parse(line);
        auto content = j["choices"][0]["delta"]["content"];
        if (!content.empty()) {
            on_token(content, false, "");  // 通知上层：又一个 token
        }
    }
}
```

**为什么需要 `static leftover`？** TCP 是流协议，没有消息边界。`ContentReceiver` 每次回调可能收到 1.5 个或 0.3 个 SSE 事件。`leftover` 保存本次未解析完的尾部字节，拼接到下一次回调的数据前面。

#### 6.2.3 SSE vs WebSocket 在本项目中的选择

| 对比维度 | SSE（本项目采用） | WebSocket（cpp-httplib 支持但未用） |
|---------|-----------------|-----------------------------------|
| 方向 | 单向：云端→边端 | 双向 |
| 协议 | HTTP/1.1 长连接 | WebSocket 升级协议 |
| 代码复杂度 | `data: ` 前缀 + `\n\n` 分隔即可 | 需要帧头解析、ping/pong、opcode 处理 |
| 本项目场景 | token 流式输出只需单向推送 | 不需要双向实时交互 |
| 实现行数 | ~40 行（已在 cloud.cpp 验证过） | ~100 行+ |

---

### 6.3 通道三：P2P Socket（边端 ↔ 边端），不走 HTTP

边端节点之间的通信不经过云端，直接使用操作系统原生 Socket API（Windows: Winsock2, Linux: BSD socket）。

**代码位置**：[p2p_mesh.cpp](p2p_mesh.cpp)

#### 6.3.1 UDP 广播 — 节点发现

UDP 是**无连接**的数据报协议。发送方不需要 `connect()`，直接 `sendto()` 到广播地址 `255.255.255.255`。

**格式定义**：纯文本，`|` 分隔字段，不超过 100 字节。

```
EDGE_DISCOVER|edge-01|15556
```

- `EDGE_DISCOVER` — 固定魔术字，区分这是一个发现包而非其他 UDP 数据
- `edge-01` — 发送方节点 ID
- `15556` — 发送方 TCP 监听端口号

**为什么不直接用 JSON？** UDP 广播包应当尽量小（避免 IP 分片，控制在 MTU 1500 字节以内）。纯文本格式比 `{"type":"EDGE_DISCOVER","node_id":"edge-01",...}` 节省约 60% 字节。

**发送代码**（`p2p_mesh.cpp:udp_broadcast_presence()`）：
```cpp
SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);  // 数据报套接字

int broadcast = 1;
setsockopt(s, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

sockaddr_in addr = {};
addr.sin_family      = AF_INET;
addr.sin_port        = htons(15555);           // 网络字节序
addr.sin_addr.s_addr = INADDR_BROADCAST;       // 255.255.255.255

char buf[256];
snprintf(buf, sizeof(buf), "EDGE_DISCOVER|%s|%d", _node_id.c_str(), _tcp_port);

sendto(s, buf, (int)strlen(buf), 0, (const sockaddr*)&addr, sizeof(addr));
// 这一行执行完，操作系统立即将数据封装为 UDP 包发出，无需等回复
```

**接收代码**（`p2p_mesh.cpp:udp_listen_loop()`）：
```cpp
// 先 bind 端口
sockaddr_in addr = {};
addr.sin_family      = AF_INET;
addr.sin_port        = htons(15555);
addr.sin_addr.s_addr = INADDR_ANY;    // 监听所有网卡
bind(s, &addr, sizeof(addr));

// 循环接收
char buf[2048];
sockaddr_in from = {};
socklen_t from_len = sizeof(from);
int n = recvfrom(s, buf, sizeof(buf) - 1, 0, (sockaddr*)&from, &from_len);
buf[n] = '\0';

std::string data(buf, n);

// 根据内容分发
if (data.find("EDGE_DISCOVER|") == 0) {
    // 解析为发现包，提取节点ID和TCP端口
    // 添加到 _peers 列表
    add_peer(peer_id, ip_str, tcp_port);
}
else if (data.find("{\"report_id\"") == 0) {
    // 以 "{" 开头 → 是 JSON 感知报告
    PerceptionReport r = parse_report(data);
    _conflict_detector.receive_peer_perception(r);
}
```

#### 6.3.2 TCP 单播 — 可靠数据传输

感知报告和决策意图需要**可靠到达**，所以用 TCP。本项目使用**短连接模式**：每次发送消息时 `connect() → send() → close()`，不保持长连接。

**格式**：紧凑 JSON（单行，无缩进）+ `\n` 换行符作为消息边界

**为什么用 `\n` 而不是 HTTP？**
- TCP 是纯流协议，没有内置消息边界。发送方连续 `send()` 两次，接收方可能一次 `recv()` 就全收到（粘包）
- HTTP 用 `Content-Length` 或 `Transfer-Encoding: chunked` 区分边界——太重了
- `\n` 是最简方案：JSON 本身被设计为单行可读，不会包含字面换行符，所以用 `\n` 作为结束标记不会误判

**线路上传输的样子**（一次 TCP 连接发送的完整字节流）：

```
{"report_id":"rpt-001","node_id":"edge-01","timestamp_ms":1721200000000,"detections":[{"target_id":"obj-42","target_type":"defect","classification":"scratch","confidence":0.72,"decision":"critical","sensor_type":"camera","bbox":[100,200,50,80]}]}\n
```

发送方代码（`p2p_mesh.cpp:send_tcp()`）：
```cpp
bool P2PMesh::send_tcp(const std::string & ip, int port,
                        const std::string & payload) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((u_short)port);
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);

    // 非阻塞 connect（2 秒超时）
    set_nonblocking(s);
    connect(s, (const sockaddr*)&addr, sizeof(addr));

    // 用 select 等待连接完成
    fd_set wfds;
    FD_ZERO(&wfds);
    FD_SET(s, &wfds);
    timeval tv = {2, 0};              // 2 秒超时
    select((int)(s + 1), nullptr, &wfds, nullptr, &tv);

    // 发送 JSON + 换行符
    std::string data = payload + "\n";
    send(s, data.c_str(), (int)data.size(), 0);

    closesocket(s);
    return true;
}
```

接收方代码（`p2p_mesh.cpp:handle_peer_connection()`）：
```cpp
void P2PMesh::handle_peer_connection(int client_fd) {
    std::string buffer;
    char buf[4096];
    int total_read = 0;

    // 循环 recv 直到遇到 '\n' 或连接关闭
    while (total_read < 65536) {      // 安全上限 64 KB
        int n = recv(client_fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;            // 连接关闭或出错
        buf[n] = '\0';
        buffer.append(buf, n);
        total_read += n;

        if (buffer.find('\n') != std::string::npos) break;  // 收到完整消息！
    }

    if (buffer.empty()) return;
    if (buffer.back() == '\n') buffer.pop_back();  // 去掉换行符

    // 现在 buffer 是完整的 JSON 字符串
    auto j = nlohmann::json::parse(buffer);

    if (j.contains("report_id")) {
        PerceptionReport r = parse_report(buffer);
        on_perception(r);            // → 触发冲突检测
    }
    else if (j.contains("intent_id")) {
        DecisionIntent di = parse_intent(buffer);
        on_decision_intent(di);      // → 触发冲突检测
    }
}
```

---

### 6.4 三通道对照总表

| 维度 | HTTP JSON (通道一) | SSE 流式 (通道二) | UDP (通道三) | TCP (通道三) |
|------|-------------------|------------------|-------------|-------------|
| **方向** | 边端 → 云端 | 云端 → 边端 | 边端 ↔ 边端 | 边端 ↔ 边端 |
| **连接模式** | 短连接（一问一答） | 长连接（分块推送） | 无连接（发完即忘） | 短连接（每条消息建连） |
| **序列化格式** | JSON | JSON event | 纯文本 `\|` 分隔 | 紧凑 JSON |
| **消息边界** | `Content-Length` 头 | `\n\n` + chunked | 每包天然边界 | `\n` 换行符 |
| **二进制处理** | Base64 嵌入 JSON 字段 | 同左 | 不支持 | Base64 嵌入 JSON 字段 |
| **可靠性** | TCP 保证可靠 | TCP 保证可靠 | UDP 不保证（可能丢） | TCP 保证可靠 |
| **延迟特征** | 中（建连+HTTP头开销） | 低（长连接，逐 token） | 极低（无连接开销） | 中（3 次握手+拆连） |
| **每消息典型体积** | <1 KB ~ 2.7 MB | 每 token ~10-60 B | <100 B | <5 KB |
| **使用场景** | 心跳上报、任务拉取、Split推理、特征上传 | token 流式输出 | 节点发现、保活广播 | 感知报告交换、冲突协商 |
| **本项目库** | cpp-httplib + nlohmann/json | cpp-httplib + nlohmann/json | 原生 Socket (Winsock2 / BSD) | 原生 Socket |
| **核心代码文件** | `heartbeat.cpp`, `task_scheduler.cpp`, `edge_cloud.cpp` | `edge_cloud.cpp` | `p2p_mesh.cpp` | `p2p_mesh.cpp` |

---

### 6.5 完整数据流示例：一次工业质检云边协同

下面是一条产线图像质检请求从采集到决策的完整数据流，覆盖所有三种传输形式：

```
┌─────────────────────────────────────────────────────────────────────┐
│ 边端节点 edge-01                                                     │
│                                                                     │
│ 1. 产线相机采集图像 → ImageSource("defect.jpg")                      │
│ 2. EdgeEngine 本地推理: Qwen3-1.7B                                  │
│    → classification="scratch", confidence=0.65                     │
│    → 低于阈值 0.85，触发云复核                                       │
│                                                                     │
│ 3. [通道一 HTTP] TaskScheduler 拉取任务                            │
│    GET /api/v1/edge/tasks?device_id=edge-01                        │
│    ← HTTP Body: [{"type":"cloud_review",...}]          (JSON)      │
│                                                                     │
│ 4. [通道一 HTTP] EdgeCloudClient::feature_offload()                │
│    POST /api/v1/infer/features                                      │
│    Content-Type: application/json                                   │
│    Body: {"prompt":"分析缺陷类型","media":{"features_b64":"SGVs..."}}│
│    ↑ 视觉特征 Base64 编码后嵌入 JSON                    (JSON+Base64) │
│                                                                     │
│ 5. [通道二 SSE] 云端流式返回 token                                  │
│    data: {"choices":[{"delta":{"content":"裂纹"}}]}\n\n   (SSE)     │
│    data: {"choices":[{"delta":{"content":"缺陷"}}]}\n\n   (SSE)     │
│    data: [DONE]\n\n                                       (SSE)     │
│                                                                     │
│ 6. fuse_results(local=0.65, cloud=0.92) → 采用云端结果             │
│                                                                     │
│ 7. [通道三 UDP] 广播感知报告                                        │
│    sendto("EDGE_DISCOVER|edge-01|15556") → 255.255.255.255:15555   │
│    ↑ 纯文本管道符分隔                                  (UDP文本)     │
│                                                                     │
│ 8. [通道三 TCP] 发送感知报告给对等节点                               │
│    connect(edge-02:15556)                                           │
│    send("{\"report_id\":\"rpt-001\",...}\n")              (TCP JSON) │
│    ↑ 紧凑 JSON + 换行符边界                                        │
│    close()                                                          │
│                                                                     │
│ 9. edge-02 认为 same target 是 "normal"(conf=0.85)                  │
│    → ConflictDetector: class_mismatch                               │
│    → 本地4级消解尝试 → 失败 (confidence gap < 0.3 且 sensor 同级)    │
│                                                                     │
│10. [通道一 HTTP] 升级云端仲裁                                        │
│    POST /api/v1/edge/conflicts                            (JSON)    │
│    Body: {"conflict_id":"...","our_decision":"critical",...}        │
│    ← 云端仲裁: "确认缺陷，置信度 0.91"                      (JSON)   │
│                                                                     │
│11. 最终决策: critical → 触发产线报警                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、云端对接指南

本章从**云端服务端**视角说明：收到边端发来的上述请求后，云端应当如何处理每一类数据、如何解析格式、如何构造响应。

### 7.1 云端需要实现的 API 端点总览

| 序号 | 方法 | 路径 | 触发方 | 频率 | 用途 |
|------|------|------|--------|------|------|
| ① | `GET` | `/api/v1/health` | 边端 | 按需（测试连接时） | 健康检查 |
| ② | `POST` | `/api/v1/edge/heartbeat` | 边端 | 5~30s | 接收心跳状态 |
| ③ | `GET` | `/api/v1/edge/tasks?device_id=xxx` | 边端 | 2s 轮询 | 下发协同任务 |
| ④ | `POST` | `/api/v1/infer/split` | 边端 | 按需 | Split Inference |
| ⑤ | `POST` | `/api/v1/infer/features` | 边端 | 按需 | 特征上传 |
| ⑥ | `POST` | `/v1/chat/completions` | 边端 | 按需 | 文本卸载（OpenAI 兼容） |
| ⑦ | `GET` | `/api/v1/edge/model-updates?device_id=xxx&current_lora_version=N` | 边端 | 定期 | 模型/规则更新查询 |
| ⑧ | `GET` | `/api/v1/edge/neighbors?device_id=xxx` | 边端 | 按需 | 邻居节点发现 |
| ⑨ | `POST` | `/api/v1/edge/conflicts` | 边端 | 仅冲突时 | 冲突仲裁 |

---

### 7.2 端点详解：如何处理边端发来的数据

#### ① GET /api/v1/health — 健康检查

**云端收到的请求**：
```
GET /api/v1/health HTTP/1.1
Host: cloud:8080
```

**云端处理逻辑**：
```
1. 无需解析 Body（GET 请求无 Body）
2. 检查自身大模型服务是否正常
3. 返回当前负载信息
```

**云端返回**（200 OK）：
```json
{
  "status": "ok",
  "load": 35,
  "version": "1.2.0",
  "max_split_layer": 28
}
```

---

#### ② POST /api/v1/edge/heartbeat — 心跳上报

**云端收到的请求**：
```
POST /api/v1/edge/heartbeat HTTP/1.1
Content-Type: application/json
Content-Length: 387

{"cpu":{"core_count":8,"usage_pct":45.2},"device_id":"edge-node-01","device_type":"jetson-orin-nano","gpus":[...],"inference":{"current_tps":45.3,"task_queue_len":2},"is_fluctuating":false,"memory":{"avail_mb":3200,"total_mb":8192},"network":{"cloud_reachable":true,"rtt_ms":12},"timestamp_ms":1721200000000}
```

**云端处理逻辑**：
```python
def handle_heartbeat(request):
    body = json.loads(request.body)

    device_id = body["device_id"]                    # "edge-node-01"
    cpu_usage = body["cpu"]["usage_pct"]             # 45.2
    mem_avail = body["memory"]["avail_mb"]           # 3200
    is_fluctuating = body["is_fluctuating"]          # false
    current_tps = body["inference"]["current_tps"]   # 45.3
    rtt_ms = body["network"]["rtt_ms"]               # 12

    # 1. 写入时序数据库（用于调度决策）
    db.insert("edge_metrics", {
        "device_id": device_id,
        "timestamp": body["timestamp_ms"],
        "cpu_pct": cpu_usage,
        "mem_avail_mb": mem_avail,
        "tps": current_tps,
        "rtt_ms": rtt_ms,
    })

    # 2. 更新设备状态表
    db.upsert("device_status", {
        "device_id": device_id,
        "last_seen": now(),
        "is_fluctuating": is_fluctuating,
        "online": True,
    })

    # 3. 调度决策：是否要对该设备下发任务？
    #    （见 7.3 节调度决策逻辑）

    return Response(200)
```

**关键：不需要解析二进制**——心跳是完全的 JSON 纯文本，直接 `json.loads()` 即可。

---

#### ③ GET /api/v1/edge/tasks — 任务下发

这是**云端主动决策**的核心端点。边端每 2 秒轮询一次，云端根据心跳数据决定是否有任务要下发。

**云端收到的请求**：
```
GET /api/v1/edge/tasks?device_id=edge-node-01 HTTP/1.1
```

**云端处理逻辑**：
```python
def handle_task_poll(request):
    device_id = request.query_params["device_id"]   # "edge-node-01"

    # 1. 查询该设备的状态
    status = db.get("device_status", device_id)
    metrics = db.get_latest("edge_metrics", device_id)

    tasks = []

    # 2. 调度决策：是否需要云复核？
    if should_cloud_review(device_id, metrics):
        tasks.append({
            "type": "cloud_review",
            "task_id": f"task-{uuid4()}",
            "description": "对边端低置信度推理结果进行全量模型复核",
            "priority": 5,
            "payload": {
                "max_tokens": 512,
                "temperature": 0.7,
            }
        })

    # 3. 调度决策：是否需要上传特征？
    if should_upload_features(device_id, metrics):
        tasks.append({
            "type": "upload_features",
            "task_id": f"task-{uuid4()}",
            "description": "上传当前检测目标的视觉特征用于二次分析",
            "priority": 3,
            "payload": {
                "layer_idx": -1,     # -1 = 使用输出层 embedding
                "targets": ["all"],
            }
        })

    # 4. 调度决策：是否需要更新模型？
    latest_lora = db.get_latest_lora_version(device_id)
    if latest_lora > status.get("current_lora_version", 0):
        tasks.append({
            "type": "model_update",
            "task_id": f"task-{uuid4()}",
            "description": f"更新质检 LoRA 到 v{latest_lora}",
            "priority": 2,
            "payload": {
                "url": f"http://cloud/models/loras/quality_v{latest_lora}.gguf",
                "version": latest_lora,
                "scale": 1.0,
            }
        })

    # 5. 如果设备波动，下发规则调整
    if metrics.get("is_fluctuating"):
        tasks.append({
            "type": "rule_sync",
            "task_id": f"task-{uuid4()}",
            "description": "网络波动，降低异常检测置信度阈值",
            "priority": 4,    # 高优先级
            "payload": {
                "rule_id": "anomaly_threshold_adjust",
                "rule_type": "threshold",
                "target": "anomaly_score",
                "value": {"min": 0.75, "max": 1.0},   # 从 0.85 降到 0.75
            }
        })

    return Response(200, body=json.dumps(tasks))
```

**注意**：如果 `tasks` 数组为空，边端只做本地推理，不触发云边协同：
```json
[]
```

---

#### ④ POST /api/v1/infer/split — Split Inference（最复杂的端点）

**云端收到的请求**：
```http
POST /api/v1/infer/split HTTP/1.1
Content-Type: application/json
Content-Length: 2796457

{"hidden":{"split_layer":12,"current_heads":16,"head_dim":128,"batch_size":1,"seq_len":512,"dtype":"float16","data_b64":"SGVsbG8...base64...","data_bytes":2097152,"checksum":"a1b2c3d4"},"context":{"n_past":512,"temperature":0.7,"top_p":0.9,"max_new_tokens":256,"prompt_tokens_b64":"...","n_prompt_tokens":512},"device":{"device_id":"edge-01","avail_mem_mb":3200,"network_rtt_ms":12},"model":"qwen3-1.7b"}
```

**云端处理逻辑**（这是最关键的解析步骤）：
```python
import base64
import struct
import numpy as np

def handle_split_inference(request):
    body = json.loads(request.body)

    # ── 第 1 步：从 Base64 还原隐藏状态为 numpy 数组 ──
    hidden = body["hidden"]
    dtype_str = hidden["dtype"]           # "float16"
    split_layer = hidden["split_layer"]   # 12
    seq_len = hidden["seq_len"]           # 512
    n_heads = hidden["current_heads"]     # 16
    head_dim = hidden["head_dim"]         # 128

    # Base64 解码 → 原始字节
    raw_bytes = base64.b64decode(hidden["data_b64"])

    # 校验数据完整性
    assert len(raw_bytes) == hidden["data_bytes"]

    # 字节 → numpy 数组（按 dtype 解析）
    if dtype_str == "float16":
        hidden_states = np.frombuffer(raw_bytes, dtype=np.float16)
    elif dtype_str == "float32":
        hidden_states = np.frombuffer(raw_bytes, dtype=np.float32)
    elif dtype_str == "bfloat16":
        # bfloat16 需要特殊处理：转成 uint16 再转 float32
        as_uint16 = np.frombuffer(raw_bytes, dtype=np.uint16)
        hidden_states = as_uint16.astype(np.float32) * (2**16)
        hidden_states = hidden_states.view(np.float32)

    # Reshape 为正确的张量形状
    hidden_dim = n_heads * head_dim      # 16 × 128 = 2048
    hidden_states = hidden_states.reshape(1, seq_len, hidden_dim)
    # shape: [batch=1, seq_len=512, hidden_dim=2048]

    # ── 第 2 步：解析推理上下文 ──
    ctx = body["context"]
    temperature = ctx["temperature"]     # 0.7
    max_tokens  = ctx["max_new_tokens"]  # 256

    # 解析原始 prompt tokens（也是 Base64 编码的）
    if "prompt_tokens_b64" in ctx:
        token_bytes = base64.b64decode(ctx["prompt_tokens_b64"])
        prompt_tokens = np.frombuffer(token_bytes, dtype=np.int32).tolist()

    # ── 第 3 步：云端继续推理 ──
    # 将 hidden_states 注入全量大模型的第 split_layer+1 层
    # （具体做法取决于推理框架，以下是伪代码）
    cloud_model = get_model(body["model"])    # 加载全量大模型

    # 方式 A：如果框架支持从中间层注入
    output = cloud_model.generate_from_hidden(
        hidden_states=hidden_states,         # [1, 512, 2048]
        start_layer=split_layer + 1,         # 从第 13 层继续
        temperature=temperature,
        max_new_tokens=max_tokens,
        stop_strings=ctx.get("stop_strings", []),
        seed=ctx["seed"],
    )

    # 方式 B：如果没有直接 API，可以用 KV Cache 重建
    # （将 hidden_states 作为第 12 层的"输出"，让模型从第 13 层开始算）

    # ── 第 4 步：SSE 流式返回 ──
    def generate_sse():
        for token_text in output.stream():
            event = {
                "choices": [{
                    "delta": {"content": token_text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(event)}\n\n"

        # 结束事件
        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'total_tokens': output.total_tokens}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_sse(), media_type="text/event-stream")
```

**Base64 解码体积对照表**：

| 字段 | Base64 体积 | 解码后体积 | numpy shape |
|------|------------|-----------|-------------|
| `hidden.data_b64` | ~2.67 MB | 2 MB | `[1, 512, 2048]` (FP16) |
| `context.prompt_tokens_b64` | ~2.7 KB | 2 KB | `[512]` (int32) |
| `context.kv_cache_b64` (可选) | ~1 MB | ~768 KB | 二进制 blob |

---

#### ⑤ POST /api/v1/infer/features — 特征上传

**云端收到的请求**：
```json
{
  "prompt": "分析缺陷类型",
  "model": "qwen2-vl-7b",
  "temperature": 0.7,
  "max_tokens": 256,
  "media": {
    "type": "feature_vector",
    "feature_shape": [49, 1024],
    "features_b64": "AAAA...base64...",
    "width": 224,
    "height": 224,
    "mime": "image/jpeg"
  }
}
```

**云端处理逻辑**：
```python
def handle_feature_upload(request):
    body = json.loads(request.body)
    media = body["media"]

    # 1. Base64 解码特征向量
    feature_bytes = base64.b64decode(media["features_b64"])
    features = np.frombuffer(feature_bytes, dtype=np.float32)
    features = features.reshape(media["feature_shape"])   # [49, 1024]

    # 2. 如果是 Vision Encoder 的 patch 输出，送入 ViT Decoder
    #    或者直接用全量大模型的多模态能力做二次分析
    prompt = body["prompt"]   # "分析缺陷类型"
    model = get_multimodal_model(body["model"])

    # 3. 用特征 + 文本 prompt 推理
    output = model.generate_from_visual_features(
        visual_features=features,
        text_prompt=prompt,
        temperature=body["temperature"],
        max_new_tokens=body["max_tokens"],
    )

    # 4. SSE 流式返回（同端点④的返回格式）
    return sse_stream_response(output)
```

**注意**：如果 `media` 中包含 `raw_bytes_b64`（原始图片 Base64）而非 `features_b64`，说明边端没有做特征提取，直接传了原始数据：
```python
if "raw_bytes_b64" in media:
    raw = base64.b64decode(media["raw_bytes_b64"])
    # raw 是 JPEG/PNG 的原始字节 → 直接送入视觉模型
    output = model.generate_from_image(raw, prompt)
```

---

#### ⑥ POST /v1/chat/completions — 文本卸载

**完全兼容 OpenAI API 格式**，边端的 `EdgeCloudClient::query_offload()` 发送的是标准格式。云端可以直接对接任何兼容 OpenAI 的推理框架（vLLM、llama-server 等）。

**边端发来的请求**：
```json
{
  "model": "deepseek-v2",
  "messages": [
    {"role": "system", "content": "你是一个工业质检助手"},
    {"role": "user", "content": "设备振动频率异常，判断是否需要停机检修？"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "stream": true
}
```

**云端处理**：标准 OpenAI chat completions 逻辑，SSE 流式返回（同端点④）。

---

#### ⑦ GET /api/v1/edge/model-updates — 模型更新

**云端收到的请求**：
```
GET /api/v1/edge/model-updates?device_id=edge-01&current_lora_version=2
```

**云端处理逻辑**：
```python
def handle_model_updates(request):
    device_id = request.query_params["device_id"]
    current_ver = int(request.query_params.get("current_lora_version", 0))

    # 查询是否有更新的 LoRA
    latest_lora = db.get_latest_lora(device_id)  # version=3

    updates = {"lora_updates": [], "rule_updates": []}

    if latest_lora and latest_lora.version > current_ver:
        updates["lora_updates"].append({
            "update_id": f"lora-v{latest_lora.version}",
            "url": f"http://cloud/models/loras/quality_v{latest_lora.version}.gguf",
            "local_path": "",
            "scale": 1.0,
            "version": latest_lora.version,
            "checksum": latest_lora.checksum_md5,
        })

    # 同时检查是否有新的决策规则
    current_rules_ver = int(request.query_params.get("current_rule_version", 0))
    new_rules = db.get_rules_since(device_id, current_rules_ver)
    updates["rule_updates"] = new_rules

    return Response(200, body=json.dumps(updates))
```

**边端收到后会自动**：
- 下载 `.gguf` LoRA 文件 → `llama_adapter_lora_init()` → `llama_set_adapters_lora()` → 热生效
- `apply_rule()` 写入 `_active_rules` → ConflictDetector 立即读取

---

#### ⑧ GET /api/v1/edge/neighbors — 邻居发现

```python
def handle_neighbors(request):
    device_id = request.query_params["device_id"]

    # 查询同一感知区域的其他在线节点
    neighbors = db.query("""
        SELECT node_id, ip_address, udp_port, tcp_port, device_type,
               monitored_asset, camera_fov
        FROM device_status
        WHERE online = True
          AND node_id != ?
          AND monitored_asset IN (
              SELECT monitored_asset FROM device_status WHERE node_id = ?
          )
    """, device_id, device_id)

    return Response(200, body=json.dumps(neighbors))
```

---

#### ⑨ POST /api/v1/edge/conflicts — 冲突仲裁

**云端收到的请求**（边端 4 级消解全部失败后才发）：
```json
{
  "conflict_id": "target_001_vs_edge-02",
  "target_id": "target_001",
  "target_type": "defect",
  "our_node_id": "edge-01",
  "our_decision": "critical",
  "our_confidence": 0.72,
  "peer_node_id": "edge-02",
  "peer_decision": "normal",
  "peer_confidence": 0.85,
  "reason": "class_mismatch",
  "severity": 3
}
```

**云端处理逻辑**：
```python
def handle_conflict_arbitration(request):
    body = json.loads(request.body)

    # 1. 收集两个节点的原始感知数据
    node_a_data = db.get_latest_perception(body["our_node_id"], body["target_id"])
    node_b_data = db.get_latest_perception(body["peer_node_id"], body["target_id"])

    # 2. 用云端全量大模型做最终裁决
    prompt = f"""两个边缘节点对同一目标做出了不同判断：
节点A ({body['our_node_id']}): {body['our_decision']}, 置信度 {body['our_confidence']}
节点B ({body['peer_node_id']}): {body['peer_decision']}, 置信度 {body['peer_confidence']}
冲突原因: {body['reason']}
请给出最终仲裁判断及理由。"""

    arbitration = cloud_llm.generate(prompt)

    # 3. 记录仲裁结果
    db.insert("conflict_arbitrations", {
        "conflict_id": body["conflict_id"],
        "result": arbitration["decision"],
        "confidence": arbitration["confidence"],
        "reasoning": arbitration["reasoning"],
        "timestamp": now(),
    })

    # 4. 返回仲裁结果（文本格式，边端会接收）
    return Response(200, body=arbitration["decision"])
```

**边端接收后的处理**（`conflict_detector.cpp:escalate_to_cloud()`）：
```cpp
// 云端返回的文本被当作最终决策
conflict.resolved = true;
conflict.resolved_locally = false;
conflict.resolution_method = "cloud";
conflict.final_decision = text;   // 云端返回的仲裁结果
++_stats.resolution_success;
```

---

### 7.3 云端调度决策逻辑

云端什么时候应该下发任务？核心判断基于心跳上报的设备状态：

```python
def should_cloud_review(device_id, metrics):
    """
    以下情况触发云复核：
    1. 边端推理置信度持续低于阈值（异常率高）
    2. 边端算力不足（内存紧张 + TPS 低）
    3. 收到边端主动的 cloud_review 请求
    """
    # 如果边端异常检测命中率突然飙升 → 可能模型误判，需要云端复核
    recent_anomalies = db.count_anomalies(device_id, minutes=5)
    if recent_anomalies > 10:    # 5分钟内超过10个异常
        return True

    # 如果边端内存不足 → 卸载计算到云端
    if metrics.get("memory", {}).get("avail_mb", 0) < 1024:  # 不足1GB
        return True

    return False


def should_upload_features(device_id, metrics):
    """
    以下情况要求边端上传特征：
    1. 云端有新版本的多模态模型，需要重新分析
    2. 周期性特征采集（每小时一次，用于云端持续训练）
    """
    # 如果云端模型版本高于边端 → 用云端模型重新分析
    cloud_model_ver = get_latest_model_version()
    edge_model_ver = db.get("device_status", device_id).get("model_version", 0)
    if cloud_model_ver > edge_model_ver:
        return True

    return False
```

---

### 7.4 云端最小化参考实现（Python/Flask）

以下是一个可直接运行的云端服务骨架，包含了边端需要的所有 9 个端点：

```python
import json
import base64
import time
import numpy as np
from flask import Flask, request, Response, jsonify, stream_with_context

app = Flask(__name__)

# ── 模拟存储 ──
device_metrics = {}       # device_id → 最新心跳数据
device_tasks = {}         # device_id → 待下发任务列表
device_status = {}        # device_id → {"online": bool, "lora_version": int}
conflict_log = []         # 冲突仲裁记录
neighbor_registry = {}    # device_id → PeerNode 信息

# ═══════════════════════════════════════════════════════════════
# ① 健康检查
# ═══════════════════════════════════════════════════════════════
@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "load": 35,
        "version": "1.2.0",
        "max_split_layer": 28,
    })

# ═══════════════════════════════════════════════════════════════
# ② 心跳上报
# ═══════════════════════════════════════════════════════════════
@app.route("/api/v1/edge/heartbeat", methods=["POST"])
def heartbeat():
    body = request.get_json()
    device_id = body["device_id"]

    # 存储
    device_metrics[device_id] = body
    device_status[device_id] = {
        "online": True,
        "last_seen": time.time(),
        "is_fluctuating": body.get("is_fluctuating", False),
    }

    # 注册邻居信息（用于 P2P 发现）
    neighbor_registry[device_id] = {
        "node_id": device_id,
        "ip": request.remote_addr,
        "udp_port": 15555,
        "tcp_port": 15556,
        "device_type": body.get("device_type", "unknown"),
        "asset": body.get("monitored_asset", "default"),
        "fov": body.get("camera_fov", ""),
    }

    # 调度决策：是否生成任务？
    _maybe_generate_tasks(device_id, body)

    return "", 200

# ═══════════════════════════════════════════════════════════════
# ③ 任务下发（边端轮询）
# ═══════════════════════════════════════════════════════════════
@app.route("/api/v1/edge/tasks", methods=["GET"])
def tasks():
    device_id = request.args.get("device_id")
    pending = device_tasks.pop(device_id, [])
    return jsonify(pending)

# ═══════════════════════════════════════════════════════════════
# ④ Split Inference
# ═══════════════════════════════════════════════════════════════
@app.route("/api/v1/infer/split", methods=["POST"])
def split_infer():
    body = request.get_json()
    hidden = body["hidden"]
    ctx = body["context"]

    # Base64 → numpy
    raw = base64.b64decode(hidden["data_b64"])
    dtype_map = {"float16": np.float16, "float32": np.float32}
    dtype = dtype_map.get(hidden["dtype"], np.float16)
    hidden_states = np.frombuffer(raw, dtype=dtype)
    hidden_dim = hidden["current_heads"] * hidden["head_dim"]
    hidden_states = hidden_states.reshape(
        hidden["batch_size"], hidden["seq_len"], hidden_dim)

    # 云端继续推理（此处用模拟输出代替）
    def generate():
        tokens = ["分析", "结果", "：", "确认为", "裂纹", "缺陷", "，", "建议", "停机", "检修"]
        for tok in tokens:
            time.sleep(0.05)  # 模拟推理延迟
            yield f"data: {json.dumps({'choices': [{'delta': {'content': tok}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'total_tokens': len(tokens)}})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()),
                    content_type="text/event-stream")

# ═══════════════════════════════════════════════════════════════
# ⑤ 特征上传
# ═══════════════════════════════════════════════════════════════
@app.route("/api/v1/infer/features", methods=["POST"])
def feature_offload():
    body = request.get_json()
    media = body["media"]

    # 解析特征
    if "features_b64" in media:
        raw = base64.b64decode(media["features_b64"])
        features = np.frombuffer(raw, dtype=np.float32)
        shape = media.get("feature_shape", [-1])
        features = features.reshape(shape)
        # features 现在是 [49, 1024] 的视觉特征矩阵
    elif "raw_bytes_b64" in media:
        raw = base64.b64decode(media["raw_bytes_b64"])
        # raw 是 JPEG/PNG 原始字节

    # SSE 流式返回（同④）
    return Response(stream_with_context(_mock_sse_generate()),
                    content_type="text/event-stream")

# ═══════════════════════════════════════════════════════════════
# ⑥ 文本卸载（OpenAI 兼容）
# ═══════════════════════════════════════════════════════════════
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    body = request.get_json()
    # body["messages"] 是标准 OpenAI 格式
    # 直接对接 llama-server 或 vLLM 的 /v1/chat/completions
    return Response(stream_with_context(_mock_sse_generate()),
                    content_type="text/event-stream")

# ═══════════════════════════════════════════════════════════════
# ⑦ 模型更新
# ═══════════════════════════════════════════════════════════════
@app.route("/api/v1/edge/model-updates", methods=["GET"])
def model_updates():
    device_id = request.args.get("device_id")
    current_ver = int(request.args.get("current_lora_version", 0))

    result = {"lora_updates": [], "rule_updates": []}

    # 模拟：如果有 v3 LoRA
    if current_ver < 3:
        result["lora_updates"].append({
            "update_id": "lora-quality-v3",
            "url": f"http://{request.host}/models/loras/quality_v3.gguf",
            "scale": 1.0,
            "version": 3,
            "checksum": "a1b2c3d4",
        })

    return jsonify(result)

# ═══════════════════════════════════════════════════════════════
# ⑧ 邻居发现
# ═══════════════════════════════════════════════════════════════
@app.route("/api/v1/edge/neighbors", methods=["GET"])
def neighbors():
    device_id = request.args.get("device_id")
    # 返回除自己外所有在线节点
    peers = [
        info for nid, info in neighbor_registry.items()
        if nid != device_id
    ]
    return jsonify(peers)

# ═══════════════════════════════════════════════════════════════
# ⑨ 冲突仲裁
# ═══════════════════════════════════════════════════════════════
@app.route("/api/v1/edge/conflicts", methods=["POST"])
def conflict_arbitration():
    body = request.get_json()

    # 云端全量模型仲裁（此处用模拟）
    # 实际实现：将冲突上下文送入全量大模型，获取裁决
    arbitration = "critical"  # 模拟结果
    reasoning = f"综合分析两节点数据，采纳 critical 判断"

    conflict_log.append({
        **body,
        "arbitration": arbitration,
        "reasoning": reasoning,
    })

    return jsonify({"decision": arbitration, "reasoning": reasoning})


# ═══════════════════════════════════════════════════════════════
# 辅助：调度决策
# ═══════════════════════════════════════════════════════════════
def _maybe_generate_tasks(device_id, metrics):
    tasks = []

    # 内存不足 → 卸载
    if metrics.get("memory", {}).get("avail_mb", 0) < 1024:
        tasks.append({
            "type": "cloud_review",
            "task_id": f"task-{int(time.time()*1000)}",
            "description": "边端内存不足，卸载推理到云端",
            "priority": 5,
            "payload": {"max_tokens": 512, "temperature": 0.7},
        })

    # 波动中 → 调整阈值
    if metrics.get("is_fluctuating"):
        tasks.append({
            "type": "rule_sync",
            "task_id": f"rule-{int(time.time()*1000)}",
            "description": "波动期间降低决策阈值",
            "priority": 4,
            "payload": {
                "rule_id": "fluctuation_adjust",
                "rule_type": "threshold",
                "target": "cloud_review_confidence_threshold",
                "value": {"min": 0.65},   # 从 0.85 降到 0.65
            },
        })

    if tasks:
        device_tasks.setdefault(device_id, []).extend(tasks)


def _mock_sse_generate():
    """模拟 SSE token 输出"""
    tokens = ["分析", "完成", "。"]
    for tok in tokens:
        time.sleep(0.05)
        ev = {"choices": [{"delta": {"content": tok}, "finish_reason": None}]}
        yield f"data: {json.dumps(ev)}\n\n"
    yield f"data: [DONE]\n\n"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

将此文件保存为 `cloud_server.py`，运行 `python cloud_server.py` 即可启动模拟云端。边端在"云端配置"面板中将地址设为 `http://localhost:8080` 即可对接测试。

---

### 7.5 云端对接检查清单

| 检查项 | 对应端点 | 验证方法 |
|--------|---------|---------|
| 健康检查正常返回 | ① | `curl http://cloud:8080/api/v1/health` |
| 心跳接收并存入数据库 | ② | 启动边端 30 秒后查询 `device_metrics` |
| 任务下发格式正确 | ③ | 检查边端 UI "任务队列"面板是否显示收到任务 |
| Base64 解码后张量形状正确 | ④ | `hidden_states.shape` 应为 `[batch, seq_len, hidden_dim]` |
| SSE 流格式正确 | ④⑤⑥ | `data: {...}\n\n` 格式，结束有 `[DONE]` |
| LoRA 文件可下载 | ⑦ | `curl -O http://cloud/models/loras/quality_v3.gguf` |
| 邻居列表返回正确 | ⑧ | 启动两个边端节点，确认彼此出现在邻居列表 |
| 冲突仲裁返回决策 | ⑨ | 模拟冲突数据 POST，确认返回有效 decision |
