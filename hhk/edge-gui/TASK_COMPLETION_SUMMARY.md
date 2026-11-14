# 边端-云端协同系统 — 本次任务完成总结

> 日期: 2026-07-20  
> 基于: [CLOUD_EDGE_INTEGRATION_PLAN.md](CLOUD_EDGE_INTEGRATION_PLAN.md)

---

## 一、前置分析

### 1.1 分析对象

| 代码库 | 路径 | 语言 | 规模 |
|--------|------|------|------|
| 云端 Demo | `D:\大学\揭榜挂帅\cloud_edge_demo_phase1` | Python | ~1800 行，12 个 .py 文件 |
| 边端项目 | `D:\llama.cpp\edge-gui` | C++17 | ~4700 行，22 个 .h/.cpp 文件 |
| 接口文档 | `D:\llama.cpp\edge-gui\CLOUD_EDGE_INTERFACE.md` | Markdown | 1879 行，9 个 API 端点定义 |

### 1.2 分析结论

- **云端 Demo**：纯离线仿真（Streamlit 面板 + 离散时隙模拟器），**无 HTTP 接口、无真实推理**，但具有 RuleScheduler + PPOScheduler 调度逻辑
- **边端项目**：完整的桌面应用，已实现本地推理 + 7 个云边协同子系统 + P2P 通信 + 冲突消解，**HTTP 客户端代码已就绪**，等待云端服务端对接
- **核心缺口**：边端期望云端有 9 个 API 端点，但云端 Demo 一个都没有；边端缺少弱网/离线时的自主决策能力

---

## 二、方案设计

### 2.1 产出文件

**[CLOUD_EDGE_INTEGRATION_PLAN.md](CLOUD_EDGE_INTEGRATION_PLAN.md)** — 完整的云边协同对接方案（~950 行）

### 2.2 核心架构决策

**双层决策模型**：

```
网络好 (RTT < 200ms)     →  云端 PPO 做主，边端拉取执行
网络差 (RTT 200-500ms)   →  边端自主初筛，高置信度本地、低置信度排队
离线   (RTT > 500ms)     →  边端完全自主，P2P 共识替代云端仲裁
```

**推理策略**：Query Offload 优先（边端先跑→云端后台复核），Split Inference 预留

### 2.3 五阶段实施计划

| Phase | 内容 | 时间 |
|-------|------|------|
| 1 | 云端服务基础架构 | Week 1 |
| 2 | 云边联调 | Week 2 |
| 3 | 技术指标达成 | Week 3 |
| 4 | 多场景部署验证 | Week 4 |
| 5 | 集成与展示 | Week 5 |

---

## 三、代码实现

### 3.1 边端新增文件

#### `edge-gui/edge_local_decision.h` (88 行)

边端本地自主决策引擎类声明：

- **`NetworkCondition` 枚举**: `Healthy` / `Degraded` / `Offline` 三级网络评估
- **`LocalAction` 枚举**: `FollowCloud` / `LocalInfer` / `QueueForCloud` / `P2PConsensus` 四级决策动作
- **`DecisionSnapshot` 结构体**: 离线决策日志记录（含时间戳、理由、使用阈值）
- **`EdgeLocalDecision` 类**:
  - `update_cloud_params()` — 从云端同步参数（断网后继续使用）
  - `decide()` — 根据网络状态和置信度做本地决策
  - `assess_network()` — 评估当前网络状况
  - `flush_offline_decisions()` — 导出离线日志供云端分析
  - `active_confidence_threshold()` — 返回当前生效的置信度阈值

#### `edge-gui/edge_local_decision.cpp` (158 行)

完整实现：

- **`assess_network()`**: RTT<200ms+可达+心跳正常 → Healthy；RTT 200-500ms 或心跳失败 2-4 次 → Degraded；RTT>500ms/不可达/心跳失败≥5 次 → Offline
- **`decide()`**: Healthy→FollowCloud；Degraded+高置信度→LocalInfer；Degraded+低置信度→QueueForCloud；Offline→LocalInfer 或 P2PConsensus（置信度低于降低后的阈值时）
- **`active_confidence_threshold()`**: 离线时返回 `max(0.5, cloud_threshold - 0.15)`，在线时返回云端原始值
- 每条决策自动记录到 `_offline_log`（上限 1000 条），支持 `fprintf(stderr)` 调试输出

### 3.2 边端修改文件

#### `edge-gui/heartbeat.h`（修改 2 处）

```cpp
// 新增公有方法
int consecutive_failures() const { return _consecutive_failures; }

// 新增私有成员
int _consecutive_failures = 0;
```

#### `edge-gui/heartbeat.cpp`（修改 4 行）

在 `run_loop()` 和 `send_now()` 中：
- 心跳成功 → `_consecutive_failures = 0`
- 心跳失败 → `_consecutive_failures++`

#### `edge-gui/app.h`（修改 2 处）

```cpp
#include "edge_local_decision.h"     // 新增 include
EdgeLocalDecision _local_decision;  // 新增成员变量
```

#### `edge-gui/app.cpp`（修改 3 处，共约 50 行）

**① `submit_message()` 中增加网络感知分流逻辑**：

```
用户提交消息
  │
  ├── 测量 RTT + 检查云端可达 + 获取心跳失败次数
  │
  ├── assess_network() 判断网络状态
  │
  ├── Healthy → 正常路径（TaskScheduler 拉取云端决策）
  ├── Degraded → decide() 判断每条任务（高置信度本地 / 低置信度排队）
  └── Offline → decide() 判断 + P2P 共识（如果邻居在线）
```

**② `render_top_bar()` 中增加网络状态指示**：

```cpp
[net:ok]        // 绿色 — Healthy
[net:degraded]  // 黄色 — Degraded
[net:offline]   // 红色 — Offline
```

**③ `update_device_metrics()` 中增加网络恢复检测**：

网络从 Degraded/Offline 恢复到 Healthy 时，自动调用 `flush_offline_decisions()` 导出离线决策日志（后续可上传云端用于 PPO 训练）。

#### `edge-gui/CMakeLists.txt`（修改 2 行）

```cmake
edge_local_decision.cpp
edge_local_decision.h
```

### 3.3 云端新建项目

#### `cloud_server/` 目录（25 个文件）

```
cloud_server/
├── main.py                        # FastAPI 入口，lifespan 事件，CORS，8 个路由
├── config.py                      # 全环境变量配置 (CLOUD_* 前缀)
├── requirements.txt               # fastapi, uvicorn, pydantic, httpx, pytest
│
├── api/
│   ├── health.py                  # ① GET  /api/v1/health
│   ├── heartbeat.py               # ② POST /api/v1/edge/heartbeat
│   ├── tasks.py                   # ③ GET  /api/v1/edge/tasks
│   ├── inference.py               # ④ POST /api/v1/infer/split + ⑤ /api/v1/infer/features
│   ├── chat.py                    # ⑥ POST /v1/chat/completions (OpenAI 兼容 SSE)
│   ├── model_updates.py           # ⑦ GET  /api/v1/edge/model-updates
│   ├── neighbors.py               # ⑧ GET  /api/v1/edge/neighbors
│   └── conflicts.py               # ⑨ POST /api/v1/edge/conflicts
│
├── scheduler/
│   ├── decision_engine.py         # CloudDecisionEngine — 6 维度任务生成
│   └── ppo_adapter.py             # PPO 模型适配器（对接 Demo 训练模型）
│
├── inference/
│   ├── model_manager.py           # 统一后端管理器 (mock/llama/vllm 可切换)
│   ├── backend_llama.py           # llama-server HTTP 代理适配器
│   ├── backend_vllm.py            # vLLM HTTP 代理适配器
│   └── streaming.py               # SSE 流式输出工具
│
├── storage/
│   ├── models.py                  # Pydantic 数据模型
│   ├── device_state.py            # 线程安全全局设备状态存储
│   └── task_queue.py              # 每设备独立任务队列 (POP 语义)
│
└── tests/
    └── test_api.py                # 17 个测试用例
```

#### 云端 API 端点完整清单

| # | 方法 | 路径 | 功能 | 请求体 | 响应格式 |
|---|------|------|------|--------|---------|
| ① | GET | `/api/v1/health` | 健康检查 | — | JSON `{status, version, load, max_split_layer}` |
| ② | POST | `/api/v1/edge/heartbeat` | 接收边端心跳 | HeartbeatPayload (CPU/Memory/GPU/Network/Inference) | 200 或 503 |
| ③ | GET | `/api/v1/edge/tasks?device_id=X` | 边端拉取待执行任务 | — | JSON 数组 `[{type, task_id, priority, payload}]` |
| ④ | POST | `/api/v1/infer/split` | Split 推理 | HiddenStates + InferenceContext + DeviceInfo | `text/event-stream` SSE 流 |
| ⑤ | POST | `/api/v1/infer/features` | 特征上传推理 | Prompt + MultimodalFeatures | `text/event-stream` SSE 流 |
| ⑥ | POST | `/v1/chat/completions` | 文本卸载 (OpenAI 兼容) | `{model, messages, temperature, stream, ...}` | `text/event-stream` SSE 流 |
| ⑦ | GET | `/api/v1/edge/model-updates?device_id=X&current_lora_version=N` | 查询待更新模型/规则 | — | JSON `{lora_updates, rule_updates}` |
| ⑧ | GET | `/api/v1/edge/neighbors?device_id=X` | P2P 邻居发现 | — | JSON 数组 `[{node_id, ip, udp_port, ...}]` |
| ⑨ | POST | `/api/v1/edge/conflicts` | 冲突仲裁升级 | ConflictRecord (双方决策+置信度) | JSON `{decision, reasoning}` |

#### 与云端 Demo 的关系

**cloud_server 引用了云端 Demo 的已有模块，对 Demo 代码零改动**：

```python
# scheduler/decision_engine.py 直接 import 复用
from scheduler import RuleScheduler          # 来自 Demo，原样使用
from scheduler import PPOScheduler           # 来自 Demo，原样使用
from observation import build_observation    # 来自 Demo，原样使用
```

**云端 Demo 完全没有的内容（cloud_server 全部新增）**：
- 9 个 HTTP REST API 端点
- 多设备实时心跳接收与状态跟踪
- 每设备独立任务队列管理
- 6 维度任务自动生成（网络恶化/内存不足/TPS 过低/波动/GPU 不足/模型版本）
- 真实大模型对接（SSE 流式返回）
- 三级冲突仲裁（置信度差距 > 安全优先 > 高置信度优先）
- Pydantic 类型校验
- 17 个 pytest 测试用例

---

## 四、启动方式

### 边端

```bash
cd D:\llama.cpp
cmake --build build --config Release --target edge-gui -j 8
build\bin\Release\edge-gui.exe -m models\Qwen_Qwen3-1.7B-IQ4_XS.gguf
```

### 云端 (本地测试用)

```bash
cd D:\llama.cpp\cloud_server
pip install -r requirements.txt

# 使用 mock 后端（无需真实模型，直接启动）
python main.py

# 或连接真实 llama-server
set CLOUD_MODEL_BACKEND=llama
set CLOUD_LLAMA_SERVER_URL=http://localhost:8081
python main.py

# 或启用 PPO 调度
set CLOUD_USE_PPO_SCHEDULER=true
set CLOUD_PPO_MODEL_PATH=D:\大学\揭榜挂帅\cloud_edge_demo_phase1\models\ppo_cloud_edge_fifo.zip
python main.py
```

### 联调测试

1. 启动云端: `python cloud_server/main.py`
2. 启动 edge-gui，在"云端配置"面板填入 `http://localhost:8080`
3. 点击"测试连接" → 验证 ① `/api/v1/health` 返回 200
4. 发送消息 → 边端先本地推理返回 → 后台自动心跳上报 → 云端决策生成任务 → 边端拉取执行

---

## 五、本次改动影响面

### 边端项目 (`D:\llama.cpp`)

| 类型 | 文件 | 状态 |
|------|------|------|
| 新建 | `edge-gui/edge_local_decision.h` | ✅ |
| 新建 | `edge-gui/edge_local_decision.cpp` | ✅ |
| 修改 | `edge-gui/heartbeat.h` | ✅ (+2 处) |
| 修改 | `edge-gui/heartbeat.cpp` | ✅ (+4 行) |
| 修改 | `edge-gui/app.h` | ✅ (+2 处) |
| 修改 | `edge-gui/app.cpp` | ✅ (+~50 行，3 处) |
| 修改 | `edge-gui/CMakeLists.txt` | ✅ (+2 行) |
| 未改 | `edge-gui/engine.*` | — |
| 未改 | `edge-gui/cloud.*` | — |
| 未改 | `edge-gui/edge_cloud.*` | — |
| 未改 | `edge-gui/device_monitor.*` | — |
| 未改 | `edge-gui/task_scheduler.*` | — |
| 未改 | `edge-gui/model_sync.*` | — |
| 未改 | `edge-gui/p2p_mesh.*` | — |
| 未改 | `edge-gui/conflict_detector.*` | — |
| 未改 | `edge-gui/media_source.*` | — |
| 未改 | `edge-gui/file_dialog.*` | — |
| 未改 | `edge-gui/main.cpp` | — |

### 云端测试项目 (`D:\llama.cpp\cloud_server\`)

| 类型 | 文件数 | 说明 |
|------|--------|------|
| 新建 | 25 个 | 独立 Python 项目，不影响 llama.cpp 编译 |

### 已有云端 Demo (`D:\大学\揭榜挂帅\cloud_edge_demo_phase1\`)

| 类型 | 说明 |
|------|------|
| **零改动** | cloud_server 仅 import 复用，不修改任何 Demo 源文件 |

---

## 六、相关文档

| 文件 | 说明 |
|------|------|
| [CLOUD_EDGE_INTEGRATION_PLAN.md](CLOUD_EDGE_INTEGRATION_PLAN.md) | 完整实施方案（含架构、Phase 1-5、风险评估） |
| [CLOUD_EDGE_INTERFACE.md](CLOUD_EDGE_INTERFACE.md) | 9 个 API 端点完整定义 + Flask 参考实现 + 数据格式详解 |
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | edge-gui 历史实现报告（Phase 1-3） |
| [TASK_COMPLETION_SUMMARY.md](TASK_COMPLETION_SUMMARY.md) | **本文档** — 本次任务完成总结 |
