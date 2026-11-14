# 边端系统链路状态报告

> 更新时间: 2026-07-21
> 职责范围: 边端模型与边缘节点系统（不含甩云策略、云端模型、边云通信协议）

---

## 一、整体架构

```
原始数据 (CSV/MAT)
  │
  ├─[A] preprocessing/ (直接转标准I/O)
  │     ├── hydraulic_preprocess.py   ← UCI液压
  │     ├── motor_preprocess.py       ← Nature三相电机
  │     ├── sensor_csv_to_input.py    ← AI4I铣削 / 通用CSV
  │     └── anomaly_injector.py       ← 故障注入(测试用)
  │
  ├─[B] edge_preprocess/ (特征提取 → CSV)
  │     ├── features.py / stream.py / datasets.py / schema.py
  │     └── → features_to_input.py (桥接到标准I/O)
  │
  ▼
标准 I/O input.json (edge-io-protocol/schemas/input.json)
  │
  ▼
edge_inference.py / edge_run.py
  ├── 加载 prompt 模板 (prompts/*.txt)
  ├── 加载 GBNF grammar (grammars/short_decision.gbnf)
  ├── 调用 llama-server /completion (n_predict=2)
  ├── 解析短决策输出 (单数字 0-6)
  └── 组装标准 output.json
  │
  ▼
分类结果 + 时延指标 + 甩云判断
```

---

## 二、已完成 ✅

### 2.1 数据输入管线

| 数据源 | 预处理脚本 | 标准I/O | 推理验证 |
|---|---|---|---|
| UCI 液压 (22文件, 多采样率) | `hydraulic_preprocess.py` | ✅ | ✅ 108ms |
| Nature 三相电机 (10×100万行) | `motor_preprocess.py` | ✅ | ✅ 127ms |
| AI4I 铣削 (10000条, 含文本列) | `sensor_csv_to_input.py --type milling` | ✅ | ✅ 124ms |
| edge_preprocess 特征CSV | `features_to_input.py` (桥接) | ✅ | ✅ 三源全通 |
| 通用 CSV | `sensor_csv_to_input.py --type generic` | ✅ | 未单独验证 |

### 2.2 推理管线

| 组件 | 状态 | 说明 |
|---|---|---|
| llama-server (CPU模式) | ✅ | Qwen3-1.7B-IQ4_XS, ngl=0, ctx=1024 |
| GBNF grammar 约束输出 | ✅ | 输出限制为单数字 [0-6] |
| 短决策解析器 | ✅ | 兼容: 单数字 / class,risk / JSON 三种格式 |
| prompt 模板 (6种任务) | ✅ | fault_diagnosis / quality_inspection / safety_check / data_analysis / status_monitoring / predictive_maintenance |
| 标准 output.json 组装 | ✅ | 含 decision_hash 一致性字段 |
| 甩云判断逻辑 | ✅ | 格式错误/低置信度/高风险 → 标记甩云 |

### 2.3 统一入口

| 工具 | 用法 | 功能 |
|---|---|---|
| `edge_run.py` | `python edge_run.py --data <路径>` | 自动检测数据类型 → 预处理 → 启动服务 → 推理 → 输出报告 |
| `demo/run_demo.bat` | 双击运行 | 一键演示 (启动服务+预处理+推理+场景执行) |
| `scenarios/scenario_runner.py` | `python scenario_runner.py industrial_inspection` | 场景配置驱动, 含分层抽样+分类指标+比赛指标对照 |

### 2.4 评测工具

| 工具 | 功能 |
|---|---|
| `llama/benchmark.py` | A卷通用能力 + B卷工业场景, 双模型对比 |
| `llama/memcheck.py` | 5种模式内存检测 (edge/gpu/compare/list/interactive) |
| `llama/monitor.py` | Flask :8081 实时监控面板 |
| `evaluation/metrics_collector.py` | 时延P50/P95/P99 + 吞吐 + 达标率 |
| `evaluation/report_generator.py` | 7节 Markdown 评测报告 |

### 2.5 性能实测 (CPU, ngl=0, ctx=1024)

| 指标 | 实测值 | 比赛要求 | 状态 |
|---|---|---|---|
| 平均端到端时延 | **108ms** | ≤200ms | ✅ |
| P95 时延 | **122ms** | — | ✅ |
| 200ms 达标率 | **100%** | ≥95% | ✅ |
| 输出 token 数 | **1-2** | — | ✅ 极短 |
| 格式成功率 | **100%** | ≥99% | ✅ |
| 生成速度 | **65 t/s** | — | — |

---

## 三、未完成 ❌

### 3.1 边端六项指标缺口

| 指标 | 状态 | 缺什么 |
|---|---|---|
| 能力保持率 80%-90% | ⚠️ 数据有但口径不对 | 需: 扩充题库(每维度≥10题), temperature=0, 正确计算 IQ4÷Q8 比值 |
| TTFT 减少 75% | ⚠️ 有数据但基准不明 | 需: 明确"云端基线"是什么, 统一测量口径 |
| 内存 ≤1.5GB | ⚠️ 1.46GB 勉强 | 需: ctx 2048→1024 重测, 压力测试, 确认峰值 |
| 断网业务保持 ≥90% | ❌ 未测试 | 需: network_simulator.py + 离线推理验证 |
| ≥2 类场景 | ⚠️ 只有工业 | 需: 第二场景接入 (交通或其他) |
| 多节点冲突率 ≤5% | ❌ 未实现 | 需: multi_edge_simulator.py + 本地仲裁器 |
| 冲突解决率 ≥90% | ❌ 未实现 | 需: conflict_resolver.py |

### 3.2 输入侧状态

| 项目 | 说明 |
|---|---|
| edge-gui ImageSource | ✅ 完整 (stb_image, PNG/JPG/WEBP/BMP) |
| edge-gui VideoSource | ✅ 完整 (Win Media Foundation, RGB32→RGB24) |
| edge-gui CameraSource | ✅ 完整 (Win MF 枚举摄像头, 实时捕获) |
| edge-gui DataStreamSource | ✅ 完整 (CSV/JSON 自动检测, 批量读取) |
| edge-gui TextStreamSource | ✅ 完整 (大文件分块, token估算) |
| Python 流式适配器 | ✅ stream_adapter.py (TCP/模拟流/目录监听) |
| 实时流式输入 (MQTT/OPC-UA) | ❌ 不存在 (但 TCP socket 已支持) |
| image_url 图像处理管线 | ⚠️ 接口完整但模型为纯文本, 无法做视觉推理 |
| 9 个未下载数据集 | FMF/RealIAD/HUST/Paderborn/NASA_CMAPSS/CWRU/XJTU_SY/SEU/MulSen_AD |

### 3.3 模型精度问题

当前模型 (Qwen3-1.7B-IQ4_XS) 在短决策模式下**分类精度低**:
- 大部分输入被分类为 "1" (轴承磨损)
- 正常数据也经常被误判
- 这是模型能力问题, 不是管线问题

可能改善方向:
- 优化 prompt (更明确的阈值规则)
- 增加 few-shot 示例
- 候选标签 logits 打分 (替代自由生成)
- 换用更大模型 (Qwen3-4B IQ3) — 但内存会超

---

## 四、文件清单

```
d:\Challenge Cup\
├── edge_run.py                     ← 🆕 统一入口 (一条命令出结果)
│
├── edge-io-protocol/               ← I/O 协议 + 推理框架
│   ├── schemas/ (input/output/cloud_escalation)
│   ├── prompts/ (6个任务模板, 含新增 status_monitoring.txt)
│   ├── grammars/short_decision.gbnf
│   ├── task_types.json
│   └── edge_inference.py           ← 核心推理入口 (已修复)
│
├── 7.20/competition-main/hhk/
│   ├── preprocessing/              ← 数据预处理
│   │   ├── hydraulic_preprocess.py
│   │   ├── motor_preprocess.py
│   │   ├── sensor_csv_to_input.py  ← 已修复混合列解析
│   │   ├── anomaly_injector.py
│   │   ├── data_split.py
│   │   └── features_to_input.py    ← 🆕 特征CSV桥接
│   │
│   ├── edge_preprocess/            ← 流式特征提取 (17维/通道)
│   │   ├── features.py / stream.py / datasets.py / schema.py
│   │
│   ├── scenarios/
│   │   ├── industrial_inspection.json  ← 已修复 pump-b02
│   │   ├── traffic_monitoring.json
│   │   └── scenario_runner.py      ← 已修复编码+路径
│   │
│   ├── demo/
│   │   ├── demo_controller.py      ← 已升级为短决策模式
│   │   └── run_demo.bat            ← 已更新(自动启服务)
│   │
│   ├── evaluation/
│   │   ├── metrics_collector.py
│   │   └── report_generator.py
│   │
│   ├── processed/                  ← 预处理产物
│   │   ├── *_features.csv          (edge_preprocess 输出)
│   │   ├── *_standard_inputs.json  (桥接后的标准I/O)
│   │   └── *_split_meta.json       (数据划分元信息)
│   │
│   └── cloud_server/               ← 云端服务 (config已修复)
│
├── llama/                          ← 推理二进制 + 评测工具
│   ├── llama-server.exe
│   ├── Qwen_Qwen3-1.7B-IQ4_XS.gguf  (边端模型, 1.18GB)
│   ├── Qwen3-1.7B-Q8_0.gguf         (基线模型, 2.01GB)
│   ├── benchmark.py
│   ├── memcheck.py
│   └── monitor.py
│
└── Data file/                      ← 原始数据集
    ├── UCI_Hydraulic/ (22文件)     ✅ 已对接
    ├── Nature_3phase_motor/ (22)   ✅ 已对接
    ├── AI4I_2020_milling/ (1)      ✅ 已对接
    └── (9个空目录)                  ❌ 待下载
```

---

## 五、快速复现

```bash
# 一条命令跑通全链路 (需要 llama-server 在 8090 端口运行)
cd "d:\Challenge Cup\7.20\competition-main\hhk"
python edge_run.py --data "D:\Challenge Cup\Data file\UCI_Hydraulic" --endpoint http://127.0.0.1:8090

# 或者让 edge_run.py 自动启动服务器 (用默认 8080 端口)
python edge_run.py --data "D:\Challenge Cup\Data file\UCI_Hydraulic"

# 跑完整场景 (含故障注入 + 分类指标)
python scenarios/scenario_runner.py industrial_inspection --endpoint http://127.0.0.1:8090

# 跑 demo
python demo/demo_controller.py --scenario industrial_inspection --edge-endpoint http://127.0.0.1:8090
```

---

## 六、下一步优先级

| 优先级 | 事项 | 预计工作量 |
|---|---|---|
| 🔴 P0 | 能力保持率正确测量 (扩充题库 + Q8基线对比) | 1天 |
| 🔴 P0 | 内存压到 1.35GB (ctx=1024 + 压力测试) | 半天 |
| 🔴 P0 | 断网业务保持测试 | 1天 |
| 🟡 P1 | 第二场景接入 (复用 edge_run.py 框架) | 1天 |
| 🟡 P1 | 多节点模拟器 + 本地仲裁 | 2天 |
| 🟡 P1 | 模型分类精度优化 (prompt工程) | 持续 |
| 🟢 P2 | 详细时延分解 (TTFT/排队/生成/解析) | 半天 |
| 🟢 P2 | 连续运行稳定性 (30分钟+500次请求) | 半天 |
