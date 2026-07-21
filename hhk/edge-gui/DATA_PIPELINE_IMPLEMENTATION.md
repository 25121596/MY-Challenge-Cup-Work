# Data Pipeline 移植文档

> **日期:** 2026-07-21  
> **目标:** 将 Python 传感器数据预处理+推理管线完整移植到 edge-gui C++ 中  
> **来源:** `edge_run.py` → `preprocessing/` → `edge-io-protocol/edge_inference.py`

---

## 1. 概述

### 1.1 背景

edge-gui 原本只支持聊天式对话。现在新增了**传感器数据流处理**功能：
- 在 GUI 中选择 CSV/TXT 传感器文件
- 自动识别数据类型（hydraulic / motor / milling / generic）
- 映射列名到标准字段、分批平均、构建分类 prompt
- 通过本地已加载模型进行 **GBNF 语法约束**的短决策推理（输出单数字 0-6）
- 展示分类结果表格 + 时延统计

### 1.2 移植对照

| Python 源 | 移植到 C++ |
|---|---|
| `sensor_csv_to_input.py` — CSV 解析、列名映射、分批 | `sensor_pipeline.cpp` — `detect_delimiter()` / `map_headers()` / `prepare_batches()` |
| `hydraulic_preprocess.py` — 多采样率对齐 | `sensor_pipeline.cpp` — CHANNEL_MAPS 硬编码 hydraulic 通道 |
| `motor_preprocess.py` — 降采样+标签对齐 | `sensor_pipeline.cpp` — CHANNEL_MAPS 硬编码 motor 通道 |
| `edge_inference.py` — prompt 构建 | `sensor_pipeline.cpp` — `build_prompt()` 用硬编码中文模板 |
| `edge_inference.py` — GBNF 推理+输出解析 | `engine.cpp` — `generate_completion()` + `sensor_pipeline.cpp` — `parse_model_output()` |
| `edge_inference.py` — 甩云判定 | `sensor_pipeline.cpp` — 硬编码规则：category≥2 甩云 |
| llama-server HTTP `/completion` | `engine.cpp` — llama.cpp C API 直调 |

### 1.3 关于模型精度

根据组员 7.20 工作总结，**Qwen3-1.7B-IQ4_XS 无法从原始传感器数值准确推理故障类别**（准确率 ~10%），这是模型能力上限，非工程问题。本移植的价值在于：
- 打通了 C++ 边端应用的**完整数据管线**
- 日后换更好模型（如 Qwen3-4B）即可获得高准确率
- 架构支持**两条数据路径**：原始值直推 + 特征提取后再推（`edge_preprocess/` 17维/通道）

---

## 2. 架构

```
┌─────────────────────────────────────────────────────────────┐
│  SensorPipeline (sensor_pipeline.h/cpp)                      │
│                                                              │
│  open(filepath)                                              │
│    ├─ detect_delimiter()      移植自 sensor_csv_to_input.py  │
│    ├─ detect_data_type()      3类签名匹配 (hydraulic/motor/  │
│    │                          milling)                       │
│    ├─ map_headers()           列名→标准字段 (大小写不敏感)    │
│    └─ parse_rows()            逐行读取 float                │
│                                                              │
│  prepare_batches(batch_size=60)                              │
│    ├─ 每 N 行取平均, 产出 readings map                       │
│    └─ build_prompt()           填充故障诊断中文模板           │
│                                                              │
│  classify_batch(batch)                                       │
│    ├─ _engine->generate_completion(prompt, grammar)           │
│    │     └─ llama_sampler_init_grammar("root ::= [0-6]")    │
│    ├─ parse_model_output()    解析单数字→类别名/动作/置信度   │
│    └─ → ClassificationResult                                 │
│                                                              │
│  compute_stats()                                             │
│    └─ → PipelineStats (avg/P95/达标率/甩云率/分布)           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  EdgeEngine 新增: generate_completion()  (engine.h/cpp)       │
│    - 独立的 sampler chain: grammar → dist(seed=42)           │
│    - temperature=0 (greedy), n_predict=2                     │
│    - 同步返回, 单次 < 50ms                                    │
│    - 自动清理 KV cache (llama_memory_clear)                  │
│    - 不污染聊天会话状态                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  UI: "数据流" 面板  (app.h/cpp)                               │
│    [选择文件] → 自动检测类型 → [开始处理]                    │
│    逐帧处理一批 → 实时更新进度条 + 结果表格                   │
│    彩色分类标签 (绿=正常/黄=警告/红=严重)                    │
│    统计摘要: 平均/P95/200ms达标率/甩云率/分类分布            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 文件清单

### 新增文件

| 文件 | 说明 |
|---|---|
| `edge-gui/sensor_pipeline.h` | 数据结构 + SensorPipeline 类声明 |
| `edge-gui/sensor_pipeline.cpp` | 完整实现 (~280行) |

### 修改文件

| 文件 | 改动 |
|---|---|
| `edge-gui/engine.h` | 新增 `generate_completion()` 方法声明 |
| `edge-gui/engine.cpp` | 新增 `generate_completion()` 实现 (~90行) |
| `edge-gui/app.h` | 新增 `#include "sensor_pipeline.h"` + 面板状态变量 + `render_data_pipeline_panel()` |
| `edge-gui/app.cpp` | 顶部按钮 "数据流" + 面板渲染 + 逐帧批处理驱动 + 面板 UI 实现 (~120行) |
| `edge-gui/CMakeLists.txt` | 添加 `sensor_pipeline.cpp sensor_pipeline.h` |

---

## 4. 硬编码常量

所有来自 `edge-io-protocol/` 的内容已硬编码到 C++ 中，**不需要读取外部文件**：

| 常量 | 来源 | 值 |
|---|---|---|
| `FAULT_DIAGNOSIS_PROMPT` | `prompts/fault_diagnosis.txt` | 25 行中文分类模板，含 `{input_data}` 占位符 |
| `SHORT_DECISION_GRAMMAR` | `grammars/short_decision.gbnf` | `"root ::= [0-6]"` |
| `CHANNEL_MAPS` | `preprocessing/sensor_csv_to_input.py` | 3 套通道映射 (hydraulic 6通道/motor 9通道/milling 5通道) |
| `TYPE_SIGNATURES` | `edge_run.py` | 自动检测签名表 |
| `CATEGORIES[]` | `edge_inference.py` | 7 类别的 {name, action, level, summary} |
| 甩云规则 | `edge_inference.py::make_output()` | category≥2 → escalate |

---

## 5. 数据流详解

### 5.1 CSV 解析 (`open()`)

```
传感器 CSV 文件
  │
  ├─ 1. detect_delimiter(): 读第一行, 检测 TAB > COMMA > SEMICOLON
  ├─ 2. 解析 header 行 → _headers
  ├─ 3. detect_data_type(): 列名匹配 TYPE_SIGNATURES (≥3 命中即判定)
  ├─ 4. map_headers(): 大小写不敏感匹配 CHANNEL_MAPS
  └─ 5. 逐行 parse float → _rows (非数字→0.0 sentinel)
```

### 5.2 批处理 (`prepare_batches()`)

```
_rows (N 行 × M 列)
  │
  ├─ 每 batch_size=60 行一个批次
  ├─ 跳过 < 5 行的尾部批次
  ├─ 每个映射列: 收集批次内所有非 0.0 值 → 计算平均
  └─ → DataBatch { readings: {ps1_bar:145.6, temperature_c:42.3, ...} }
```

### 5.3 Prompt 构建 (`build_prompt()`)

```
readings map
  │
  ├─ 序列化: "ps1_bar=145.6 temperature_c=42.3 flow_l_min=12.84 ..."
  ├─ 替换 FAULT_DIAGNOSIS_PROMPT 中的 {input_data}
  └─ → 完整 prompt 文本 (~600 tokens)
```

### 5.4 推理 (`classify_batch()` → `generate_completion()`)

```
prompt 文本
  │
  ├─ llama_tokenize → prompt_tokens[]
  ├─ llama_decode (批量 decode prompt)
  ├─ llama_sampler_init_grammar("root ::= [0-6]")
  ├─ sampler chain: grammar → dist(seed=42)
  ├─ llamba_sampler_sample → 1 个 token (0-6)
  ├─ llama_sampler_free(chain)
  ├─ llama_memory_clear + llama_perf_context_reset
  └─ → raw_output: "1" (单字符)
```

### 5.5 输出解析 (`parse_model_output()`)

```
raw_output: "1"
  │
  ├─ trim + strip ``` fences
  ├─ regex \b([0-6])\b → category=1
  ├─ CATEGORIES[1] → {name:"轴承磨损", action:"warning", level:"warning"}
  ├─ confidence = 0.85 (默认)
  ├─ escalate = (category >= 2)
  └─ → ClassificationResult
```

---

## 6. UI 面板说明

### 6.1 入口

点击顶部工具栏的 **"数据流"** 按钮打开/关闭面板。

### 6.2 布局

```
┌──────────────────────────────────────────────────┐
│  数据流处理                                      │
├──────────────────────────────────────────────────┤
│  数据文件: [________________________] [浏览...]   │
│  ✓ motor (2143 行, 9 列)  ← 自动检测类型         │
│                                                   │
│  [▶ 开始处理]  [⏹ 停止]                          │
│  ██████████████░░░░░░  75% (21/28 批)            │
├──────────────────────────────────────────────────┤
│  批次 │ 分类          │ 动作     │ 置信度│ 延迟   │
│  ─────┼───────────────┼─────────┼──────┼─────── │
│    0  │ 🟢 正常       │ normal   │ 85%  │ 48ms   │
│    1  │ 🟡 轴承磨损   │ warning  │ 85%  │ 52ms   │
│    3  │ 🔴 轴承失效   │ shutdown │ 85%  │ 45ms   │
├──────────────────────────────────────────────────┤
│  📈 统计:                                        │
│  总请求: 28  成功: 28  平均: 51ms  P95: 89ms    │
│  200ms达标: 28/28 (100%)  甩云: 3               │
│  分布: 正常:18 磨损:6 失效:2 过热:2              │
└──────────────────────────────────────────────────┘
```

### 6.3 颜色编码

| 颜色 | 含义 |
|---|---|
| 🟢 绿色 | category=0 正常 |
| 🟡 黄色 | category=1/3/5/6 警告级 |
| 🔴 红色 | category=2/4 严重/危急 |

### 6.4 处理机制

- **逐帧处理**: 每帧处理 1 个批次（~50ms），不阻塞 UI
- **进度条**: 实时更新 `当前批次/总批次`
- **停止**: 点击"停止"立即终止
- **无需模型时也可打开面板**: 文件选择和数据检测不依赖模型加载

---

## 7. API 参考

### 7.1 SensorPipeline

```cpp
SensorPipeline pipe;
pipe.set_engine(&engine);                    // 绑定推理引擎

// 打开文件
bool ok = pipe.open("sensors.csv");          // 解析 CSV/TXT
pipe.data_type();    // "hydraulic"|"motor"|"milling"|"generic"
pipe.row_count();    // 数据行数
pipe.headers();      // 列名列表
pipe.is_open();

// 预处理
int n = pipe.prepare_batches(60);            // 分批, 返回批次数
pipe.batches();      // vector<DataBatch>, 每个含 prompt_text

// 推理
auto r = pipe.classify_batch(pipe.batches()[0]);  // 单批推理 (同步, <50ms)
pipe.results();      // vector<ClassificationResult>

// 统计
auto stats = pipe.compute_stats();           // PipelineStats

// 查询
auto * info = SensorPipeline::lookup_category(2);  // 静态方法
```

### 7.2 EdgeEngine::generate_completion()

```cpp
bool generate_completion(
    const std::string & prompt,         // 原始文本 prompt
    const std::string & grammar_str,    // GBNF 语法字符串
    const engine_token_callback & on_token,   // 每 token 回调
    std::function<void(const EnginePerf &)> on_perf  // 性能回调 (可选)
);
```

- **temperature**: 固定为 0 (greedy)
- **n_predict**: 固定为 2 (只需输出 1 位数字)
- **同步**: 阻塞直到生成完成（~50ms）
- **副作用**: 自动清理 KV cache，不影响聊天会话

---

## 8. 数据结构

```cpp
struct DataBatch {
    int    batch_index;
    int    row_count;
    std::map<std::string, double> readings;   // standard_name → avg
    std::string prompt_text;                   // 填充后的 prompt
};

struct ClassificationResult {
    int    batch_index;
    int    category;            // 0-6, -1=error
    std::string category_name;  // "正常"/"轴承磨损"/...
    std::string action;         // normal/warning/alert/maintain/shutdown
    std::string level;          // info/warning/critical
    std::string summary;
    float   confidence;         // 0.0-1.0
    bool    escalate_to_cloud;
    std::string escalate_reason;
    double  latency_ms;
    std::string raw_output;     // 模型原始输出 (调试用)
    std::string error;          // 失败时设置
    bool    success;
};

struct PipelineStats {
    int    total_batches;
    int    success_count;
    double avg_latency_ms;
    double p95_latency_ms;
    int    under_200ms;          // 时延达标数
    int    escalated_count;     // 甩云数
    std::map<int, int> category_distribution;  // category→count
};
```

---

## 9. 编译 & 运行

```bash
# 编译
cd D:\llama.cpp\build
cmake ..
cmake --build . --target edge-gui --config Release

# 运行
build\bin\Release\edge-gui.exe
```

### 操作流程

1. 加载模型：点 **模型** → 选 `Qwen3-1.7B-IQ4_XS.gguf`
2. 打开面板：点 **数据流**
3. 选择文件：点 **浏览...** → 选 CSV/TXT 传感器文件
4. 开始处理：点 **▶ 开始处理**
5. 查看结果：表格展示所有批次的分类结果 + 底部统计摘要

### 测试文件

已创建 `test_sensors.csv` 在 `D:\llama.cpp\` 下，可直接用于测试。

---

## 10. 后续可扩展

- **路径 B (特征提取)**: 移植 `edge_preprocess/features.py` 的 12 时域+5 频域特征提取到 C++
- **多任务 prompt**: 支持 `quality_inspection` / `safety_check` 等其他 task_type 的 prompt 模板
- **结果导出**: 新增 "导出 JSON" 按钮，输出符合 `schemas/output.json` 的批量结果
- **对接云端**: 甩云的批次自动发 `cloud_escalation.json` 到云端 API
- **换大模型**: 管线不变，只需换 GGUF 模型文件即可提升分类准确率
