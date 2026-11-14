# 边端模型 I/O 协议

> 面向云边协同场景的分布式人工智能感知与决策 — 边端小组

---

## 核心思路

就像 USB 接口一样，边端模型需要一个统一的"插头标准"：
- **输入**：不管什么数据源，都套同一个 JSON 模板喂给模型
- **输出**：模型返回结构化 JSON，下游系统（SCADA、报警、云端）直接消费
- **边云交互**：边端不确定的任务，用标准格式甩给云端大模型

## 文件结构

```
edge-io-protocol/
├── README.md                   ← 你在看
│
├── schemas/                    ← JSON 模板定义
│   ├── input.json              输入格式（传感器→模型）
│   ├── output.json             输出格式（模型→下游系统）
│   └── cloud_escalation.json   边端→云端 交互格式
│
├── prompts/                    ← 各场景 Prompt 模板
│   ├── fault_diagnosis.txt     故障诊断
│   ├── quality_inspection.txt  质量检测
│   ├── safety_check.txt        安全识别
│   └── data_analysis.txt       数据分析
│
├── task_types.json             ← 任务分类表
│
└── edge_inference.py           ← 示例代码框架（可直接跑）
```

## 怎么用

1. 把 `schemas/` 下的 JSON 模板给后端开发，他们按这个写数据接口
2. 把 `prompts/` 下的模板灌进模型推理代码
3. 把 `task_types.json` 给架构组，这是你的任务分类体系
4. 运行 `edge_inference.py` 跑一个完整的"边端推理 → 云端确认"流程示例

## 题目指标对应

| 题目要求 | 对应设计 |
|---|---|
| 单次推理 ≤ 1.5GB | IQ4_XS 量化 + 精简 prompt + 限制 context |
| 能力保持 80-90% | prompt 工程 + 置信度阈值 + 云端兜底 |
| TTFT 减少 75% | 短 prompt + 小模型 + GPU offload |
| 端到端 ≤ 0.2s | n_predict 限制在 100 tokens 以内 |
| 冲突率 ≤ 5% | decision_hash 机制 + 云端仲裁 |
