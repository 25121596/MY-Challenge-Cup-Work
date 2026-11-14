# AI4I真实云端复核与边云指标

该流程使用同一批 AI4I held-out test 数据，将 HHK V2 已生成的边端概率和
Post-edge 复核标记，与运行在 `llama-server` 上的真实 Qwen3-8B 云端输出按
`task_id` 对齐。它属于单机双服务的真实模型复核，不把云模型输出称为真值，
也不修改 HHK V2 或 Scheduler Training。

## 输入契约

命令显式接收三个只读输入，Integration 不依赖 HHK 内部 Python 模块：

```text
test_predictions.jsonl
  task_id, label, failure_probability, cloud_review

metrics.json
  cloud_review_policy.normal_failure_probability_cutoff

ai4i_milling_qwen_review_v1.jsonl
  task_id, qwen_input_encoding, prompt
```

程序会拒绝重复或缺失 `task_id`、版本不匹配、复核标记与阈值不一致，以及
云端提示中出现 `label`、`ground_truth` 或 `readings`。真实标签只在云端调用
完成后的评估阶段使用。

## 1. 启动云端模型

从仓库根目录运行以下单行命令：

```bat
hhk_v2_snapshot\build\bin\Release\llama-server.exe -m cloud_model\models\Qwen3-8B\Qwen3-8B-Q4_K_M.gguf --host 127.0.0.1 --port 8081 -c 2048 -ngl auto --alias Qwen3-8B-cloud
```

保留该窗口，另开一个 Anaconda Prompt：

```bat
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8081/v1/models
```

## 2. 运行20条分层 pilot

进入 `integration`：

```bat
cd integration
python run_ai4i_cloud_review.py --edge-predictions ..\hhk_v2_snapshot\data\training\ai4i_tabular_moe\test_predictions.jsonl --edge-metrics ..\hhk_v2_snapshot\data\training\ai4i_tabular_moe\metrics.json --qwen-inputs ..\hhk_v2_snapshot\data\standard_inputs\ai4i_milling_qwen_review_v1.jsonl --output-dir outputs\ai4i_edge_cloud_pilot --pilot-per-label 10
```

Pilot 从 Post-edge 子集中分别选择10条正常标签和10条故障标签。它只用于检查
模型服务、输出格式、任务对齐和时延。因为没有覆盖全部881条复核任务，报告
不会声称已经得到完整系统 Accuracy/F1。

## 3. 运行完整 Post-edge 复核

```bat
python run_ai4i_cloud_review.py --edge-predictions ..\hhk_v2_snapshot\data\training\ai4i_tabular_moe\test_predictions.jsonl --edge-metrics ..\hhk_v2_snapshot\data\training\ai4i_tabular_moe\metrics.json --qwen-inputs ..\hhk_v2_snapshot\data\standard_inputs\ai4i_milling_qwen_review_v1.jsonl --output-dir outputs\ai4i_edge_cloud_review --progress-every 10
```

默认只调用 `cloud_review=true` 的881条任务。程序每条完成后立即追加 JSONL；
意外中断后执行同一命令会跳过同一模型、同一提示的已有结果并继续。若需要
重试已经记录的失败或无效输出，在同一命令末尾增加：

```bat
--retry-failed
```

只重新生成对齐记录和报告、不调用模型：

```bat
python run_ai4i_cloud_review.py --edge-predictions ..\hhk_v2_snapshot\data\training\ai4i_tabular_moe\test_predictions.jsonl --edge-metrics ..\hhk_v2_snapshot\data\training\ai4i_tabular_moe\metrics.json --qwen-inputs ..\hhk_v2_snapshot\data\standard_inputs\ai4i_milling_qwen_review_v1.jsonl --output-dir outputs\ai4i_edge_cloud_review --report-only
```

## 交付物

输出目录包含：

```text
cloud_review_results.jsonl
edge_cloud_feedback.jsonl
edge_cloud_metrics.json
edge_cloud_report.md
```

- `cloud_review_results.jsonl`：可恢复的逐任务云端原始结果、版本、解析和时延。
- `edge_cloud_feedback.jsonl`：HHK V2 `edge-cloud-feedback/v1` 平面字段契约。
- `edge_cloud_metrics.json`：机器可读的一致性、正确性和时延指标。
- `edge_cloud_report.md`：可直接交付和复核的汇总报告。

生成物位于已忽略的 `integration/outputs/`，不会进入 Git。

## 指标口径

报告严格区分：

```text
disagreement = edge_prediction != cloud_prediction
review_useful = edge错误 且 cloud正确
cloud_regression = edge正确 且 cloud错误
```

`disagreement` 不是“云端纠正”。`review_useful` 和 Accuracy/F1 使用 AI4I
held-out test 标签计算。

报告还会同时列出两个边端口径：

- `MoE classifier reference (0.5)`：模型本身按0.5阈值分类的参考指标；
- `Edge-only operational (HHK cutoff)`：HHK 版本化低风险阈值产生的实际运行决策。

最终双层系统与第二项比较，因为它才对应无云端时的实际 Edge-only 路径。
如果云调用失败或输出无效，最终系统显式回退到边端结果。
