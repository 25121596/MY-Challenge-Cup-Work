# 蒸馏数据目录

建议生成以下文件：

```text
teacher_outputs.jsonl
  每条包含student_input、teacher_output、label_source和可选ground_truth。

distillation_train.jsonl
  学生LoRA训练集。

distillation_test.jsonl
  学生对照测试集。只有含独立ground_truth时才能评估真实判断准确率。

dataset_manifest.json
  数据筛选、拆分、拒绝原因和样本来源统计。
```

不要把教师伪标签直接命名为“真实标签”。
