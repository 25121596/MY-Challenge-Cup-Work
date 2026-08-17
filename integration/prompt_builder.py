from __future__ import annotations

import json
from typing import Any


def build_common_prompt(task: dict[str, Any]) -> str:
    task_id = task.get("task_id", "unknown")
    task_type = task.get("task_type", "unknown")
    priority = task.get("priority", "medium")
    data = task.get("data", {})

    return f"""你是工业设备分析助手。

请分析下面这一条标准化工业任务。要求：
1. 只输出一个JSON对象，不要使用Markdown代码块；
2. JSON必须包含category、level、confidence、ood、summary；
3. level只能是info、warning、critical之一；
4. confidence必须是0到1之间的数值；
5. ood表示输入是否明显超出当前模型能力或已知分布；
6. category无法可靠判断时使用unknown；
7. summary简短说明结论和最重要依据，不要展开长篇推理。

task_id: {task_id}
task_type: {task_type}
priority: {priority}

data:
{json.dumps(data, ensure_ascii=False, indent=2)}
"""


def build_cloud_review_prompt(
    task: dict[str, Any],
    edge_answer: str,
) -> str:
    return f"""你是云端高能力工业诊断模型，负责复核边端轻量模型的初步判断。

原始任务：
{json.dumps(task, ensure_ascii=False, indent=2)}

边端初步结果：
{edge_answer}

请独立检查原始任务，再参考边端结果完成复核。
如果边端判断不合理，请明确修正。
最后给出简洁的最终判断和关键依据。
"""
