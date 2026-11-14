"""生成包含8类任务的边端标准input样例。

样例只包含Pre-route决策时已经存在的信息，不预置尚未执行边端推理才能得到的
置信度。仿真所需的边端置信度由任务类型先验在输入适配阶段补齐。
"""

from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np

from task_profiles import TASK_TYPE_ORDER, get_task_profile, sample_log_uniform


def main() -> None:
    rng = np.random.default_rng(20260724)
    start = datetime(2026, 7, 24, 9, 0, 0)
    rows = []
    for index in range(96):
        task_type = TASK_TYPE_ORDER[index % len(TASK_TYPE_ORDER)]
        profile = get_task_profile(task_type)
        timestamp = start + timedelta(seconds=0.15 * index)
        rows.append(
            {
                "task_id": f"T-PHASE2-{index + 1:04d}",
                "node_id": f"edge-node-{index % 3 + 1:02d}",
                "timestamp": timestamp.isoformat(timespec="milliseconds"),
                "task_type": task_type,
                "priority": profile.default_priority,
                "max_latency_ms": int(
                    profile.default_deadline_s
                    * rng.uniform(0.95, 1.50)
                    * 1000
                ),
                "payload_size_mb": round(
                    sample_log_uniform(
                        rng,
                        profile.payload_min_mb,
                        profile.payload_max_mb,
                    ),
                    5,
                ),
                "data": {
                    "device_id": f"device-{index % 20:02d}",
                    "device_type": "industrial_device",
                    "readings": {
                        "feature_mean": round(float(rng.normal(0.0, 1.0)), 4),
                        "feature_std": round(float(rng.uniform(0.1, 2.0)), 4),
                        "feature_peak": round(float(rng.uniform(0.5, 8.0)), 4),
                    },
                    "context": f"{profile.name}预处理特征样例",
                },
            }
        )

    output = Path("data/sample_edge_inputs.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    print(f"已生成{len(rows)}条任务：{output}")


if __name__ == "__main__":
    main()
