from pathlib import Path

import pandas as pd
import streamlit as st

from edge_input_adapter import EdgeInputFormatError
from scheduler import (
    AlwaysCloudScheduler,
    AlwaysLocalFirstScheduler,
    PPOScheduler,
    RuleScheduler,
)
from simulator import CloudEdgeSimulator
from task_profiles import profiles_as_dict


st.set_page_config(page_title="Scheduler Training云边协同与蒸馏", page_icon="⚙️", layout="wide")
st.title("Scheduler Training工业云边协同PPO与蒸馏")
st.caption(
    "24D Pre-route状态与LOCAL_FIRST / DIRECT_CLOUD二动作调度，"
    "本地完成后由Post-edge规则决定是否云端复核；"
    "并加入与hhk输入兼容的可插拔教师伪标签和蒸馏数据管线。"
)

with st.sidebar:
    st.header("任务与环境")
    source_label = st.selectbox(
        "任务来源",
        ["工业类型仿真", "边端预处理input", "Phase1旧随机任务"],
    )
    task_source = {
        "工业类型仿真": "typed_simulation",
        "边端预处理input": "edge_input",
        "Phase1旧随机任务": "legacy_simulation",
    }[source_label]

    edge_input_path = None
    if task_source == "edge_input":
        edge_input_path = st.text_input(
            "input JSON路径",
            value="data/sample_edge_inputs.json",
        )

    queue_policy = st.selectbox("队列", ["edf", "fifo"], index=0)
    slot_duration = st.number_input(
        "时隙长度（秒）", 0.01, 1.0, 0.05, 0.01
    )
    arrival_probability = st.slider(
        "每时隙任务到达概率",
        0.01,
        0.80,
        0.10,
        0.01,
        disabled=task_source == "edge_input",
    )
    network_mode = st.selectbox("网络模式", ["markov", "legacy"])
    resource_mode = st.selectbox("资源模式", ["consume_recover", "legacy"])
    reward_mode = st.selectbox(
        "奖励模式", ["deadline_aware", "legacy", "lyapunov"]
    )

    st.header("调度器")
    strategy = st.selectbox(
        "策略",
        [
            "Pre-route规则调度器",
            "固定本地优先",
            "固定直接云端",
            "二动作Pre-route PPO",
        ],
    )
    model_path = st.text_input(
        "PPO模型",
        value="models/releases/pre_route_v2/pre_route_ppo.zip",
        disabled=strategy != "二动作Pre-route PPO",
    )
    seed = int(st.number_input("随机种子", min_value=0, value=42, step=1))
    steps = int(st.number_input("运行时隙数", min_value=10, value=2000, step=100))


def build_scheduler():
    if strategy == "固定本地优先":
        return AlwaysLocalFirstScheduler()
    if strategy == "固定直接云端":
        return AlwaysCloudScheduler()
    if strategy == "二动作Pre-route PPO":
        return PPOScheduler(model_path)
    return RuleScheduler()


if st.button("运行评估", type="primary"):
    try:
        if strategy == "二动作Pre-route PPO" and not Path(model_path).is_file():
            raise FileNotFoundError(f"未找到PPO模型：{model_path}")
        simulator = CloudEdgeSimulator(
            seed=seed,
            arrival_probability=arrival_probability,
            slot_duration=slot_duration,
            scheduler=build_scheduler(),
            queue_policy=queue_policy,
            network_mode=network_mode,
            resource_mode=resource_mode,
            reward_mode=reward_mode,
            task_source=task_source,
            edge_input_path=edge_input_path,
        )
        if task_source == "edge_input":
            simulator.run_until_complete(max_steps=steps)
        else:
            simulator.run(steps)
        st.session_state["simulator"] = simulator
    except (FileNotFoundError, EdgeInputFormatError, ValueError, ImportError) as exc:
        st.error(str(exc))

simulator = st.session_state.get("simulator")
if simulator is not None:
    metrics = simulator.get_metrics()
    columns = st.columns(8)
    values = [
        ("完成率", metrics["completion_rate"]),
        ("超时率", metrics["timeout_rate"]),
        ("平均时延/s", metrics["average_success_latency"]),
        ("P95/s", metrics["latency_p95_s"]),
        ("边端比例", metrics["edge_ratio"]),
        ("云端比例", metrics["cloud_ratio"]),
        ("协同比例", metrics["collaborative_ratio"]),
        ("上传量/MB", metrics["upload_mb"]),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, "—" if value != value else f"{value:.3f}")

    st.subheader("按任务类型")
    st.dataframe(simulator.get_task_type_metrics(), use_container_width=True)
    st.subheader("协同任务完成与超时记录")
    completed = simulator.get_completed_tasks()
    if not completed.empty and "route" in completed:
        st.dataframe(
            completed[completed["route"] == "COLLABORATIVE"].tail(200),
            use_container_width=True,
        )
    st.subheader("最近物理时隙记录")
    st.dataframe(simulator.get_records().tail(200), use_container_width=True)

with st.expander("任务类型配置"):
    profile_df = pd.DataFrame(profiles_as_dict()).T.reset_index(names="task_type")
    st.dataframe(profile_df, use_container_width=True)


st.divider()
st.header("Scheduler Training 蒸馏数据管线")
st.caption(
    "该区域不假设边端学生模型。当前可以验证教师服务、生成教师伪标签和构建数据集；"
    "LoRA训练需在学生模型确定后通过命令行显式执行。"
)

with st.expander("教师服务与数据生成", expanded=False):
    from dataclasses import asdict
    import json

    from distillation.dataset_builder import build_dataset
    from distillation.schema import TeacherConfig
    from distillation.teacher_backends import build_teacher_backend
    from distillation.teacher_pipeline import generate_teacher_outputs

    teacher_backend = st.selectbox(
        "教师后端", ["api", "simulated"],
        help="simulated仅用于流程验证，不能用于真实准确率结论。",
    )
    teacher_api_base = st.text_input(
        "教师API Base", value="http://127.0.0.1:8080/v1",
        disabled=teacher_backend != "api",
    )
    teacher_model = st.text_input(
        "教师模型别名", value="qwen3-8b-teacher",
        disabled=teacher_backend != "api",
    )
    distill_input = st.text_input(
        "hhk任务输入", value="data/sample_edge_inputs.json",
    )
    teacher_output_path = st.text_input(
        "教师输出", value="data/distillation/teacher_outputs.jsonl",
    )
    sample_limit = int(st.number_input("本次最多生成", 1, 10000, 8, 1))
    config = TeacherConfig(
        backend=teacher_backend,
        api_base=teacher_api_base,
        model=teacher_model,
    )

    left, right = st.columns(2)
    if left.button("检查教师服务"):
        result = build_teacher_backend(config).health()
        if result.get("ok"):
            st.success("教师服务可用")
        else:
            st.error("教师服务不可用")
        st.json(result)

    if right.button("生成教师输出"):
        try:
            report = generate_teacher_outputs(
                distill_input,
                teacher_output_path,
                config,
                limit=sample_limit,
                continue_on_error=True,
            )
            st.success("教师输出生成完成")
            st.json(asdict(report))
        except Exception as exc:
            st.error(str(exc))

with st.expander("构建蒸馏训练集", expanded=False):
    build_source = st.text_input(
        "教师输出JSONL", value="data/distillation/teacher_outputs.jsonl",
        key="phase4_build_source",
    )
    target_policy = st.selectbox(
        "训练目标选择",
        ["prefer-ground-truth", "ground-truth-only", "teacher-only"],
        help="默认优先使用真实标签，没有真实标签时才使用教师伪标签。",
    )
    require_truth_test = st.checkbox(
        "测试集必须包含真实标签",
        value=False,
        help="启用后，没有ground_truth会直接报错，适合正式准确率实验。",
    )
    if st.button("构建Phase4数据集"):
        try:
            report = build_dataset(
                build_source,
                "data/distillation/distillation_train.jsonl",
                "data/distillation/distillation_test.jsonl",
                "data/distillation/dataset_manifest.json",
                target_policy=target_policy,
                require_ground_truth_for_test=require_truth_test,
            )
            st.success("数据集构建完成")
            st.json(asdict(report))
        except Exception as exc:
            st.error(str(exc))

with st.expander("学生模型状态", expanded=False):
    registry_path = Path("models/student_registry.json")
    if registry_path.is_file():
        st.json(json.loads(registry_path.read_text(encoding="utf-8")))
    st.info(
        "当前没有确定真实边端学生模型。训练侧不会自动训练、量化或替换边端模型；"
        "确定模型后再运行train_student_lora.py和evaluate_student.py。"
    )
