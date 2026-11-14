"""Phase3工业云边PPO调度仿真器。

保留Phase2的50ms时隙、EDF、8类任务和edge_input适配；新增第三动作：
COLLABORATIVE = 边端初判 -> 独立传输阶段 -> 云端复核。

当前仍是任务级仿真，不调用真实1.5B/7B模型。协同任务始终只产生一条最终
完成或超时记录，并且无论在哪个阶段超时，route都保持COLLABORATIVE。
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edge_input_adapter import EdgeInputLoadReport, load_edge_inputs
from environment import EnvironmentDynamics
from metrics import MetricsTracker
from models import Action, Task, TaskStatus
from reward import RewardCalculator, RewardConfig
from scheduler import RuleScheduler, estimate_delays_for_task
from task_profiles import sample_profile_values, sample_task_type


VALID_QUEUE_POLICIES = {"fifo", "edf"}
VALID_TASK_SOURCES = {"legacy_simulation", "typed_simulation", "edge_input"}


class CloudEdgeSimulator:
    def __init__(
        self,
        seed: int = 42,
        arrival_probability: float = 0.10,
        slot_duration: float = 0.05,
        scheduler: object | None = None,
        latency_weight: float = 2.0,
        queue_penalty_weight: float = 0.2,
        timeout_penalty: float = 20.0,
        accuracy_penalty_weight: float = 3.0,
        communication_penalty_weight: float = 0.25,
        edge_compute_penalty_weight: float = 0.05,
        cloud_compute_penalty_weight: float = 0.05,
        cloud_accuracy: float = 0.95,
        queue_policy: str = "edf",
        network_mode: str = "markov",
        resource_mode: str = "consume_recover",
        reward_mode: str = "deadline_aware",
        task_source: str = "typed_simulation",
        edge_input_path: str | None = None,
        edge_input_strict: bool = True,
        edge_input_interval_s: float = 0.05,
    ) -> None:
        queue_policy = queue_policy.lower()
        task_source = task_source.lower()
        if slot_duration <= 0:
            raise ValueError("slot_duration必须大于0")
        if not 0.0 <= arrival_probability <= 1.0:
            raise ValueError("arrival_probability必须位于[0, 1]")
        if queue_policy not in VALID_QUEUE_POLICIES:
            raise ValueError(f"非法queue_policy：{queue_policy}")
        if task_source not in VALID_TASK_SOURCES:
            raise ValueError(f"非法task_source：{task_source}")
        if edge_input_interval_s <= 0:
            raise ValueError("edge_input_interval_s必须大于0")

        self.seed = int(seed)
        self.arrival_probability = float(arrival_probability)
        self.slot_duration = float(slot_duration)
        self.scheduler = scheduler if scheduler is not None else RuleScheduler()

        self.latency_weight = float(latency_weight)
        self.queue_penalty_weight = float(queue_penalty_weight)
        self.timeout_penalty = float(timeout_penalty)
        self.accuracy_penalty_weight = float(accuracy_penalty_weight)
        self.communication_penalty_weight = float(communication_penalty_weight)
        self.edge_compute_penalty_weight = float(edge_compute_penalty_weight)
        self.cloud_compute_penalty_weight = float(cloud_compute_penalty_weight)
        self.cloud_accuracy = float(cloud_accuracy)

        self.queue_policy = queue_policy
        self.network_mode = network_mode
        self.resource_mode = resource_mode
        self.reward_mode = reward_mode
        self.task_source = task_source
        self.edge_input_path = None if edge_input_path is None else str(Path(edge_input_path))
        self.edge_input_strict = bool(edge_input_strict)
        self.edge_input_interval_s = float(edge_input_interval_s)

        self.external_tasks: deque[Task] = deque()
        self.edge_input_load_report: EdgeInputLoadReport | None = None
        if self.task_source == "edge_input":
            if not self.edge_input_path:
                raise ValueError("edge_input模式必须提供edge_input_path")
            tasks, report = load_edge_inputs(
                self.edge_input_path,
                strict=self.edge_input_strict,
            )
            self.external_tasks.extend(tasks)
            self.edge_input_load_report = report

        self._environment = EnvironmentDynamics(
            slot_duration=self.slot_duration,
            resource_mode=self.resource_mode,
            network_mode=self.network_mode,
        )
        self._reward_calculator = RewardCalculator(
            RewardConfig(
                latency_weight=self.latency_weight,
                queue_penalty_weight=self.queue_penalty_weight,
                timeout_penalty=self.timeout_penalty,
                accuracy_penalty_weight=self.accuracy_penalty_weight,
                communication_penalty_weight=self.communication_penalty_weight,
                edge_compute_penalty_weight=self.edge_compute_penalty_weight,
                cloud_compute_penalty_weight=self.cloud_compute_penalty_weight,
                cloud_accuracy=self.cloud_accuracy,
                reward_mode=self.reward_mode,
            )
        )
        self._metrics = MetricsTracker()
        self.rng = np.random.default_rng(self.seed)
        self.current_time = 0
        self.next_task_id = 1
        self.generated_task_count = 0
        self.edge_queue: deque[dict] = deque()
        self.cloud_queue: deque[dict] = deque()
        self.transmission_queue: deque[dict] = deque()
        self.edge_active: dict | None = None
        self.cloud_active: dict | None = None
        self.cumulative_reward = 0.0

    @property
    def edge_resource(self) -> float:
        return self._environment.edge_resource

    @edge_resource.setter
    def edge_resource(self, value: float) -> None:
        self._environment.edge_resource = float(value)

    @property
    def cloud_resource(self) -> float:
        return self._environment.cloud_resource

    @cloud_resource.setter
    def cloud_resource(self, value: float) -> None:
        self._environment.cloud_resource = float(value)

    @property
    def network_quality(self) -> float:
        return self._environment.network_quality

    @network_quality.setter
    def network_quality(self, value: float) -> None:
        self._environment.network_quality = float(value)
        self._environment.network_state = "GOOD" if value >= 0.5 else "BAD"

    @property
    def network_state(self) -> str:
        return self._environment.network_state

    @property
    def bandwidth_mbps(self) -> float:
        return self._environment.bandwidth_mbps

    @property
    def records(self) -> list[dict]:
        return self._metrics.records

    @property
    def completed_tasks(self) -> list[dict]:
        return self._metrics.completed_tasks

    @property
    def scheduler_name(self) -> str:
        return getattr(self.scheduler, "name", self.scheduler.__class__.__name__)

    def _generate_legacy_task(self) -> Task:
        task = Task(
            task_id=self.next_task_id,
            arrival_time=self.current_time,
            size_mb=float(self.rng.uniform(2.0, 20.0)),
            deadline_s=float(self.rng.uniform(3.0, 12.0)),
            confidence=float(self.rng.uniform(0.50, 0.99)),
            task_type="generic_anomaly",
            priority="medium",
            complexity=0.5,
            edge_base_latency_s=0.15,
            cloud_base_latency_s=0.15,
            estimated_cloud_accuracy=self.cloud_accuracy,
            source="legacy_simulation",
            confidence_source="legacy_random",
            size_source="legacy_random",
            deadline_source="legacy_random",
            performance_source="legacy_formula_compatible",
        )
        self.next_task_id += 1
        return task

    def _generate_typed_task(self) -> Task:
        task_type = sample_task_type(self.rng)
        values = sample_profile_values(task_type, self.rng)
        task = Task(
            task_id=self.next_task_id,
            arrival_time=self.current_time,
            size_mb=float(values["size_mb"]),
            deadline_s=float(values["deadline_s"]),
            confidence=float(values["confidence"]),
            external_task_id=f"SIM-{self.next_task_id:06d}",
            node_id="sim-edge-01",
            task_type=task_type,
            priority=str(values["priority"]),
            device_id=f"device-{self.next_task_id % 20:02d}",
            device_type="simulated_industrial_device",
            complexity=float(values["complexity"]),
            edge_base_latency_s=float(values["edge_base_latency_s"]),
            cloud_base_latency_s=float(values["cloud_base_latency_s"]),
            estimated_cloud_accuracy=float(values["cloud_accuracy"]),
            collaborative_edge_latency_s=float(values["collaborative_edge_latency_s"]),
            collaborative_cloud_latency_s=float(values["collaborative_cloud_latency_s"]),
            collaborative_accuracy=float(values["collaborative_accuracy"]),
            collaborative_payload_ratio=float(values["collaborative_payload_ratio"]),
            edge_input={
                "task_id": f"SIM-{self.next_task_id:06d}",
                "task_type": task_type,
                "priority": str(values["priority"]),
            },
            confidence_source="task_type_distribution",
            size_source="task_type_distribution",
            deadline_source="task_type_distribution",
            performance_source="task_type_profile",
            source="typed_simulation",
        )
        self.next_task_id += 1
        return task

    def _external_task_due(self, task: Task) -> bool:
        elapsed_s = self.current_time * self.slot_duration
        if task.source_arrival_offset_s is not None:
            return elapsed_s + 1e-12 >= task.source_arrival_offset_s
        file_index = max(task.task_id - 1, 0)
        return elapsed_s + 1e-12 >= file_index * self.edge_input_interval_s

    def generate_task(self) -> Task | None:
        if self.task_source == "edge_input":
            if not self.external_tasks:
                return None
            next_task = self.external_tasks[0]
            if not self._external_task_due(next_task):
                return None
            task = self.external_tasks.popleft()
            task.arrival_time = self.current_time
            self.generated_task_count += 1
            return task

        if self.rng.random() > self.arrival_probability:
            return None
        task = (
            self._generate_legacy_task()
            if self.task_source == "legacy_simulation"
            else self._generate_typed_task()
        )
        self.generated_task_count += 1
        return task

    def get_node_workload(self, queue: deque, active: dict | None) -> float:
        return self._environment.node_workload(queue, active, self.slot_duration)

    def get_state(self) -> dict:
        edge_workload_s = self.get_node_workload(self.edge_queue, self.edge_active)
        cloud_ready_workload_s = self.get_node_workload(self.cloud_queue, self.cloud_active)
        reserved_cloud_workload_s = sum(
            int(item.get("stage2_slots", 0)) * self.slot_duration
            for item in self.transmission_queue
        )
        return {
            "current_time": self.current_time,
            "current_time_s": self.current_time * self.slot_duration,
            "edge_queue": len(self.edge_queue),
            # 观测中的cloud_queue包含正在传输但已承诺给云端的协同任务。
            "cloud_queue": len(self.cloud_queue) + len(self.transmission_queue),
            "cloud_ready_queue": len(self.cloud_queue),
            "transmission_queue": len(self.transmission_queue),
            "edge_busy": self.edge_active is not None,
            "cloud_busy": self.cloud_active is not None,
            "edge_active_task": None if self.edge_active is None else self.edge_active["task"].task_id,
            "cloud_active_task": None if self.cloud_active is None else self.cloud_active["task"].task_id,
            "edge_workload_s": edge_workload_s,
            "cloud_workload_s": cloud_ready_workload_s + reserved_cloud_workload_s,
            "cloud_ready_workload_s": cloud_ready_workload_s,
            "reserved_cloud_workload_s": reserved_cloud_workload_s,
            "edge_resource": self.edge_resource,
            "cloud_resource": self.cloud_resource,
            "network_quality": self.network_quality,
            "network_state": self.network_state,
            "bandwidth_mbps": self.bandwidth_mbps,
            "queue_policy": self.queue_policy,
            "network_mode": self.network_mode,
            "resource_mode": self.resource_mode,
            "reward_mode": self.reward_mode,
            "task_source": self.task_source,
            "external_tasks_remaining": len(self.external_tasks),
            "edge_inputs_loaded": (
                0
                if self.edge_input_load_report is None
                else self.edge_input_load_report.loaded_count
            ),
        }

    @staticmethod
    def _slots(duration_s: float, slot_duration: float, *, at_least_one: bool = True) -> int:
        slots = math.ceil(max(0.0, float(duration_s)) / slot_duration)
        return max(1, slots) if at_least_one else max(0, slots)

    def enqueue_task(self, task: Task, action: Action, details: dict) -> None:
        if action == Action.EDGE:
            service_time = float(details["edge_service_time"])
            target_queue = self.edge_queue
            stage = "EDGE_WAITING"
        elif action == Action.CLOUD:
            service_time = float(details["cloud_service_time"])
            target_queue = self.cloud_queue
            stage = "CLOUD_WAITING"
        else:
            raise ValueError("enqueue_task只接收EDGE或CLOUD")

        service_slots = self._slots(service_time, self.slot_duration)
        task.route = action
        task.status = TaskStatus.WAITING
        task.execution_stage = stage
        target_queue.append(
            {
                "task": task,
                "service_slots": service_slots,
                "remaining_slots": service_slots,
                "route": action,
                "stage": stage,
                "decision_details": dict(details),
            }
        )

    def _enqueue_collaborative_stage1(self, task: Task, details: dict) -> None:
        service_slots = self._slots(
            float(details["collaborative_edge_stage_s"]),
            self.slot_duration,
        )
        task.route = Action.COLLABORATIVE
        task.status = TaskStatus.WAITING
        task.execution_stage = "EDGE_WAITING"
        task.edge_stage_done = False
        task.uploaded_to_cloud = False
        self.edge_queue.append(
            {
                "task": task,
                "service_slots": service_slots,
                "remaining_slots": service_slots,
                "route": Action.COLLABORATIVE,
                "stage": "EDGE_STAGE",
                "stage1_slots": service_slots,
                "decision_details": dict(details),
            }
        )

    def _start_collaborative_transmission(self, edge_item: dict) -> None:
        """边端阶段结束后进入真实推进时间的传输阶段。"""

        task: Task = edge_item["task"]
        task.edge_stage_done = True
        task.edge_stage_finish_time = self.current_time
        task.status = TaskStatus.TRANSMITTING
        task.execution_stage = "TRANSMITTING"
        task.edge_draft_result = {
            "category": "simulated_edge_draft",
            "confidence": float(task.confidence),
            "summary": "Phase3仿真边端初步判断；真实Backend接入后由模型输出替换",
        }

        # 同一个统一估计函数在阶段切换时按当前资源/网络重新计算实际后续服务量。
        transition_details = estimate_delays_for_task(task, self.get_state())
        upload_mb = float(transition_details["collaborative_upload_mb"])
        upload_time_s = float(transition_details["collaborative_upload_time_s"])
        network_rtt_s = float(transition_details["collaborative_network_rtt_s"])
        transfer_total_s = float(transition_details["collaborative_transfer_total_s"])
        cloud_stage_s = float(transition_details["collaborative_cloud_stage_s"])
        transfer_slots = self._slots(transfer_total_s, self.slot_duration)
        stage2_slots = self._slots(cloud_stage_s, self.slot_duration)

        transmission_start_slot = self.current_time + 1
        ready_slot = transmission_start_slot + transfer_slots
        task.transmission_start_time = transmission_start_slot
        task.transmission_ready_time = ready_slot
        task.collaborative_request = {
            "task_id": task.external_task_id or task.task_id,
            "edge_draft": task.edge_draft_result,
            "compressed_payload_mb": upload_mb,
            "request_type": "cloud_review",
        }

        self.transmission_queue.append(
            {
                "task": task,
                "route": Action.COLLABORATIVE,
                "stage": "TRANSMITTING",
                "stage1_slots": int(edge_item["stage1_slots"]),
                "edge_start_slot": int(edge_item.get("edge_start_slot", task.start_time or task.arrival_time)),
                "edge_finish_slot": self.current_time,
                "transmission_start_slot": transmission_start_slot,
                "transmission_ready_slot": ready_slot,
                "transfer_slots": transfer_slots,
                "stage2_slots": stage2_slots,
                "upload_mb": upload_mb,
                "upload_time_estimated_s": upload_time_s,
                "network_rtt_estimated_s": network_rtt_s,
                "transfer_total_estimated_s": transfer_total_s,
                "cloud_stage_estimated_s": cloud_stage_s,
                "decision_details": edge_item.get("decision_details", {}),
            }
        )

    def _promote_ready_transmissions(self) -> list[int]:
        promoted: list[int] = []
        remaining: deque[dict] = deque()
        while self.transmission_queue:
            item = self.transmission_queue.popleft()
            if self.current_time >= int(item["transmission_ready_slot"]):
                task: Task = item["task"]
                task.uploaded_to_cloud = True
                task.status = TaskStatus.WAITING
                task.execution_stage = "CLOUD_REVIEW_WAITING"
                item["stage"] = "CLOUD_REVIEW"
                item["service_slots"] = int(item["stage2_slots"])
                item["remaining_slots"] = int(item["stage2_slots"])
                self.cloud_queue.append(item)
                promoted.append(task.task_id)
            else:
                remaining.append(item)
        self.transmission_queue.extend(remaining)
        return promoted

    def _sort_queue_if_needed(self, queue: deque) -> None:
        if self.queue_policy != "edf" or len(queue) <= 1:
            return
        ordered = sorted(
            queue,
            key=lambda item: (
                item["task"].arrival_time * self.slot_duration
                + item["task"].deadline_s,
                item["task"].arrival_time,
                item["task"].task_id,
            ),
        )
        queue.clear()
        queue.extend(ordered)

    def start_next_task(self, route: Action) -> None:
        if route == Action.EDGE:
            if self.edge_active is not None or not self.edge_queue:
                return
            self._sort_queue_if_needed(self.edge_queue)
            item = self.edge_queue.popleft()
            self.edge_active = item
            task: Task = item["task"]
            if task.route == Action.COLLABORATIVE:
                item["edge_start_slot"] = self.current_time
                task.edge_stage_start_time = self.current_time
                task.execution_stage = "EDGE_RUNNING"
            else:
                task.execution_stage = "EDGE_RUNNING"
        elif route == Action.CLOUD:
            if self.cloud_active is not None or not self.cloud_queue:
                return
            self._sort_queue_if_needed(self.cloud_queue)
            item = self.cloud_queue.popleft()
            self.cloud_active = item
            task = item["task"]
            if task.route == Action.COLLABORATIVE:
                item["cloud_stage_start_slot"] = self.current_time
                task.cloud_stage_start_time = self.current_time
                task.execution_stage = "CLOUD_REVIEW_RUNNING"
            else:
                task.execution_stage = "CLOUD_RUNNING"
        else:
            raise ValueError("start_next_task只接收EDGE或CLOUD")

        task.status = TaskStatus.RUNNING
        if task.start_time is None:
            task.start_time = self.current_time

    def _normal_finished_result(
        self,
        task: Task,
        route: Action,
        *,
        timeout: bool,
        reason: str,
        latency_s: float,
        timeout_stage: str | None = None,
    ) -> dict:
        task.status = TaskStatus.TIMEOUT if timeout else TaskStatus.COMPLETED
        task.finish_time = self.current_time
        task.execution_stage = "TIMEOUT" if timeout else "FINISHED"
        waiting_time_s = (
            float(latency_s)
            if task.start_time is None
            else max(0.0, (task.start_time - task.arrival_time) * self.slot_duration)
        )
        expected_accuracy = (
            float(task.confidence)
            if route == Action.EDGE
            else float(task.estimated_cloud_accuracy)
        )
        upload_mb = float(task.size_mb) if route == Action.CLOUD else 0.0
        result = {
            "task_id": task.task_id,
            "external_task_id": task.external_task_id,
            "task_source": task.source,
            "node_id": task.node_id,
            "task_type": task.task_type,
            "priority": task.priority,
            "device_id": task.device_id,
            "device_type": task.device_type,
            "route": route.name,
            "arrival_time": task.arrival_time,
            "start_time": task.start_time,
            "finish_time": self.current_time,
            "waiting_time_s": waiting_time_s,
            "latency_s": float(latency_s),
            "deadline_s": task.deadline_s,
            "timeout": bool(timeout),
            "timeout_stage": timeout_stage,
            "status": task.status.value,
            "finish_reason": reason,
            "confidence": task.confidence,
            "expected_accuracy": expected_accuracy,
            "size_mb": task.size_mb,
            "upload_mb": upload_mb,
            "complexity": task.complexity,
            "queue_policy": self.queue_policy,
        }
        self._metrics.add_completed_task(result)
        return result

    def _collaborative_components(
        self,
        task: Task,
        item: dict,
        *,
        latency_s: float,
        timeout_stage: str | None,
        completed: bool,
    ) -> dict[str, float]:
        slot = self.slot_duration
        edge_start = item.get("edge_start_slot", task.edge_stage_start_time)
        edge_wait_s = (
            max(0.0, (int(edge_start) - task.arrival_time) * slot)
            if edge_start is not None
            else 0.0
        )
        edge_stage_s = 0.0
        if item.get("edge_finish_slot") is not None or task.edge_stage_done:
            edge_stage_s = int(item.get("stage1_slots", 0)) * slot
        elif timeout_stage == "EDGE_RUNNING":
            edge_stage_s = max(0.0, latency_s - edge_wait_s)
        elif timeout_stage == "EDGE_WAITING":
            edge_wait_s = latency_s

        transfer_s = 0.0
        transfer_start = item.get("transmission_start_slot", task.transmission_start_time)
        transfer_ready = item.get("transmission_ready_slot", task.transmission_ready_time)
        if transfer_start is not None:
            if transfer_ready is not None and (
                self.current_time >= int(transfer_ready)
                or item.get("cloud_stage_start_slot") is not None
                or completed
            ):
                transfer_s = int(item.get("transfer_slots", 0)) * slot
            elif timeout_stage == "TRANSMITTING":
                transfer_s = max(0.0, latency_s - edge_wait_s - edge_stage_s)

        cloud_wait_s = 0.0
        cloud_start = item.get("cloud_stage_start_slot", task.cloud_stage_start_time)
        if transfer_ready is not None:
            if cloud_start is not None:
                cloud_wait_s = max(0.0, (int(cloud_start) - int(transfer_ready)) * slot)
            elif timeout_stage == "CLOUD_WAITING":
                cloud_wait_s = max(
                    0.0,
                    latency_s - edge_wait_s - edge_stage_s - transfer_s,
                )

        cloud_stage_s = 0.0
        if cloud_start is not None:
            if completed:
                cloud_stage_s = int(item.get("stage2_slots", 0)) * slot
            elif timeout_stage == "CLOUD_RUNNING":
                cloud_stage_s = max(
                    0.0,
                    latency_s
                    - edge_wait_s
                    - edge_stage_s
                    - transfer_s
                    - cloud_wait_s,
                )

        values = {
            "edge_wait_s": edge_wait_s,
            "edge_stage_latency": edge_stage_s,
            "transmission_latency": transfer_s,
            "cloud_wait_s": cloud_wait_s,
            "cloud_stage_latency": cloud_stage_s,
        }
        component_sum = sum(values.values())
        delta = float(latency_s) - component_sum
        if abs(delta) > 1e-9:
            if timeout_stage == "EDGE_WAITING":
                values["edge_wait_s"] += delta
            elif timeout_stage == "EDGE_RUNNING":
                values["edge_stage_latency"] += delta
            elif timeout_stage == "TRANSMITTING":
                values["transmission_latency"] += delta
            elif timeout_stage == "CLOUD_WAITING":
                values["cloud_wait_s"] += delta
            else:
                values["cloud_stage_latency"] += delta
        return values

    def _collaborative_finished_result(
        self,
        task: Task,
        item: dict,
        *,
        timeout: bool,
        reason: str,
        latency_s: float,
        timeout_stage: str | None = None,
    ) -> dict:
        task.status = TaskStatus.TIMEOUT if timeout else TaskStatus.COMPLETED
        task.finish_time = self.current_time
        task.execution_stage = "TIMEOUT" if timeout else "FINISHED"
        components = self._collaborative_components(
            task,
            item,
            latency_s=float(latency_s),
            timeout_stage=timeout_stage,
            completed=not timeout,
        )
        planned_upload_mb = float(
            item.get("upload_mb", task.size_mb * task.collaborative_payload_ratio)
        )
        result = {
            "task_id": task.task_id,
            "external_task_id": task.external_task_id,
            "task_source": task.source,
            "node_id": task.node_id,
            "task_type": task.task_type,
            "priority": task.priority,
            "device_id": task.device_id,
            "device_type": task.device_type,
            "route": Action.COLLABORATIVE.name,
            "arrival_time": task.arrival_time,
            "start_time": task.start_time,
            "finish_time": self.current_time,
            "edge_stage_start_time": task.edge_stage_start_time,
            "edge_stage_finish_time": task.edge_stage_finish_time,
            "transmission_start_time": task.transmission_start_time,
            "transmission_ready_time": task.transmission_ready_time,
            "cloud_stage_start_time": task.cloud_stage_start_time,
            **components,
            "upload_time_estimated_s": float(item.get("upload_time_estimated_s", 0.0)),
            "network_rtt_estimated_s": float(item.get("network_rtt_estimated_s", 0.0)),
            "waiting_time_s": components["edge_wait_s"] + components["cloud_wait_s"],
            "latency_s": float(latency_s),
            "component_sum_s": float(sum(components.values())),
            "deadline_s": task.deadline_s,
            "timeout": bool(timeout),
            "timeout_stage": timeout_stage,
            "status": task.status.value,
            "finish_reason": reason,
            "confidence": task.confidence,
            "expected_accuracy": task.collaborative_accuracy,
            "size_mb": task.size_mb,
            "upload_mb": planned_upload_mb,
            "complexity": task.complexity,
            "queue_policy": self.queue_policy,
            "edge_draft_result": task.edge_draft_result,
            "collaborative_request": task.collaborative_request,
        }
        self._metrics.add_completed_task(result)
        return result

    def _timeout_item(
        self,
        item: dict,
        *,
        physical_route: Action | None,
        timeout_stage: str,
        reason: str,
        latency_s: float,
    ) -> dict:
        task: Task = item["task"]
        if task.route == Action.COLLABORATIVE:
            return self._collaborative_finished_result(
                task,
                item,
                timeout=True,
                reason=reason,
                latency_s=latency_s,
                timeout_stage=timeout_stage,
            )
        if physical_route not in (Action.EDGE, Action.CLOUD):
            raise ValueError("普通任务超时必须给出EDGE或CLOUD物理路径")
        return self._normal_finished_result(
            task,
            physical_route,
            timeout=True,
            reason=reason,
            latency_s=latency_s,
            timeout_stage=timeout_stage,
        )

    def remove_expired_waiting_tasks(
        self,
        queue: deque,
        physical_route: Action,
        timeout_stage: str,
    ) -> list[dict]:
        remaining: deque[dict] = deque()
        timed_out: list[dict] = []
        elapsed_s = lambda task: (self.current_time - task.arrival_time) * self.slot_duration
        while queue:
            item = queue.popleft()
            task: Task = item["task"]
            waiting_s = elapsed_s(task)
            if waiting_s >= task.deadline_s:
                timed_out.append(
                    self._timeout_item(
                        item,
                        physical_route=physical_route,
                        timeout_stage=timeout_stage,
                        reason=f"任务在{timeout_stage}阶段超过截止时间",
                        latency_s=waiting_s,
                    )
                )
            else:
                remaining.append(item)
        queue.extend(remaining)
        return timed_out

    def remove_expired_transmissions(self) -> list[dict]:
        remaining: deque[dict] = deque()
        timed_out: list[dict] = []
        while self.transmission_queue:
            item = self.transmission_queue.popleft()
            task: Task = item["task"]
            elapsed_s = (self.current_time - task.arrival_time) * self.slot_duration
            if elapsed_s >= task.deadline_s:
                timed_out.append(
                    self._timeout_item(
                        item,
                        physical_route=None,
                        timeout_stage="TRANSMITTING",
                        reason="协同任务在传输阶段超过截止时间",
                        latency_s=elapsed_s,
                    )
                )
            else:
                remaining.append(item)
        self.transmission_queue.extend(remaining)
        return timed_out

    def remove_expired_active_task(self, physical_route: Action) -> dict | None:
        active = self.edge_active if physical_route == Action.EDGE else self.cloud_active
        if active is None:
            return None
        task: Task = active["task"]
        elapsed_s = (self.current_time - task.arrival_time) * self.slot_duration
        if elapsed_s < task.deadline_s:
            return None
        if task.route == Action.COLLABORATIVE:
            timeout_stage = (
                "EDGE_RUNNING" if physical_route == Action.EDGE else "CLOUD_RUNNING"
            )
        else:
            timeout_stage = (
                "EDGE_RUNNING" if physical_route == Action.EDGE else "CLOUD_RUNNING"
            )
        result = self._timeout_item(
            active,
            physical_route=physical_route,
            timeout_stage=timeout_stage,
            reason=f"任务在{timeout_stage}阶段超过截止时间",
            latency_s=elapsed_s,
        )
        if physical_route == Action.EDGE:
            self.edge_active = None
        else:
            self.cloud_active = None
        return result

    def process_active_task(self, physical_route: Action) -> dict | None:
        active = self.edge_active if physical_route == Action.EDGE else self.cloud_active
        if active is None:
            return None
        active["remaining_slots"] -= 1
        if active["remaining_slots"] > 0:
            return None

        task: Task = active["task"]
        if physical_route == Action.EDGE and task.route == Action.COLLABORATIVE:
            self.edge_active = None
            self._start_collaborative_transmission(active)
            return None

        latency_s = (self.current_time - task.arrival_time + 1) * self.slot_duration
        timeout = latency_s > task.deadline_s

        if physical_route == Action.CLOUD and task.route == Action.COLLABORATIVE:
            self.cloud_active = None
            return self._collaborative_finished_result(
                task,
                active,
                timeout=timeout,
                reason=(
                    "协同任务完成时超过截止时间"
                    if timeout
                    else "协同任务在截止时间内完成"
                ),
                latency_s=latency_s,
                timeout_stage="CLOUD_RUNNING" if timeout else None,
            )

        result = self._normal_finished_result(
            task,
            physical_route,
            timeout=timeout,
            reason=(
                "任务完成时超过截止时间" if timeout else "任务在截止时间内完成"
            ),
            latency_s=latency_s,
            timeout_stage=(
                "EDGE_RUNNING"
                if timeout and physical_route == Action.EDGE
                else "CLOUD_RUNNING"
                if timeout
                else None
            ),
        )
        if physical_route == Action.EDGE:
            self.edge_active = None
        else:
            self.cloud_active = None
        return result

    def _sync_reward_config(self) -> None:
        config = self._reward_calculator.config
        config.latency_weight = self.latency_weight
        config.queue_penalty_weight = self.queue_penalty_weight
        config.timeout_penalty = self.timeout_penalty
        config.accuracy_penalty_weight = self.accuracy_penalty_weight
        config.communication_penalty_weight = self.communication_penalty_weight
        config.edge_compute_penalty_weight = self.edge_compute_penalty_weight
        config.cloud_compute_penalty_weight = self.cloud_compute_penalty_weight
        config.cloud_accuracy = self.cloud_accuracy
        config.reward_mode = self.reward_mode

    def calculate_reward(
        self,
        task: Task | None,
        action: Action,
        details: dict,
        state_before: dict,
        state_after: dict,
        timeout_count: int,
    ) -> dict:
        self._sync_reward_config()
        return self._reward_calculator.calculate(
            task=task,
            action=action,
            details=details,
            state_before=state_before,
            state_after=state_after,
            timeout_count=timeout_count,
        )

    def update_system_state(self, *, edge_executed: bool, cloud_executed: bool) -> None:
        self._environment.update(
            edge_busy=self.edge_active is not None or bool(self.edge_queue),
            cloud_busy=self.cloud_active is not None or bool(self.cloud_queue),
            edge_executed=edge_executed,
            cloud_executed=cloud_executed,
            rng=self.rng,
        )

    def prepare_decision(self) -> dict:
        promoted = self._promote_ready_transmissions()
        edge_waiting_timeouts = self.remove_expired_waiting_tasks(
            self.edge_queue,
            Action.EDGE,
            "EDGE_WAITING",
        )
        cloud_waiting_timeouts = self.remove_expired_waiting_tasks(
            self.cloud_queue,
            Action.CLOUD,
            "CLOUD_WAITING",
        )
        transmission_timeouts = self.remove_expired_transmissions()
        edge_active_timeout = self.remove_expired_active_task(Action.EDGE)
        cloud_active_timeout = self.remove_expired_active_task(Action.CLOUD)
        return {
            "promoted_transmissions": promoted,
            "edge_waiting_timeouts": edge_waiting_timeouts,
            "cloud_waiting_timeouts": cloud_waiting_timeouts,
            "transmission_timeouts": transmission_timeouts,
            "edge_active_timeout": edge_active_timeout,
            "cloud_active_timeout": cloud_active_timeout,
            "state_before": self.get_state(),
            "task": self.generate_task(),
        }

    @staticmethod
    def _empty_details() -> dict[str, float]:
        keys = [
            "edge_delay",
            "cloud_delay",
            "edge_waiting_time",
            "cloud_waiting_time",
            "edge_service_time",
            "cloud_service_time",
            "cloud_inference_time",
            "transmission_time",
            "network_rtt_s",
            "collaborative_delay",
            "collaborative_edge_wait_s",
            "collaborative_edge_stage_s",
            "collaborative_upload_mb",
            "collaborative_upload_time_s",
            "collaborative_network_rtt_s",
            "collaborative_transfer_total_s",
            "collaborative_cloud_wait_s",
            "collaborative_cloud_stage_s",
            "collaborative_expected_accuracy",
        ]
        return {key: float("nan") for key in keys}

    def apply_decision(
        self,
        context: dict,
        action: Action,
        reason: str,
        details: dict | None = None,
    ) -> dict:
        task: Task | None = context["task"]
        state_before = context["state_before"]
        if task is None:
            action = Action.IDLE
            reason = "当前时隙没有新任务"
            details = self._empty_details()
        else:
            if action not in (Action.EDGE, Action.CLOUD, Action.COLLABORATIVE):
                raise ValueError("存在新任务时只能选择EDGE/CLOUD/COLLABORATIVE")
            if details is None:
                details = estimate_delays_for_task(task, state_before)
            if action == Action.COLLABORATIVE:
                self._enqueue_collaborative_stage1(task, details)
            else:
                self.enqueue_task(task, action, details)

        self.start_next_task(Action.EDGE)
        self.start_next_task(Action.CLOUD)
        edge_executed = self.edge_active is not None
        cloud_executed = self.cloud_active is not None
        edge_completed = self.process_active_task(Action.EDGE)
        cloud_completed = self.process_active_task(Action.CLOUD)
        state_after = self.get_state()

        timeout_count = (
            len(context["edge_waiting_timeouts"])
            + len(context["cloud_waiting_timeouts"])
            + len(context["transmission_timeouts"])
            + int(context["edge_active_timeout"] is not None)
            + int(context["cloud_active_timeout"] is not None)
            + int(edge_completed is not None and edge_completed["timeout"])
            + int(cloud_completed is not None and cloud_completed["timeout"])
        )
        reward_details = self.calculate_reward(
            task,
            action,
            details,
            state_before,
            state_after,
            timeout_count,
        )
        self.cumulative_reward += reward_details["reward"]

        record = {
            "time": self.current_time,
            "time_s": round(self.current_time * self.slot_duration, 4),
            "slot_duration_s": self.slot_duration,
            "scheduler_name": self.scheduler_name,
            "queue_policy": self.queue_policy,
            "network_mode": self.network_mode,
            "resource_mode": self.resource_mode,
            "reward_mode": self.reward_mode,
            "task_id": None if task is None else task.task_id,
            "external_task_id": None if task is None else task.external_task_id,
            "task_source": self.task_source if task is None else task.source,
            "node_id": None if task is None else task.node_id,
            "task_type": None if task is None else task.task_type,
            "priority": None if task is None else task.priority,
            "complexity": np.nan if task is None else round(task.complexity, 3),
            "task_size_mb": np.nan if task is None else round(task.size_mb, 6),
            "deadline_s": np.nan if task is None else round(task.deadline_s, 4),
            "confidence": np.nan if task is None else round(task.confidence, 3),
            "cloud_accuracy_prior": (
                np.nan if task is None else round(task.estimated_cloud_accuracy, 3)
            ),
            "collaborative_accuracy_prior": (
                np.nan if task is None else round(task.collaborative_accuracy, 3)
            ),
            "action": int(action),
            "action_name": action.name,
            "reason": reason,
            "edge_estimated_delay": (
                np.nan if task is None else round(details["edge_delay"], 4)
            ),
            "cloud_estimated_delay": (
                np.nan if task is None else round(details["cloud_delay"], 4)
            ),
            "collaborative_estimated_delay": (
                np.nan if task is None else round(details["collaborative_delay"], 4)
            ),
            "edge_service_time": (
                np.nan if task is None else round(details["edge_service_time"], 4)
            ),
            "cloud_service_time": (
                np.nan if task is None else round(details["cloud_service_time"], 4)
            ),
            "transmission_time": (
                np.nan if task is None else round(details["transmission_time"], 4)
            ),
            "network_rtt_s": (
                np.nan if task is None else round(details["network_rtt_s"], 4)
            ),
            "collaborative_edge_stage_s": (
                np.nan
                if task is None
                else round(details["collaborative_edge_stage_s"], 4)
            ),
            "collaborative_upload_time_s": (
                np.nan
                if task is None
                else round(details["collaborative_upload_time_s"], 4)
            ),
            "collaborative_network_rtt_s": (
                np.nan
                if task is None
                else round(details["collaborative_network_rtt_s"], 4)
            ),
            "collaborative_cloud_stage_s": (
                np.nan
                if task is None
                else round(details["collaborative_cloud_stage_s"], 4)
            ),
            "selected_upload_mb": round(reward_details["selected_upload_mb"], 6),
            "edge_queue": state_after["edge_queue"],
            "cloud_queue": state_after["cloud_queue"],
            "cloud_ready_queue": state_after["cloud_ready_queue"],
            "transmission_queue": state_after["transmission_queue"],
            "edge_workload_s": round(state_after["edge_workload_s"], 4),
            "cloud_workload_s": round(state_after["cloud_workload_s"], 4),
            "edge_resource": round(self.edge_resource, 3),
            "cloud_resource": round(self.cloud_resource, 3),
            "network_quality": round(self.network_quality, 3),
            "network_state": self.network_state,
            "bandwidth_mbps": round(self.bandwidth_mbps, 2),
            "promoted_transmissions": len(context["promoted_transmissions"]),
            "edge_completed_task": None if edge_completed is None else edge_completed["task_id"],
            "cloud_completed_task": None if cloud_completed is None else cloud_completed["task_id"],
            "timeout_count": timeout_count,
            "selected_delay_s": round(reward_details["selected_delay_s"], 4),
            "deadline_ratio": round(reward_details["deadline_ratio"], 4),
            "expected_accuracy": round(reward_details["expected_accuracy"], 3),
            "delay_penalty": round(reward_details["delay_penalty"], 4),
            "queue_penalty": round(reward_details["queue_penalty"], 4),
            "timeout_penalty": round(reward_details["timeout_penalty"], 4),
            "accuracy_penalty": round(reward_details["accuracy_penalty"], 4),
            "communication_penalty": round(
                reward_details["communication_penalty"], 4
            ),
            "resource_penalty": round(reward_details["resource_penalty"], 4),
            "lyapunov_drift": round(reward_details["lyapunov_drift"], 4),
            "reward": round(reward_details["reward"], 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
        }
        self._metrics.add_record(record)
        self.update_system_state(
            edge_executed=edge_executed,
            cloud_executed=cloud_executed,
        )
        self.current_time += 1
        return record

    def step(self) -> dict:
        context = self.prepare_decision()
        task = context["task"]
        if task is None:
            return self.apply_decision(context, Action.IDLE, "当前时隙没有新任务")
        action, reason, details = self.scheduler.select_action(
            task,
            context["state_before"],
        )
        return self.apply_decision(context, action, reason, details)

    def run(self, steps: int) -> pd.DataFrame:
        if steps <= 0:
            raise ValueError("steps必须大于0")
        for _ in range(steps):
            self.step()
        return self.get_records()

    def run_until_complete(self, max_steps: int = 20_000) -> pd.DataFrame:
        if self.task_source != "edge_input":
            raise RuntimeError("run_until_complete仅用于edge_input")
        for _ in range(max_steps):
            if self.external_tasks_finished():
                break
            self.step()
        else:
            raise RuntimeError("达到max_steps后仍有外部任务未完成")
        return self.get_records()

    def external_tasks_finished(self) -> bool:
        return (
            self.task_source == "edge_input"
            and not self.external_tasks
            and not self.edge_queue
            and not self.cloud_queue
            and not self.transmission_queue
            and self.edge_active is None
            and self.cloud_active is None
        )

    def get_records(self) -> pd.DataFrame:
        return self._metrics.get_records_df()

    def get_completed_tasks(self) -> pd.DataFrame:
        return self._metrics.get_completed_df()

    def get_task_type_metrics(self) -> pd.DataFrame:
        return self._metrics.summarize_by_task_type()

    def get_metrics(self) -> dict:
        in_system = (
            len(self.edge_queue)
            + len(self.cloud_queue)
            + len(self.transmission_queue)
            + int(self.edge_active is not None)
            + int(self.cloud_active is not None)
        )
        return self._metrics.summarize(
            generated_tasks=self.generated_task_count,
            in_system_tasks=in_system,
        )

    def reset(self) -> None:
        self.__init__(
            seed=self.seed,
            arrival_probability=self.arrival_probability,
            slot_duration=self.slot_duration,
            scheduler=self.scheduler,
            latency_weight=self.latency_weight,
            queue_penalty_weight=self.queue_penalty_weight,
            timeout_penalty=self.timeout_penalty,
            accuracy_penalty_weight=self.accuracy_penalty_weight,
            communication_penalty_weight=self.communication_penalty_weight,
            edge_compute_penalty_weight=self.edge_compute_penalty_weight,
            cloud_compute_penalty_weight=self.cloud_compute_penalty_weight,
            cloud_accuracy=self.cloud_accuracy,
            queue_policy=self.queue_policy,
            network_mode=self.network_mode,
            resource_mode=self.resource_mode,
            reward_mode=self.reward_mode,
            task_source=self.task_source,
            edge_input_path=self.edge_input_path,
            edge_input_strict=self.edge_input_strict,
            edge_input_interval_s=self.edge_input_interval_s,
        )


if __name__ == "__main__":
    simulator = CloudEdgeSimulator()
    simulator.run(500)
    print(pd.Series(simulator.get_metrics()).to_string())
