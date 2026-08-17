"""Phase4.5 real-ready cloud-edge simulator.

Key changes from Phase4 v1.8.5:
- continuous simulation time in seconds (no remaining_slots service engine)
- PPO decisions only at task-arrival decision events
- non-preemptive EDF waiting queues
- task lifecycle aligned with Integration v0.3
- pluggable arrival/service/resource/network dynamics
- StepOutcome -> RewardCalculator separation

All numeric task profiles, resource/network parameters and reward weights remain
unchanged by default. Phase4.5 is *real-ready*, not real-calibrated.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

from dynamics import (
    LegacyNetworkDynamics,
    LegacyResourceDynamics,
    LegacyTaskArrivalModel,
    NetworkDynamics,
    ProfileServiceTimeModel,
    ResourceDynamics,
    ServiceTimeModel,
    TaskArrivalModel,
)
from edge_input_adapter import EdgeInputLoadReport, load_edge_inputs
from edf_queue import EDFTaskQueue, QueueItem
from lifecycle import (
    CLOUD_REVIEW_RUNNING,
    CLOUD_REVIEW_WAITING,
    CLOUD_RUNNING,
    CLOUD_WAITING,
    EDGE_RESULT_READY,
    EDGE_RUNNING,
    EDGE_WAITING,
    TRANSMITTING,
    mark_finished,
    mark_edge_result_ready,
    mark_running,
    mark_timeout,
    mark_transmitting,
    mark_waiting,
)
from metrics import MetricsTracker
from models import Action, ROUTE_TO_PPO_ACTION, Task, TaskStatus
from outcome import StepOutcome
from post_edge_policy import DEFAULT_RULES_PATH, PostEdgeDecision, PostEdgePolicy
from reward import RewardCalculator, RewardConfig
from scheduler import RuleScheduler
from state_provider import SimStateProvider
from system_state import SystemState
from task_profiles import sample_profile_values, sample_task_type


VALID_QUEUE_POLICIES = {"fifo", "edf"}
VALID_TASK_SOURCES = {"legacy_simulation", "typed_simulation", "edge_input"}
EPS = 1e-10


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
        *,
        arrival_model: TaskArrivalModel | None = None,
        service_time_model: ServiceTimeModel | None = None,
        resource_dynamics: ResourceDynamics | None = None,
        network_dynamics: NetworkDynamics | None = None,
        post_edge_policy: PostEdgePolicy | None = None,
        post_edge_rules_path: str | Path = DEFAULT_RULES_PATH,
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
        # Retained as a legacy arrival-probability reference and reporting unit;
        # service/lifecycle progression no longer advances by this slot.
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

        self._external_task_template: list[Task] = []
        self.edge_input_load_report: EdgeInputLoadReport | None = None
        if self.task_source == "edge_input":
            if not self.edge_input_path:
                raise ValueError("edge_input模式必须提供edge_input_path")
            tasks, report = load_edge_inputs(self.edge_input_path, strict=self.edge_input_strict)
            self._external_task_template = tasks
            self.edge_input_load_report = report

        external_offsets = [
            float(t.source_arrival_offset_s) if t.source_arrival_offset_s is not None else -1.0
            for t in self._external_task_template
        ]
        self.arrival_model = arrival_model or LegacyTaskArrivalModel(
            arrival_probability=self.arrival_probability,
            slot_duration=self.slot_duration,
            task_source=self.task_source,
            external_offsets_s=external_offsets,
            edge_input_interval_s=self.edge_input_interval_s,
        )
        self.service_time_model = service_time_model or ProfileServiceTimeModel()
        self.resource_dynamics = resource_dynamics or LegacyResourceDynamics(self.resource_mode)
        self.network_dynamics = network_dynamics or LegacyNetworkDynamics(self.network_mode)
        self.post_edge_policy = post_edge_policy or PostEdgePolicy.from_json(post_edge_rules_path)

        self._reward_calculator = RewardCalculator(RewardConfig(
            latency_weight=self.latency_weight,
            queue_penalty_weight=self.queue_penalty_weight,
            timeout_penalty=self.timeout_penalty,
            accuracy_penalty_weight=self.accuracy_penalty_weight,
            communication_penalty_weight=self.communication_penalty_weight,
            edge_compute_penalty_weight=self.edge_compute_penalty_weight,
            cloud_compute_penalty_weight=self.cloud_compute_penalty_weight,
            cloud_accuracy=self.cloud_accuracy,
            reward_mode=self.reward_mode,
        ))
        self._metrics = MetricsTracker()
        self.state_provider = SimStateProvider(self)
        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.current_time_s = 0.0
        self.next_task_id = 1
        self.generated_task_count = 0
        self._external_index = 0
        self.external_tasks: deque[Task] = deque(self._clone_external_tasks())

        self.edge_queue = EDFTaskQueue(self.queue_policy)
        self.cloud_queue = EDFTaskQueue(self.queue_policy)
        self.transmission_queue: list[QueueItem] = []
        self.edge_active: QueueItem | None = None
        self.cloud_active: QueueItem | None = None
        self.cumulative_reward = 0.0
        self._pending_decision_task: Task | None = None
        self._interval_timeout_ids: list[int] = []
        self._interval_completed_ids: list[int] = []

        self.resource_dynamics.reset()
        self.network_dynamics.reset()
        self.arrival_model.reset(rng=self.rng, now_s=0.0)

    def _clone_external_tasks(self) -> list[Task]:
        # edge_input tasks are mutable lifecycle objects; reconstruct from fields.
        import copy
        return [copy.deepcopy(task) for task in self._external_task_template]

    # ------------------------- compatibility properties -------------------------
    @property
    def current_time(self) -> float:
        """Phase4.5 compatibility alias: current time is now seconds."""
        return self.current_time_s

    @property
    def edge_resource(self) -> float:
        return float(self.resource_dynamics.edge_resource)

    @edge_resource.setter
    def edge_resource(self, value: float) -> None:
        self.resource_dynamics.edge_resource = float(value)

    @property
    def cloud_resource(self) -> float:
        return float(self.resource_dynamics.cloud_resource)

    @cloud_resource.setter
    def cloud_resource(self, value: float) -> None:
        self.resource_dynamics.cloud_resource = float(value)

    @property
    def network_quality(self) -> float:
        return float(self.network_dynamics.network_quality)

    @network_quality.setter
    def network_quality(self, value: float) -> None:
        self.network_dynamics.network_quality = float(value)
        self.network_dynamics.network_state = "GOOD" if value >= 0.5 else "BAD"

    @property
    def network_state(self) -> str:
        return str(self.network_dynamics.network_state)

    @property
    def bandwidth_mbps(self) -> float:
        return float(self.network_dynamics.bandwidth_mbps)

    @property
    def records(self) -> list[dict]:
        return self._metrics.records

    @property
    def completed_tasks(self) -> list[dict]:
        return self._metrics.completed_tasks

    @property
    def scheduler_name(self) -> str:
        return getattr(self.scheduler, "name", self.scheduler.__class__.__name__)

    # ------------------------------- task source -------------------------------
    def _generate_legacy_task(self) -> Task:
        task = Task(
            task_id=self.next_task_id,
            arrival_time=self.current_time_s,
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
            arrival_time=self.current_time_s,
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

    def generate_task(self) -> Task | None:
        """Compatibility helper: consume a task only if an arrival is due now.

        Phase4.5 normally exposes arrivals through advance_to_next_decision().
        """
        return self._consume_arrival_task()

    def _consume_arrival_task(self) -> Task | None:
        due = self.arrival_model.next_arrival_time_s()
        if due is None or due > self.current_time_s + EPS:
            return None
        if self.task_source == "edge_input":
            if not self.external_tasks:
                return None
            task = self.external_tasks.popleft()
            task.arrival_time = self.current_time_s
        else:
            task = self._generate_legacy_task() if self.task_source == "legacy_simulation" else self._generate_typed_task()
            task.arrival_time = self.current_time_s
        self.arrival_model.consume_arrival(rng=self.rng, now_s=self.current_time_s)
        self.generated_task_count += 1
        return task

    # ---------------------------------- state ----------------------------------
    @staticmethod
    def _active_remaining(item: QueueItem | None, now_s: float) -> float:
        if item is None:
            return 0.0
        finish = float(item.metadata.get("finish_time_s", now_s))
        return max(0.0, finish - now_s)

    def get_system_state(self) -> SystemState:
        edge_workload = self.edge_queue.workload_s() + self._active_remaining(self.edge_active, self.current_time_s)
        cloud_ready_workload = self.cloud_queue.workload_s() + self._active_remaining(self.cloud_active, self.current_time_s)
        reserved_cloud_workload = float(sum(
            max(0.0, float(item.metadata.get("cloud_stage_s", 0.0)))
            for item in self.transmission_queue
        ))
        return SystemState(
            current_time_s=self.current_time_s,
            edge_queue=len(self.edge_queue),
            cloud_queue=len(self.cloud_queue) + len(self.transmission_queue),
            edge_workload_s=edge_workload,
            cloud_workload_s=cloud_ready_workload + reserved_cloud_workload,
            edge_resource=self.edge_resource,
            cloud_resource=self.cloud_resource,
            network_quality=self.network_quality,
            bandwidth_mbps=self.bandwidth_mbps,
            edge_busy=self.edge_active is not None,
            cloud_busy=self.cloud_active is not None,
            cloud_ready_queue=len(self.cloud_queue),
            transmission_queue=len(self.transmission_queue),
            cloud_ready_workload_s=cloud_ready_workload,
            reserved_cloud_workload_s=reserved_cloud_workload,
            edge_active_task=None if self.edge_active is None else self.edge_active.task.task_id,
            cloud_active_task=None if self.cloud_active is None else self.cloud_active.task.task_id,
            network_state=self.network_state,
            queue_policy=self.queue_policy,
            network_mode=self.network_mode,
            resource_mode=self.resource_mode,
            reward_mode=self.reward_mode,
            task_source=self.task_source,
            external_tasks_remaining=len(self.external_tasks),
            edge_inputs_loaded=0 if self.edge_input_load_report is None else self.edge_input_load_report.loaded_count,
        )

    def get_state(self) -> dict:
        return self.get_system_state().to_dict()

    def get_node_workload(self, queue, active) -> float:
        if isinstance(queue, EDFTaskQueue):
            return queue.workload_s() + self._active_remaining(active, self.current_time_s)
        # Compatibility for old callers with iterables of QueueItem-like objects.
        total = 0.0
        for item in queue:
            if hasattr(item, "remaining_service_time_s"):
                total += float(item.remaining_service_time_s)
            elif isinstance(item, dict):
                total += float(item.get("remaining_service_time_s", 0.0))
        return float(total)

    # --------------------------- enqueue / lifecycle ---------------------------
    def enqueue_task(self, task: Task, action: Action, details: dict) -> None:
        if action == Action.LOCAL_FIRST:
            service_time = float(details["edge_service_time"])
            queue = self.edge_queue
            stage = EDGE_WAITING
        elif action == Action.DIRECT_CLOUD:
            service_time = float(details["cloud_service_time"])
            queue = self.cloud_queue
            stage = CLOUD_WAITING
        else:
            raise ValueError("enqueue_task只接收LOCAL_FIRST或DIRECT_CLOUD")
        task.pre_route_action = action
        mark_waiting(task, action, stage)
        queue.push(QueueItem(
            task=task,
            route=action,
            stage=stage,
            service_time_s=service_time,
            remaining_service_time_s=service_time,
            decision_details=dict(details),
        ))

    def _start_next_edge(self) -> None:
        if self.edge_active is not None or not self.edge_queue:
            return
        item = self.edge_queue.pop()
        self.edge_active = item
        task = item.task
        if item.route == Action.LOCAL_FIRST:
            task.edge_stage_start_time = self.current_time_s
        mark_running(task, EDGE_RUNNING, self.current_time_s)
        item.remaining_service_time_s = item.service_time_s
        item.metadata["start_time_s"] = self.current_time_s
        item.metadata["finish_time_s"] = self.current_time_s + item.service_time_s

    def _start_next_cloud(self) -> None:
        if self.cloud_active is not None or not self.cloud_queue:
            return
        item = self.cloud_queue.pop()
        self.cloud_active = item
        task = item.task
        stage = CLOUD_REVIEW_RUNNING if item.route == Action.CLOUD_REVIEW else CLOUD_RUNNING
        if item.route == Action.CLOUD_REVIEW:
            task.cloud_stage_start_time = self.current_time_s
        mark_running(task, stage, self.current_time_s)
        item.remaining_service_time_s = item.service_time_s
        item.metadata["start_time_s"] = self.current_time_s
        item.metadata["finish_time_s"] = self.current_time_s + item.service_time_s

    def _start_available_workers(self) -> None:
        self._start_next_edge()
        self._start_next_cloud()

    @staticmethod
    def _simulated_edge_result(task: Task) -> dict[str, Any]:
        return {
            "category": f"simulated_{task.task_type}_prediction",
            "level": "high" if task.priority == "high" else "medium",
            "confidence": float(task.confidence),
            "output_valid": True,
            "ood": False,
            "summary": "Phase4.5 simulated edge result; replace with measured model output in Integration",
        }

    def _start_post_edge_transmission(
        self,
        edge_item: QueueItem,
        transition_details: dict[str, float],
    ) -> None:
        task = edge_item.task
        task.route = Action.CLOUD_REVIEW
        edge_item.route = Action.CLOUD_REVIEW
        task.final_route = "COLLABORATIVE"
        mark_transmitting(task, self.current_time_s)
        transfer_s = float(transition_details["local_first_review_transfer_total_s"])
        cloud_stage_s = float(transition_details["local_first_review_cloud_stage_s"])
        upload_mb = float(transition_details["local_first_review_upload_mb"])
        ready_s = self.current_time_s + transfer_s
        task.transmission_ready_time = ready_s
        task.collaborative_request = {
            "task_id": task.external_task_id or task.task_id,
            "edge_draft": task.edge_draft_result,
            "compressed_payload_mb": upload_mb,
            "request_type": "cloud_review",
        }
        edge_item.stage = TRANSMITTING
        edge_item.metadata.update({
            "edge_finish_time_s": self.current_time_s,
            "transmission_start_time_s": self.current_time_s,
            "transmission_ready_time_s": ready_s,
            "transfer_s": transfer_s,
            "cloud_stage_s": cloud_stage_s,
            "upload_mb": upload_mb,
            "transition_details": transition_details,
        })
        self.transmission_queue.append(edge_item)

    def _handle_local_first_completion(self, edge_item: QueueItem) -> bool:
        """Apply PostEdgePolicy after the simulated edge result exists.

        Returns True when the task finishes locally and False when it continues
        through transmission and cloud review.
        """
        task = edge_item.task
        mark_edge_result_ready(task, self.current_time_s)
        task.edge_draft_result = self._simulated_edge_result(task)
        transition_details = self.service_time_model.estimate(task, self.get_system_state())
        policy_result = self.post_edge_policy.evaluate(
            task=task,
            edge_result=task.edge_draft_result,
            now_s=self.current_time_s,
            cloud_review_estimated_s=float(
                transition_details["local_first_review_remaining_s"]
            ),
        )
        task.post_edge_decision = policy_result.decision.value
        task.post_edge_reason = policy_result.reason
        task.post_edge_rule_version = policy_result.rule_version
        edge_item.metadata["post_edge_policy"] = policy_result.to_dict()
        edge_item.metadata["transition_details"] = transition_details

        if policy_result.decision == PostEdgeDecision.CLOUD_REVIEW:
            self._start_post_edge_transmission(edge_item, transition_details)
            return False

        task.final_route = "EDGE"
        self._add_result(task, timeout=False, reason=policy_result.reason)
        return True

    def _promote_ready_transmissions(self) -> list[int]:
        promoted: list[int] = []
        remaining: list[QueueItem] = []
        for item in self.transmission_queue:
            ready = float(item.metadata["transmission_ready_time_s"])
            if ready <= self.current_time_s + EPS:
                task = item.task
                task.uploaded_to_cloud = True
                mark_waiting(task, Action.CLOUD_REVIEW, CLOUD_REVIEW_WAITING)
                service = float(item.metadata["cloud_stage_s"])
                item.stage = "CLOUD_REVIEW"
                item.service_time_s = service
                item.remaining_service_time_s = service
                self.cloud_queue.push(item)
                promoted.append(task.task_id)
            else:
                remaining.append(item)
        self.transmission_queue = remaining
        return promoted

    # ------------------------------ task results ------------------------------
    def _add_result(self, task: Task, *, timeout: bool, reason: str, timeout_stage: str | None = None) -> dict:
        latency = max(0.0, self.current_time_s - float(task.arrival_time))
        if timeout:
            mark_timeout(task, self.current_time_s)
        else:
            mark_finished(task, self.current_time_s)
        waiting = latency if task.start_time is None else max(0.0, float(task.start_time) - float(task.arrival_time))
        route = task.route or Action.LOCAL_FIRST
        if task.final_route is not None:
            final_route = task.final_route
        elif route == Action.DIRECT_CLOUD:
            final_route = "CLOUD"
        elif route == Action.CLOUD_REVIEW:
            final_route = "COLLABORATIVE"
        else:
            final_route = "EDGE"
        task.final_route = final_route
        expected_accuracy = (
            float(task.collaborative_accuracy) if final_route == "COLLABORATIVE"
            else float(task.confidence) if final_route == "EDGE"
            else float(task.estimated_cloud_accuracy)
        )
        upload_mb = (
            float(task.size_mb) if final_route == "CLOUD"
            else float(task.size_mb * task.collaborative_payload_ratio) if final_route == "COLLABORATIVE"
            else 0.0
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
            "route": final_route,
            "pre_route_action": None if task.pre_route_action is None else task.pre_route_action.name,
            "post_edge_decision": task.post_edge_decision,
            "post_edge_reason": task.post_edge_reason,
            "post_edge_rule_version": task.post_edge_rule_version,
            "arrival_time": task.arrival_time,
            "start_time": task.start_time,
            "finish_time": self.current_time_s,
            "waiting_time_s": waiting,
            "latency_s": latency,
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
            "execution_stage": task.execution_stage,
            "edge_stage_start_time": task.edge_stage_start_time,
            "edge_stage_finish_time": task.edge_stage_finish_time,
            "transmission_start_time": task.transmission_start_time,
            "transmission_ready_time": task.transmission_ready_time,
            "cloud_stage_start_time": task.cloud_stage_start_time,
        }
        if final_route == "COLLABORATIVE":
            now = self.current_time_s
            edge_start = task.edge_stage_start_time
            edge_finish = task.edge_stage_finish_time
            transfer_start = task.transmission_start_time
            transfer_ready = task.transmission_ready_time
            cloud_start = task.cloud_stage_start_time
            edge_wait = max(0.0, (edge_start if edge_start is not None else now) - float(task.arrival_time))
            edge_exec = 0.0 if edge_start is None else max(0.0, (edge_finish if edge_finish is not None else now) - edge_start)
            transfer = 0.0 if transfer_start is None else max(0.0, (transfer_ready if transfer_ready is not None and transfer_ready <= now else now) - transfer_start)
            cloud_wait = 0.0
            if transfer_ready is not None and transfer_ready <= now:
                cloud_wait = max(0.0, (cloud_start if cloud_start is not None else now) - transfer_ready)
            cloud_exec = 0.0 if cloud_start is None else max(0.0, now - cloud_start)
            component_sum = edge_wait + edge_exec + transfer + cloud_wait + cloud_exec
            result.update({
                "edge_waiting_time_s": edge_wait,
                "edge_execution_time_s": edge_exec,
                "transmission_latency": transfer,
                "cloud_waiting_time_s": cloud_wait,
                "cloud_execution_time_s": cloud_exec,
                "component_sum_s": component_sum,
            })
        self._metrics.add_completed_task(result)
        if timeout:
            self._interval_timeout_ids.append(task.task_id)
        else:
            self._interval_completed_ids.append(task.task_id)
        return result

    def _deadline_tasks(self) -> list[tuple[str, QueueItem]]:
        rows: list[tuple[str, QueueItem]] = []
        rows.extend(("EDGE_WAITING", i) for i in self.edge_queue.items())
        rows.extend(("CLOUD_WAITING", i) for i in self.cloud_queue.items())
        rows.extend(("TRANSMITTING", i) for i in self.transmission_queue)
        if self.edge_active is not None:
            rows.append(("EDGE_RUNNING", self.edge_active))
        if self.cloud_active is not None:
            rows.append(("CLOUD_RUNNING", self.cloud_active))
        return rows

    def _expire_due_tasks(self) -> list[int]:
        expired: list[int] = []
        for stage, item in list(self._deadline_tasks()):
            if item.task.absolute_deadline_s > self.current_time_s + EPS:
                continue
            task = item.task
            if item is self.edge_active:
                self.edge_active = None
            elif item is self.cloud_active:
                self.cloud_active = None
            elif item in self.transmission_queue:
                self.transmission_queue.remove(item)
            else:
                try:
                    if stage.startswith("EDGE"):
                        self.edge_queue.remove(item)
                    else:
                        self.cloud_queue.remove(item)
                except ValueError:
                    continue
            self._add_result(task, timeout=True, reason="超过任务deadline", timeout_stage=task.execution_stage or stage)
            expired.append(task.task_id)
        return expired

    def _complete_due_active(self) -> list[int]:
        completed: list[int] = []
        for physical in (Action.LOCAL_FIRST, Action.DIRECT_CLOUD):
            item = self.edge_active if physical == Action.LOCAL_FIRST else self.cloud_active
            if item is None:
                continue
            if float(item.metadata.get("finish_time_s", math.inf)) > self.current_time_s + EPS:
                continue
            if physical == Action.LOCAL_FIRST:
                self.edge_active = None
            else:
                self.cloud_active = None
            task = item.task
            if item.route == Action.LOCAL_FIRST and physical == Action.LOCAL_FIRST:
                if self._handle_local_first_completion(item):
                    completed.append(task.task_id)
            else:
                self._add_result(task, timeout=False, reason="服务阶段完成")
                completed.append(task.task_id)
        return completed

    # ----------------------------- event advancement -----------------------------
    def _next_deadline_time(self) -> float | None:
        times = [item.task.absolute_deadline_s for _, item in self._deadline_tasks()]
        return min(times) if times else None

    def _next_event_time(self, *, include_arrival: bool = True) -> float | None:
        candidates: list[float] = []
        if include_arrival:
            t = self.arrival_model.next_arrival_time_s()
            if t is not None:
                candidates.append(float(t))
        for active in (self.edge_active, self.cloud_active):
            if active is not None:
                candidates.append(float(active.metadata.get("finish_time_s", math.inf)))
        candidates.extend(float(i.metadata["transmission_ready_time_s"]) for i in self.transmission_queue)
        deadline = self._next_deadline_time()
        if deadline is not None:
            candidates.append(float(deadline))
        if not candidates:
            return None
        return max(self.current_time_s, min(candidates))

    def _advance_clock(self, target_time_s: float) -> None:
        target = max(self.current_time_s, float(target_time_s))
        dt = target - self.current_time_s
        if dt <= EPS:
            self.current_time_s = target
            return
        edge_busy = self.edge_active is not None
        cloud_busy = self.cloud_active is not None
        self.resource_dynamics.advance(
            delta_t_s=dt,
            edge_busy=edge_busy,
            cloud_busy=cloud_busy,
            rng=self.rng,
        )
        self.network_dynamics.advance(delta_t_s=dt, rng=self.rng)
        self.current_time_s = target

    def _process_internal_events_now(self) -> dict[str, list[int]]:
        # Deadline first: at exact equality the task has exhausted its deadline.
        timeouts = self._expire_due_tasks()
        completed = self._complete_due_active()
        promoted = self._promote_ready_transmissions()
        self._start_available_workers()
        return {"timeouts": timeouts, "completed": completed, "promoted": promoted}

    def advance_to_next_decision(self) -> tuple[dict | None, float, dict[str, Any]]:
        """Advance physical events until a new task needs a routing decision."""
        interval_state_before = self.get_system_state()
        self._interval_timeout_ids = []
        self._interval_completed_ids = []
        start = self.current_time_s
        event_count = 0

        for _ in range(100_000):
            # A due arrival at the current clock becomes a decision immediately.
            due = self.arrival_model.next_arrival_time_s()
            if due is not None and due <= self.current_time_s + EPS:
                task = self._consume_arrival_task()
                if task is not None:
                    self._pending_decision_task = task
                    state = self.get_system_state()
                    return {
                        "task": task,
                        "state_before": state,
                        "promoted_transmissions": [],
                        "edge_waiting_timeouts": [],
                        "cloud_waiting_timeouts": [],
                        "transmission_timeouts": [],
                        "edge_active_timeout": None,
                        "cloud_active_timeout": None,
                    }, self._interval_event_reward(interval_state_before), {
                        "advanced_time_s": self.current_time_s - start,
                        "event_count": event_count,
                        "completed_task_ids": list(self._interval_completed_ids),
                        "timeout_task_ids": list(self._interval_timeout_ids),
                    }

            if self.external_tasks_finished():
                return None, self._interval_event_reward(interval_state_before), {
                    "advanced_time_s": self.current_time_s - start,
                    "event_count": event_count,
                    "completed_task_ids": list(self._interval_completed_ids),
                    "timeout_task_ids": list(self._interval_timeout_ids),
                }

            next_time = self._next_event_time(include_arrival=True)
            if next_time is None:
                return None, self._interval_event_reward(interval_state_before), {
                    "advanced_time_s": self.current_time_s - start,
                    "event_count": event_count,
                    "completed_task_ids": list(self._interval_completed_ids),
                    "timeout_task_ids": list(self._interval_timeout_ids),
                }
            self._advance_clock(next_time)
            # If arrival and internal event coincide, process deadline/completion
            # first then expose the new arrival against the updated state.
            self._process_internal_events_now()
            event_count += 1
        raise RuntimeError("等待下一决策事件超过安全上限")

    def _interval_event_reward(self, state_before: SystemState) -> float:
        """Reward for physical outcomes between decisions.

        Uses the unchanged reward calculator with task=None. Timeout penalties and
        post-event queue penalties are therefore still governed by the same
        Phase4 numeric weights; only the transition cadence is event-based.
        """
        if not self._interval_timeout_ids and self.current_time_s <= state_before.current_time_s + EPS:
            return 0.0
        state_after = self.get_system_state()
        outcome = StepOutcome(
            task=None,
            action=Action.IDLE,
            state_before=state_before,
            state_after=state_after,
            timeout_count=len(self._interval_timeout_ids),
            completed_task_ids=list(self._interval_completed_ids),
            timeout_task_ids=list(self._interval_timeout_ids),
        )
        return float(self._reward_calculator.calculate(outcome)["reward"])

    # ----------------------------- decision interface -----------------------------
    def prepare_decision(self) -> dict:
        if self._pending_decision_task is not None:
            task = self._pending_decision_task
            return {"task": task, "state_before": self.get_system_state()}
        context, _, _ = self.advance_to_next_decision()
        if context is None:
            return {"task": None, "state_before": self.get_system_state()}
        return context

    @staticmethod
    def _empty_details() -> dict[str, float]:
        keys = [
            "edge_delay", "cloud_delay", "edge_waiting_time", "cloud_waiting_time",
            "edge_service_time", "cloud_service_time", "cloud_inference_time",
            "transmission_time", "network_rtt_s", "local_first_review_delay",
            "local_first_review_edge_wait_s", "local_first_review_edge_stage_s",
            "local_first_review_upload_mb", "local_first_review_upload_time_s",
            "local_first_review_network_rtt_s", "local_first_review_transfer_total_s",
            "local_first_review_cloud_wait_s", "local_first_review_cloud_stage_s",
            "local_first_review_remaining_s", "local_first_review_expected_accuracy",
        ]
        return {key: float("nan") for key in keys}

    def apply_decision(
        self,
        context: dict,
        action: Action | int,
        reason: str,
        details: dict | None = None,
    ) -> dict:
        task: Task | None = context.get("task")
        state_before_obj = context.get("state_before", self.get_system_state())
        state_before = state_before_obj if isinstance(state_before_obj, SystemState) else self.get_system_state()
        try:
            action = Action(action)
        except ValueError:
            raise ValueError(f"非法动作：{action}")

        if task is None:
            # Compatibility: IDLE no longer advances one physical slot. The caller
            # should use advance_to_next_decision().
            outcome = StepOutcome(
                task=None, action=Action.IDLE,
                state_before=state_before, state_after=self.get_system_state(),
            )
            reward_details = self._reward_calculator.calculate(outcome)
            record = self._build_record(None, Action.IDLE, "无新任务；Phase4.5不执行slot推进", self._empty_details(), reward_details)
            self._metrics.add_record(record)
            return record

        if task is not self._pending_decision_task and self._pending_decision_task is not None:
            raise RuntimeError("decision context与当前待调度任务不一致")
        if action not in {Action.LOCAL_FIRST, Action.DIRECT_CLOUD}:
            raise ValueError("存在新任务时只能选择LOCAL_FIRST或DIRECT_CLOUD")
        if details is None:
            details = self.service_time_model.estimate(task, state_before)

        self.enqueue_task(task, action, details)
        self._pending_decision_task = None
        self._start_available_workers()
        state_after = self.get_system_state()

        predicted_post_edge = None
        if action == Action.LOCAL_FIRST:
            preview = self.post_edge_policy.evaluate(
                task=task,
                edge_result=self._simulated_edge_result(task),
                now_s=float(task.arrival_time) + float(details["edge_delay"]),
                cloud_review_estimated_s=float(details["local_first_review_remaining_s"]),
            )
            predicted_post_edge = preview.to_dict()
            if preview.decision == PostEdgeDecision.CLOUD_REVIEW:
                selected_delay = float(details["local_first_review_delay"])
                expected_accuracy = float(task.collaborative_accuracy)
                upload_mb = float(details["local_first_review_upload_mb"])
                edge_compute = float(details["edge_service_time"])
                cloud_compute = float(details["local_first_review_cloud_stage_s"])
            else:
                selected_delay = float(details["edge_delay"])
                expected_accuracy = float(task.confidence)
                upload_mb = 0.0
                edge_compute = float(details["edge_service_time"])
                cloud_compute = 0.0
        else:
            selected_delay = float(details["cloud_delay"])
            expected_accuracy = float(task.estimated_cloud_accuracy if task.estimated_cloud_accuracy > 0 else self.cloud_accuracy)
            upload_mb = float(task.size_mb)
            edge_compute = 0.0
            cloud_compute = float(details["cloud_inference_time"])

        details = {**dict(details), "predicted_post_edge": predicted_post_edge}
        outcome = StepOutcome(
            task=task,
            action=action,
            state_before=state_before,
            state_after=state_after,
            timeout_count=0,
            selected_delay_s=selected_delay,
            expected_accuracy=expected_accuracy,
            upload_mb=upload_mb,
            edge_compute_s=edge_compute,
            cloud_compute_s=cloud_compute,
            details=dict(details),
        )
        reward_details = self._reward_calculator.calculate(outcome)
        self.cumulative_reward += float(reward_details["reward"])
        record = self._build_record(task, action, reason, details, reward_details)
        self._metrics.add_record(record)
        return record

    def _build_record(self, task, action, reason, details, reward_details) -> dict:
        state_after = self.get_system_state()
        return {
            "time": self.current_time_s,
            "time_s": round(self.current_time_s, 6),
            "slot_duration_s": self.slot_duration,
            "time_engine": "continuous_event",
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
            "latent_edge_confidence": np.nan if task is None else round(task.confidence, 3),
            "action": ROUTE_TO_PPO_ACTION.get(action, -1),
            "internal_action_value": int(action),
            "action_name": action.name,
            "reason": reason,
            "edge_estimated_delay": np.nan if task is None else round(details["edge_delay"], 4),
            "cloud_estimated_delay": np.nan if task is None else round(details["cloud_delay"], 4),
            "local_first_review_estimated_delay": np.nan if task is None else round(details["local_first_review_delay"], 4),
            "edge_service_time": np.nan if task is None else round(details["edge_service_time"], 4),
            "cloud_service_time": np.nan if task is None else round(details["cloud_service_time"], 4),
            "transmission_time": np.nan if task is None else round(details["transmission_time"], 4),
            "network_rtt_s": np.nan if task is None else round(details["network_rtt_s"], 4),
            "local_first_review_edge_stage_s": np.nan if task is None else round(details["local_first_review_edge_stage_s"], 4),
            "local_first_review_upload_time_s": np.nan if task is None else round(details["local_first_review_upload_time_s"], 4),
            "local_first_review_network_rtt_s": np.nan if task is None else round(details["local_first_review_network_rtt_s"], 4),
            "local_first_review_cloud_stage_s": np.nan if task is None else round(details["local_first_review_cloud_stage_s"], 4),
            "predicted_post_edge_decision": (
                None if task is None else (details.get("predicted_post_edge") or {}).get("decision")
            ),
            "post_edge_rule_version": None if task is None else self.post_edge_policy.rule_version,
            "selected_upload_mb": round(float(reward_details["selected_upload_mb"]), 6),
            "edge_queue": state_after.edge_queue,
            "cloud_queue": state_after.cloud_queue,
            "cloud_ready_queue": state_after.cloud_ready_queue,
            "transmission_queue": state_after.transmission_queue,
            "edge_workload_s": round(state_after.edge_workload_s, 4),
            "cloud_workload_s": round(state_after.cloud_workload_s, 4),
            "edge_resource": round(self.edge_resource, 3),
            "cloud_resource": round(self.cloud_resource, 3),
            "network_quality": round(self.network_quality, 3),
            "network_state": self.network_state,
            "bandwidth_mbps": round(self.bandwidth_mbps, 2),
            "timeout_count": int(reward_details.get("timeout_penalty", 0.0) > 0),
            "selected_delay_s": round(float(reward_details["selected_delay_s"]), 4),
            "deadline_ratio": round(float(reward_details["deadline_ratio"]), 4),
            "expected_accuracy": round(float(reward_details["expected_accuracy"]), 3),
            "delay_penalty": round(float(reward_details["delay_penalty"]), 4),
            "queue_penalty": round(float(reward_details["queue_penalty"]), 4),
            "timeout_penalty": round(float(reward_details["timeout_penalty"]), 4),
            "accuracy_penalty": round(float(reward_details["accuracy_penalty"]), 4),
            "communication_penalty": round(float(reward_details["communication_penalty"]), 4),
            "resource_penalty": round(float(reward_details["resource_penalty"]), 4),
            "lyapunov_drift": round(float(reward_details["lyapunov_drift"]), 4),
            "reward": round(float(reward_details["reward"]), 4),
            "cumulative_reward": round(self.cumulative_reward, 4),
        }

    # ------------------------------- public loops -------------------------------
    def step(self) -> dict:
        context = self.prepare_decision()
        task = context["task"]
        if task is None:
            return self.apply_decision(context, Action.IDLE, "没有更多可调度任务")
        action, reason, details = self.scheduler.select_action(task, context["state_before"])
        return self.apply_decision(context, action, reason, details)

    def run(self, steps: int) -> pd.DataFrame:
        for _ in range(int(steps)):
            if self.external_tasks_finished():
                break
            self.step()
        return self.get_records()

    def run_until_complete(self, max_steps: int = 20_000) -> pd.DataFrame:
        decisions = 0
        while not self.external_tasks_finished() and decisions < max_steps:
            context, event_reward, _ = self.advance_to_next_decision()
            self.cumulative_reward += event_reward
            if context is None:
                break
            task = context["task"]
            action, reason, details = self.scheduler.select_action(task, context["state_before"])
            self.apply_decision(context, action, reason, details)
            decisions += 1
        if decisions >= max_steps and not self.external_tasks_finished():
            raise RuntimeError("run_until_complete达到max_steps仍未完成")
        return self.get_records()

    def external_tasks_finished(self) -> bool:
        arrival_exhausted = self.arrival_model.next_arrival_time_s() is None
        if self.task_source != "edge_input":
            # Simulated arrivals are an open stream; PPO episodes terminate by
            # decision-step limit rather than source exhaustion.
            arrival_exhausted = False
        return (
            arrival_exhausted
            and self._pending_decision_task is None
            and not self.external_tasks
            and len(self.edge_queue) == 0
            and len(self.cloud_queue) == 0
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
            len(self.edge_queue) + len(self.cloud_queue) + len(self.transmission_queue)
            + int(self.edge_active is not None) + int(self.cloud_active is not None)
            + int(self._pending_decision_task is not None)
        )
        summary = self._metrics.summarize(
            generated_tasks=self.generated_task_count,
            in_system_tasks=in_system,
        )
        # Event rewards may occur after the final decision record (for example a
        # deadline expiry while draining edge_input tasks). Keep the simulator
        # accumulator authoritative.
        summary["cumulative_reward"] = float(self.cumulative_reward)
        return summary

    def reset(self) -> None:
        self._metrics.reset()
        self._initialize_runtime()


if __name__ == "__main__":
    sim = CloudEdgeSimulator()
    for _ in range(20):
        context, reward, info = sim.advance_to_next_decision()
        sim.cumulative_reward += reward
        if context is None:
            break
        task = context["task"]
        action, reason, details = sim.scheduler.select_action(task, context["state_before"])
        sim.apply_decision(context, action, reason, details)
    print(pd.Series(sim.get_metrics()).to_string())
