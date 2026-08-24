from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import threading
import time
from typing import Any, Callable

from config import MAX_TOKENS, TEMPERATURE
from edge_result_adapter import normalize_edge_result
from policy_bundle_loader import PolicyBundle, PostEdgeDecision
from prompt_builder import build_cloud_review_prompt, build_common_prompt
from runtime_task import TaskStatus
from service_time_estimator import ServiceTimeEstimator


ACTION_NAMES = {0: "LOCAL_FIRST", 1: "DIRECT_CLOUD"}


@dataclass
class RealTaskEnvelope:
    task: Any
    raw_task: dict[str, Any]
    arrival_monotonic: float
    sequence: int
    absolute_deadline: float
    arrival_wall_time_s: float = 0.0
    requested_action_index: int | None = None
    requested_action_name: str | None = None
    action_index: int | None = None
    action_name: str | None = None
    final_route: str | None = None
    runtime_override: str | None = None
    post_edge_result: dict[str, Any] = field(default_factory=dict)
    edge_signals: dict[str, Any] = field(default_factory=dict)
    status: str = "waiting"
    execution_stage: str = "NOT_ROUTED"
    final_answer: str | None = None
    edge_result: dict[str, Any] | None = None
    cloud_result: dict[str, Any] | None = None
    error: str | None = None
    finish_reason: str | None = None
    estimated_edge_service_s: float = 0.0
    estimated_cloud_service_s: float = 0.0
    decision_state: dict[str, Any] = field(default_factory=dict)
    observation: list[float] = field(default_factory=list)
    policy_info: dict[str, Any] = field(default_factory=dict)
    resource_snapshots: dict[str, Any] = field(default_factory=dict)
    timestamps: dict[str, float] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def task_id(self) -> str:
        return str(
            getattr(self.task, "external_task_id", None)
            or getattr(self.task, "task_id", self.sequence)
        )

    @property
    def deadline_s(self) -> float:
        return float(getattr(self.task, "deadline_s", 0.0))

    def elapsed_s(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.arrival_monotonic)

    def remaining_deadline_s(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return self.absolute_deadline - current

    def is_terminal(self) -> bool:
        return self.done_event.is_set()


@dataclass(frozen=True)
class _PostEdgeTaskView:
    task_type: str
    absolute_deadline_s: float


class EDFTaskQueue:
    """Thread-safe non-preemptive EDF waiting queue."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._heap: list[tuple[float, int, RealTaskEnvelope]] = []
        self._condition = threading.Condition()
        self._closed = False

    def put(self, env: RealTaskEnvelope) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError(f"{self.name} queue已关闭")
            heapq.heappush(self._heap, (env.absolute_deadline, env.sequence, env))
            self._condition.notify()

    def get(self, timeout_s: float = 0.2) -> RealTaskEnvelope | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._heap and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if not self._heap:
                return None
            return heapq.heappop(self._heap)[2]

    def expire(self, now: float) -> list[RealTaskEnvelope]:
        expired: list[RealTaskEnvelope] = []
        with self._condition:
            kept: list[tuple[float, int, RealTaskEnvelope]] = []
            while self._heap:
                item = heapq.heappop(self._heap)
                env = item[2]
                if env.done_event.is_set():
                    continue
                if env.absolute_deadline <= now:
                    expired.append(env)
                else:
                    kept.append(item)
            for item in kept:
                heapq.heappush(self._heap, item)
        return expired

    def snapshot(self) -> list[RealTaskEnvelope]:
        with self._condition:
            ordered = sorted(self._heap)
            return [item[2] for item in ordered if not item[2].done_event.is_set()]

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def __len__(self) -> int:
        return len(self.snapshot())


class RealTaskManager:
    """Wall-clock task runtime preserving EDF and explicit lifecycle semantics.

    The edge/cloud workers are independent. New tasks may arrive and be routed
    while previous real model calls are still running. EDGE and CLOUD waiting
    queues are EDF and running work is non-preemptive.
    """

    def __init__(
        self,
        *,
        edge_client: Any,
        cloud_client: Any,
        network_monitor: Any,
        estimator: ServiceTimeEstimator | None = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        timeout_scan_s: float = 0.02,
        on_terminal: Callable[[RealTaskEnvelope], None] | None = None,
        profiler: Any | None = None,
    ) -> None:
        self.edge_client = edge_client
        self.cloud_client = cloud_client
        self.network_monitor = network_monitor
        self.estimator = estimator or ServiceTimeEstimator()
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout_scan_s = max(0.005, float(timeout_scan_s))
        self.on_terminal = on_terminal
        self.profiler = profiler

        self.edge_queue = EDFTaskQueue("edge")
        self.cloud_queue = EDFTaskQueue("cloud")
        self.transfer_queue = EDFTaskQueue("transmission")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sequence = 0
        self._all: list[RealTaskEnvelope] = []
        self._edge_active: RealTaskEnvelope | None = None
        self._cloud_active: RealTaskEnvelope | None = None
        self._transfer_active: RealTaskEnvelope | None = None
        self._edge_active_started = 0.0
        self._cloud_active_started = 0.0
        self._transfer_active_started = 0.0
        self._edge_active_expected = 0.0
        self._cloud_active_expected = 0.0
        self._transfer_active_expected = 0.0
        self.policy_bundle: PolicyBundle | None = None

    def configure_policy_bundle(self, bundle: PolicyBundle) -> None:
        """Attach the validated bundle that owns Post-edge policy semantics."""

        self.policy_bundle = bundle

    def set_profiler(self, profiler: Any | None) -> None:
        self.profiler = profiler

    def _capture_snapshot(self, env: RealTaskEnvelope, label: str) -> None:
        profiler = self.profiler
        if profiler is None:
            return
        try:
            snapshot = profiler.snapshot()
        except Exception:
            snapshot = None
        if snapshot is not None:
            env.resource_snapshots[label] = snapshot

    def start(self) -> None:
        if self._threads:
            return
        specs = [
            ("edge-worker", self._edge_loop),
            ("cloud-worker", self._cloud_loop),
            ("transfer-worker", self._transfer_loop),
            ("deadline-monitor", self._deadline_loop),
        ]
        for name, target in specs:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def close(self) -> None:
        self._stop.set()
        self.edge_queue.close()
        self.cloud_queue.close()
        self.transfer_queue.close()
        for thread in self._threads:
            thread.join(timeout=1.0)

    def register_arrival(self, task: Any) -> RealTaskEnvelope:
        now = time.monotonic()
        wall_now = time.time()
        with self._lock:
            self._sequence += 1
            env = RealTaskEnvelope(
                task=task,
                raw_task=dict(getattr(task, "edge_input", {}) or {}),
                arrival_monotonic=now,
                sequence=self._sequence,
                arrival_wall_time_s=wall_now,
                absolute_deadline=now + max(0.001, float(task.deadline_s)),
                status="waiting",
                execution_stage="NOT_ROUTED",
            )
            env.timestamps["arrival_monotonic"] = now
            self._all.append(env)
        task.status = self._task_status("WAITING")
        task.execution_stage = "NOT_ROUTED"
        return env

    @staticmethod
    def _task_status(name: str) -> Any:
        return getattr(TaskStatus, name)

    def route(
        self,
        env: RealTaskEnvelope,
        action_index: int,
        *,
        requested_action_index: int | None = None,
        state_before: dict[str, Any],
        observation: list[float],
        policy_info: dict[str, Any] | None = None,
    ) -> None:
        if int(action_index) not in ACTION_NAMES:
            raise ValueError(f"非法PPO动作: {action_index}")
        if env.is_terminal():
            return
        requested = int(action_index if requested_action_index is None else requested_action_index)
        if requested not in ACTION_NAMES:
            raise ValueError(f"非法PPO请求动作: {requested}")
        env.requested_action_index = requested
        env.requested_action_name = ACTION_NAMES[requested]
        env.action_index = int(action_index)
        env.action_name = ACTION_NAMES[int(action_index)]
        env.decision_state = dict(state_before)
        env.observation = [float(value) for value in observation]
        env.policy_info = dict(policy_info or {})
        env.runtime_override = env.policy_info.get("runtime_override")
        env.resource_snapshots["decision"] = {
            "wall_time_s": time.time(),
            "monotonic_s": time.monotonic(),
            "state": dict(state_before),
        }
        env.timestamps["decision_monotonic"] = time.monotonic()
        if env.remaining_deadline_s() <= 0:
            self._mark_timeout(env, "PPO_DECISION", "PPO决策完成前已超过deadline")
            return

        if action_index == 0:
            env.estimated_edge_service_s = self.estimator.estimate(env.task, "edge")
            env.estimated_cloud_service_s = self.estimator.estimate(env.task, "collab_cloud")
            self._mark_waiting(env, "EDGE_WAITING")
            self.edge_queue.put(env)
        elif action_index == 1:
            env.final_route = "CLOUD"
            env.estimated_cloud_service_s = self.estimator.estimate(env.task, "cloud")
            # Before the first real sample, cloud_base_latency is an
            # inference prior only. Add current transfer + RTT so workload keeps
            # the same semantic as measured cloud service time. Once real samples
            # exist, EWMA already measures the full client-side cloud call.
            if not self.estimator.has_measurement(env.task, "cloud"):
                bandwidth = max(0.1, float(state_before.get("bandwidth_mbps", 25.0)))
                network = state_before.get("real_network_metrics", {})
                rtt_s = (
                    max(0.0, float(network.get("rtt_ms", 0.0))) / 1000.0
                    if isinstance(network, dict)
                    else 0.0
                )
                env.estimated_cloud_service_s += float(env.task.size_mb) * 8.0 / bandwidth + rtt_s
            self._mark_waiting(env, "CLOUD_WAITING")
            self.cloud_queue.put(env)

    def _mark_waiting(self, env: RealTaskEnvelope, stage: str) -> None:
        with env._lock:
            env.status = "waiting"
            env.execution_stage = stage
            env.task.status = self._task_status("WAITING")
            env.task.execution_stage = stage
            env.timestamps[f"{stage.lower()}_at"] = time.monotonic()

    def _mark_running(self, env: RealTaskEnvelope, stage: str) -> bool:
        now = time.monotonic()
        if env.done_event.is_set():
            return False
        if env.absolute_deadline <= now:
            self._mark_timeout(env, stage, f"任务在{stage}开始前超过deadline")
            return False
        with env._lock:
            env.status = "running"
            env.execution_stage = stage
            env.task.status = self._task_status("RUNNING")
            env.task.execution_stage = stage
            env.timestamps[f"{stage.lower()}_start"] = now
            if "first_start_monotonic" not in env.timestamps:
                env.timestamps["first_start_monotonic"] = now
        return True

    def _mark_transmitting(self, env: RealTaskEnvelope) -> bool:
        now = time.monotonic()
        if env.done_event.is_set():
            return False
        if env.absolute_deadline <= now:
            self._mark_timeout(env, "TRANSMITTING", "协同任务传输前超过deadline")
            return False
        with env._lock:
            env.status = "transmitting"
            env.execution_stage = "TRANSMITTING"
            env.task.status = self._task_status("TRANSMITTING")
            env.task.execution_stage = "TRANSMITTING"
            env.timestamps["transmission_start"] = now
        return True

    def _mark_timeout(self, env: RealTaskEnvelope, stage: str, reason: str) -> None:
        with env._lock:
            if env.done_event.is_set():
                return
            now = time.monotonic()
            env.status = "timeout"
            env.execution_stage = "TIMEOUT"
            env.finish_reason = reason
            env.measurements["total_wall_s"] = env.elapsed_s(now)
            env.timestamps["finish_monotonic"] = now
            env.timestamps["timeout_monotonic"] = now
            env.measurements["deadline_overrun_s"] = max(
                0.0, env.elapsed_s(now) - env.deadline_s
            )
            env.error = env.error or (None if stage else "timeout")
            env.task.status = self._task_status("TIMEOUT")
            env.task.execution_stage = "TIMEOUT"
            env.done_event.set()
        self._capture_snapshot(env, "logical_timeout")
        self._notify_terminal(env)

    def _finish_success(self, env: RealTaskEnvelope, reason: str) -> None:
        now = time.monotonic()
        if env.done_event.is_set():
            return
        if now > env.absolute_deadline:
            self._mark_timeout(env, env.execution_stage, reason + "；但完成时已超过deadline")
            return
        with env._lock:
            env.status = "completed"
            env.execution_stage = "FINISHED"
            env.finish_reason = reason
            env.measurements["total_wall_s"] = env.elapsed_s(now)
            env.timestamps["finish_monotonic"] = now
            env.task.status = self._task_status("COMPLETED")
            env.task.execution_stage = "FINISHED"
            env.done_event.set()
        self._capture_snapshot(env, "logical_finish")
        self._notify_terminal(env)

    def _finish_error(self, env: RealTaskEnvelope, stage: str, exc: Exception) -> None:
        env.error = f"{type(exc).__name__}: {exc}"
        self._mark_timeout(env, stage, f"{stage}真实执行失败")

    def _notify_terminal(self, env: RealTaskEnvelope) -> None:
        callback = self.on_terminal
        if callback is not None:
            try:
                callback(env)
            except Exception:
                pass

    def _mark_edge_result_ready(self, env: RealTaskEnvelope) -> bool:
        now = time.monotonic()
        if env.done_event.is_set():
            return False
        if env.absolute_deadline <= now:
            self._mark_timeout(env, "EDGE_RESULT_READY", "边端结果产生时已超过deadline")
            return False
        with env._lock:
            env.status = "running"
            env.execution_stage = "EDGE_RESULT_READY"
            env.task.status = self._task_status("RUNNING")
            env.task.execution_stage = "EDGE_RESULT_READY"
            env.timestamps["edge_result_ready_at"] = now
        return True

    def _cloud_reachable_at_decision(self, env: RealTaskEnvelope) -> bool | None:
        network = env.decision_state.get("real_network_metrics", {})
        if not isinstance(network, dict) or "reachable" not in network:
            return None
        return bool(network.get("reachable"))

    def _estimated_cloud_review_remaining_s(self, env: RealTaskEnvelope) -> float:
        state = self.queue_state()
        bandwidth = max(0.1, float(env.decision_state.get("bandwidth_mbps", 25.0)))
        network = env.decision_state.get("real_network_metrics", {})
        rtt_s = (
            max(0.0, float(network.get("rtt_ms", 0.0))) / 1000.0
            if isinstance(network, dict)
            else 0.0
        )
        payload_mb = max(0.001, float(getattr(env.task, "size_mb", 0.0))) * max(
            0.0,
            float(getattr(env.task, "collaborative_payload_ratio", 0.2)),
        )
        transfer_s = payload_mb * 8.0 / bandwidth + rtt_s
        return max(0.0, float(state.get("cloud_workload_s", 0.0))) + transfer_s + max(
            0.001,
            float(env.estimated_cloud_service_s),
        )

    def _apply_post_edge_policy(self, env: RealTaskEnvelope) -> None:
        bundle = self.policy_bundle
        if bundle is None:
            raise RuntimeError("RealTaskManager尚未配置Scheduler policy bundle")
        if not self._mark_edge_result_ready(env):
            return
        signals = normalize_edge_result(env.edge_result or {})
        env.edge_signals = signals
        now = time.monotonic()
        result = bundle.post_edge_policy.evaluate(
            task=_PostEdgeTaskView(
                task_type=str(getattr(env.task, "task_type", "generic_anomaly")),
                absolute_deadline_s=env.absolute_deadline,
            ),
            edge_result=signals,
            now_s=now,
            cloud_review_estimated_s=self._estimated_cloud_review_remaining_s(env),
        )
        payload = result.to_dict()
        payload["policy_decision"] = result.decision.value
        payload["edge_escalation_hint"] = signals.get("edge_escalation_hint")
        payload["runtime_override"] = None

        decision = result.decision
        if decision == PostEdgeDecision.CLOUD_REVIEW and self._cloud_reachable_at_decision(env) is False:
            decision = PostEdgeDecision.ACCEPT_EDGE
            override = "CLOUD_REVIEW_TO_ACCEPT_EDGE_CLOUD_UNREACHABLE"
            payload.update({
                "decision": decision.value,
                "reason": f"{payload['reason']}_BUT_CLOUD_UNREACHABLE",
                "cloud_review_feasible": False,
                "runtime_override": override,
            })
            env.runtime_override = (
                override
                if not env.runtime_override
                else f"{env.runtime_override};{override}"
            )

        env.post_edge_result = payload
        env.timestamps["post_edge_decision_at"] = time.monotonic()
        if decision == PostEdgeDecision.ACCEPT_EDGE:
            env.final_route = "EDGE"
            env.final_answer = (env.edge_result or {}).get("answer")
            self._finish_success(env, "Post-edge接受边端结果")
            return

        env.final_route = "COLLABORATIVE"
        if self._mark_transmitting(env):
            self.transfer_queue.put(env)

    def _fallback_to_edge_after_review_failure(
        self,
        env: RealTaskEnvelope,
        *,
        stage: str,
        exc: Exception,
    ) -> bool:
        if env.edge_result is None or env.done_event.is_set():
            return False
        override = f"CLOUD_REVIEW_FAILURE_TO_ACCEPT_EDGE_{stage}"
        env.error = f"{type(exc).__name__}: {exc}"
        env.runtime_override = (
            override if not env.runtime_override else f"{env.runtime_override};{override}"
        )
        env.final_route = "EDGE"
        env.final_answer = env.edge_result.get("answer")
        env.post_edge_result.update({
            "decision": PostEdgeDecision.ACCEPT_EDGE.value,
            "reason": f"{env.post_edge_result.get('reason', 'CLOUD_REVIEW')}_BUT_{stage}_FAILED",
            "cloud_review_feasible": False,
            "runtime_override": override,
        })
        self._finish_success(env, f"{stage}失败，降级接受边端结果")
        return True

    def _edge_loop(self) -> None:
        while not self._stop.is_set():
            env = self.edge_queue.get()
            if env is None:
                continue
            if env.done_event.is_set():
                continue
            stage = "EDGE_RUNNING"
            if not self._mark_running(env, stage):
                continue
            expected = env.estimated_edge_service_s
            with self._lock:
                self._edge_active = env
                self._edge_active_started = time.monotonic()
                self._edge_active_expected = expected
            self._capture_snapshot(env, "edge_start")
            start = time.perf_counter()
            try:
                result = self.edge_client.chat(
                    build_common_prompt(env.raw_task),
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                measured = time.perf_counter() - start
                env.measurements["real_edge_stage_s"] = measured
                env.edge_result = result
                self._capture_snapshot(env, "edge_finish")
                self.estimator.update(
                    env.task,
                    "edge",
                    measured,
                )
                if env.done_event.is_set():
                    continue
                self._apply_post_edge_policy(env)
            except Exception as exc:
                self._finish_error(env, stage, exc)
            finally:
                with self._lock:
                    if self._edge_active is env:
                        self._edge_active = None
                        self._edge_active_expected = 0.0

    def _transfer_loop(self) -> None:
        while not self._stop.is_set():
            env = self.transfer_queue.get()
            if env is None:
                continue
            if env.done_event.is_set():
                continue
            if env.remaining_deadline_s() <= 0:
                self._mark_timeout(env, "TRANSMITTING", "协同任务等待传输时超过deadline")
                continue
            with self._lock:
                self._transfer_active = env
                self._transfer_active_started = time.monotonic()
                # A network transaction is normally much shorter than model inference.
                self._transfer_active_expected = min(0.25, max(0.005, env.deadline_s * 0.10))
            self._capture_snapshot(env, "transfer_start")
            try:
                draft = "" if env.edge_result is None else str(env.edge_result.get("answer", ""))
                payload_obj = {
                    "task_id": env.task_id,
                    "task_type": getattr(env.task, "task_type", None),
                    "edge_draft": draft,
                    "raw": env.raw_task,
                }
                payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
                target_bytes = int(
                    max(0.001, float(getattr(env.task, "size_mb", 0.0)))
                    * float(getattr(env.task, "collaborative_payload_ratio", 0.2))
                    * 1024.0
                    * 1024.0
                )
                if len(payload) < target_bytes:
                    payload += b"0" * (target_bytes - len(payload))
                start = time.perf_counter()
                transfer_s, mbps = self.network_monitor.probe_payload(payload)
                measured = time.perf_counter() - start
                env.measurements["real_transfer_stage_s"] = max(transfer_s, measured)
                env.measurements["real_transfer_bytes"] = float(len(payload))
                if mbps is not None:
                    env.measurements["real_transfer_mbps"] = float(mbps)
                env.timestamps["transmission_finish"] = time.monotonic()
                self._capture_snapshot(env, "transfer_finish")
                if env.done_event.is_set():
                    continue
                if env.remaining_deadline_s() <= 0:
                    self._mark_timeout(env, "TRANSMITTING", "协同任务真实传输后超过deadline")
                    continue
                self._mark_waiting(env, "CLOUD_REVIEW_WAITING")
                self.cloud_queue.put(env)
            except Exception as exc:
                if not self._fallback_to_edge_after_review_failure(
                    env,
                    stage="TRANSMITTING",
                    exc=exc,
                ):
                    self._finish_error(env, "TRANSMITTING", exc)
            finally:
                with self._lock:
                    if self._transfer_active is env:
                        self._transfer_active = None
                        self._transfer_active_expected = 0.0

    def _cloud_loop(self) -> None:
        while not self._stop.is_set():
            env = self.cloud_queue.get()
            if env is None:
                continue
            if env.done_event.is_set():
                continue
            is_collab = env.final_route == "COLLABORATIVE"
            stage = "CLOUD_REVIEW_RUNNING" if is_collab else "CLOUD_RUNNING"
            if not self._mark_running(env, stage):
                continue
            expected = env.estimated_cloud_service_s
            with self._lock:
                self._cloud_active = env
                self._cloud_active_started = time.monotonic()
                self._cloud_active_expected = expected
            self._capture_snapshot(env, "cloud_review_start" if is_collab else "cloud_start")
            start = time.perf_counter()
            try:
                if is_collab:
                    edge_answer = "" if env.edge_result is None else str(env.edge_result.get("answer", ""))
                    prompt = build_cloud_review_prompt(env.raw_task, edge_answer)
                    estimator_stage = "collab_cloud"
                else:
                    prompt = build_common_prompt(env.raw_task)
                    estimator_stage = "cloud"
                result = self.cloud_client.chat(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                measured = time.perf_counter() - start
                env.measurements["real_cloud_stage_s"] = measured
                env.cloud_result = result
                self._capture_snapshot(env, "cloud_review_finish" if is_collab else "cloud_finish")
                self.estimator.update(env.task, estimator_stage, measured)
                if env.done_event.is_set():
                    continue
                env.final_answer = result.get("answer")
                self._finish_success(
                    env,
                    "COLLABORATIVE云端复核完成" if is_collab else "CLOUD真实执行完成",
                )
            except Exception as exc:
                if not (
                    is_collab
                    and self._fallback_to_edge_after_review_failure(
                        env,
                        stage="CLOUD_REVIEW_RUNNING",
                        exc=exc,
                    )
                ):
                    self._finish_error(env, stage, exc)
            finally:
                with self._lock:
                    if self._cloud_active is env:
                        self._cloud_active = None
                        self._cloud_active_expected = 0.0

    def _deadline_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            for queue, stage in (
                (self.edge_queue, "EDGE_WAITING"),
                (self.cloud_queue, "CLOUD_WAITING"),
                (self.transfer_queue, "TRANSMITTING"),
            ):
                for env in queue.expire(now):
                    timeout_stage = (
                        "CLOUD_REVIEW_WAITING"
                        if stage == "CLOUD_WAITING" and env.final_route == "COLLABORATIVE"
                        else stage
                    )
                    self._mark_timeout(env, timeout_stage, f"任务在{timeout_stage}阶段超过deadline")

            with self._lock:
                active_rows = [
                    (self._edge_active, "EDGE_RUNNING"),
                    (self._cloud_active, "CLOUD_RUNNING"),
                    (self._transfer_active, "TRANSMITTING"),
                ]
            for env, stage in active_rows:
                if env is not None and not env.done_event.is_set() and env.absolute_deadline <= now:
                    if stage == "CLOUD_RUNNING" and env.final_route == "COLLABORATIVE":
                        stage = "CLOUD_REVIEW_RUNNING"
                    self._mark_timeout(env, stage, f"任务在{stage}阶段超过deadline")
            self._stop.wait(self.timeout_scan_s)

    @staticmethod
    def _active_remaining(
        env: RealTaskEnvelope | None,
        started: float,
        expected: float,
        now: float,
    ) -> float:
        if env is None:
            return 0.0
        elapsed = max(0.0, now - started)
        # If a real call outlives the current estimate, keep a small positive
        # workload while the resource is still occupied.
        return max(0.02, expected - elapsed)

    def queue_state(self) -> dict[str, Any]:
        now = time.monotonic()
        edge_waiting = self.edge_queue.snapshot()
        cloud_waiting = self.cloud_queue.snapshot()
        transmissions = self.transfer_queue.snapshot()
        with self._lock:
            edge_active = self._edge_active
            cloud_active = self._cloud_active
            transfer_active = self._transfer_active
            edge_active_remaining = self._active_remaining(
                edge_active, self._edge_active_started, self._edge_active_expected, now
            )
            cloud_active_remaining = self._active_remaining(
                cloud_active, self._cloud_active_started, self._cloud_active_expected, now
            )
            transfer_active_remaining = self._active_remaining(
                transfer_active,
                self._transfer_active_started,
                self._transfer_active_expected,
                now,
            )

        edge_wait_s = sum(max(0.001, env.estimated_edge_service_s) for env in edge_waiting)
        cloud_wait_s = sum(max(0.001, env.estimated_cloud_service_s) for env in cloud_waiting)
        # cloud_queue/cloud_workload reserve Post-edge review tasks that are
        # still transmitting but already committed to cloud review.
        reserved_cloud_s = sum(
            max(0.001, env.estimated_cloud_service_s) for env in transmissions
        )
        if transfer_active is not None and transfer_active.final_route == "COLLABORATIVE":
            reserved_cloud_s += max(0.001, transfer_active.estimated_cloud_service_s)

        transmission_count = len(transmissions) + int(transfer_active is not None)
        return {
            "edge_queue": len(edge_waiting),
            "cloud_queue": len(cloud_waiting) + transmission_count,
            "cloud_ready_queue": len(cloud_waiting),
            "transmission_queue": transmission_count,
            "edge_busy": edge_active is not None,
            "cloud_busy": cloud_active is not None,
            "edge_workload_s": edge_active_remaining + edge_wait_s,
            "cloud_workload_s": (
                cloud_active_remaining
                + cloud_wait_s
                + reserved_cloud_s
                + transfer_active_remaining
            ),
        }

    def all_envelopes(self) -> list[RealTaskEnvelope]:
        with self._lock:
            return list(self._all)

    def wait_all(self, timeout_s: float = 60.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() < deadline:
            rows = self.all_envelopes()
            all_terminal = bool(rows) and all(env.done_event.is_set() for env in rows)
            state = self.queue_state()
            no_physical_work = (
                state["edge_queue"] == 0
                and state["cloud_ready_queue"] == 0
                and state["transmission_queue"] == 0
                and not state["edge_busy"]
                and not state["cloud_busy"]
                and self._transfer_active is None
            )
            if all_terminal and no_physical_work:
                return True
            time.sleep(0.01)
        return False

    def records(self) -> list[dict[str, Any]]:
        return [self.envelope_to_record(env) for env in self.all_envelopes()]

    def envelope_to_record(self, env: RealTaskEnvelope) -> dict[str, Any]:
        task = env.task
        arrival = env.arrival_monotonic

        def rel(key: str) -> float | None:
            value = env.timestamps.get(key)
            return None if value is None else round(max(0.0, value - arrival), 6)

        def duration(start_key: str, end_key: str) -> float | None:
            start_v = env.timestamps.get(start_key)
            end_v = env.timestamps.get(end_key)
            if start_v is None or end_v is None:
                return None
            return round(max(0.0, end_v - start_v), 6)

        decision_state = env.decision_state if isinstance(env.decision_state, dict) else {}
        network = decision_state.get("real_network_metrics", {}) if isinstance(decision_state, dict) else {}
        edge_metrics = decision_state.get("real_edge_metrics", {}) if isinstance(decision_state, dict) else {}
        cloud_metrics = decision_state.get("real_cloud_metrics", {}) if isinstance(decision_state, dict) else {}
        edge_profile = (env.edge_result or {}).get("profile", {}) if isinstance(env.edge_result, dict) else {}
        cloud_profile = (env.cloud_result or {}).get("profile", {}) if isinstance(env.cloud_result, dict) else {}
        raw_bytes = len(json.dumps(env.raw_task, ensure_ascii=False).encode("utf-8"))
        input_tokens = edge_profile.get("prompt_tokens") or cloud_profile.get("prompt_tokens")
        policy_probs = env.policy_info.get("action_probabilities")
        policy_value = env.policy_info.get("value_estimate")
        post_edge = env.post_edge_result if isinstance(env.post_edge_result, dict) else {}
        asset_routing = env.policy_info.get("asset_routing", {})
        asset_routing = asset_routing if isinstance(asset_routing, dict) else {}

        edge_wait = duration("edge_waiting_at", "edge_running_start")
        cloud_wait = duration("cloud_waiting_at", "cloud_running_start")
        review_wait = duration("cloud_review_waiting_at", "cloud_review_running_start")

        return {
            "schema_version": "cloud-edge-real-trace/2.0",
            "bundle_schema_version": (
                self.policy_bundle.manifest.get("bundle_schema_version")
                if self.policy_bundle is not None
                else None
            ),
            "policy_version": (
                self.policy_bundle.policy_version if self.policy_bundle is not None else None
            ),
            "post_edge_rule_version": (
                self.policy_bundle.rule_version if self.policy_bundle is not None else None
            ),
            "task_id": env.task_id,
            "task_type": getattr(task, "task_type", None),
            "priority": getattr(task, "priority", None),
            "deadline_s": float(getattr(task, "deadline_s", 0.0)),
            "input_size_bytes": raw_bytes,
            "size_mb": float(getattr(task, "size_mb", 0.0)),
            "input_tokens": input_tokens,
            "task_profile_confidence": float(getattr(task, "confidence", 0.0)),
            "complexity": float(getattr(task, "complexity", 0.0)),
            "arrival_wall_time_s": env.arrival_wall_time_s,
            "requested_action_index": env.requested_action_index,
            "requested_action_name": env.requested_action_name,
            "ppo_action_index": env.policy_info.get("ppo_action_index"),
            "ppo_action_name": env.policy_info.get("ppo_action_name"),
            "asset_routing_enabled": asset_routing.get("enabled"),
            "asset_routing_policy_version": asset_routing.get("policy_version"),
            "asset_routing_reason": asset_routing.get("reason"),
            "asset_routing_matched_by": asset_routing.get("matched_by"),
            "asset_routing_route": asset_routing.get("route"),
            "action_index": env.action_index,
            "action_name": env.action_name,
            "final_route": env.final_route,
            "runtime_override": env.runtime_override,
            "policy_action_probabilities": policy_probs,
            "policy_value_estimate": policy_value,
            "post_edge_policy_decision": post_edge.get("policy_decision"),
            "post_edge_decision": post_edge.get("decision"),
            "post_edge_reason": post_edge.get("reason"),
            "post_edge_rule_id": post_edge.get("rule_id"),
            "post_edge_review_recommended": post_edge.get("review_recommended"),
            "post_edge_cloud_review_feasible": post_edge.get("cloud_review_feasible"),
            "post_edge_confidence_threshold": post_edge.get("confidence_threshold"),
            "post_edge_failure_probability": post_edge.get("failure_probability"),
            "post_edge_review_state": post_edge.get("edge_review_state"),
            "edge_confidence": env.edge_signals.get("confidence"),
            "edge_failure_probability": env.edge_signals.get("failure_probability"),
            "edge_local_prediction": env.edge_signals.get("local_prediction"),
            "edge_review_state": env.edge_signals.get("review_state"),
            "edge_safety_priority": env.edge_signals.get("safety_priority"),
            "edge_safety_trigger": env.edge_signals.get("safety_trigger"),
            "edge_action": env.edge_signals.get("edge_action"),
            "edge_model_route": env.edge_signals.get("model_route"),
            "edge_level": env.edge_signals.get("level"),
            "edge_category": env.edge_signals.get("category"),
            "edge_output_valid": env.edge_signals.get("output_valid"),
            "edge_ood": env.edge_signals.get("ood"),
            "edge_escalation_hint": env.edge_signals.get("edge_escalation_hint"),
            "status": env.status,
            "execution_stage": env.execution_stage,
            "finish_reason": env.finish_reason,
            "error": env.error,
            "arrival_t": 0.0,
            "decision_t": rel("decision_monotonic"),
            "post_edge_decision_t": rel("post_edge_decision_at"),
            "first_start_t": rel("first_start_monotonic"),
            "edge_running_t": rel("edge_running_start"),
            "transmission_start_t": rel("transmission_start"),
            "transmission_finish_t": rel("transmission_finish"),
            "cloud_running_t": rel("cloud_running_start"),
            "cloud_review_running_t": rel("cloud_review_running_start"),
            "finish_t": rel("finish_monotonic"),
            "edge_queue_wait_s": edge_wait,
            "cloud_queue_wait_s": cloud_wait,
            "cloud_review_queue_wait_s": review_wait,
            "real_edge_stage_s": env.measurements.get("real_edge_stage_s"),
            "real_transfer_stage_s": env.measurements.get("real_transfer_stage_s"),
            "real_cloud_stage_s": env.measurements.get("real_cloud_stage_s"),
            "real_total_wall_s": env.measurements.get("total_wall_s"),
            "deadline_overrun_s": env.measurements.get("deadline_overrun_s", 0.0),
            "real_transfer_bytes": env.measurements.get("real_transfer_bytes"),
            "real_transfer_mbps": env.measurements.get("real_transfer_mbps"),
            "edge_queue_at_decision": decision_state.get("edge_queue"),
            "cloud_queue_at_decision": decision_state.get("cloud_queue"),
            "edge_workload_s_at_decision": decision_state.get("edge_workload_s"),
            "cloud_workload_s_at_decision": decision_state.get("cloud_workload_s"),
            "edge_resource_at_decision": decision_state.get("edge_resource"),
            "cloud_resource_at_decision": decision_state.get("cloud_resource"),
            "network_quality_at_decision": decision_state.get("network_quality"),
            "bandwidth_mbps_at_decision": decision_state.get("bandwidth_mbps"),
            "rtt_ms_at_decision": network.get("rtt_ms") if isinstance(network, dict) else None,
            "jitter_ms_at_decision": network.get("jitter_ms") if isinstance(network, dict) else None,
            "network_failure_rate_at_decision": network.get("failure_rate") if isinstance(network, dict) else None,
            "packet_loss_pct_at_decision": network.get("packet_loss_pct") if isinstance(network, dict) else None,
            "network_rtt_window_ms": network.get("rtt_window_ms") if isinstance(network, dict) else [],
            "edge_cpu_pct_at_decision": edge_metrics.get("cpu_usage_pct") if isinstance(edge_metrics, dict) else None,
            "edge_memory_available_mb_at_decision": edge_metrics.get("memory_available_mb") if isinstance(edge_metrics, dict) else None,
            "edge_memory_total_mb_at_decision": edge_metrics.get("memory_total_mb") if isinstance(edge_metrics, dict) else None,
            "edge_gpu_usage_pct_at_decision": edge_metrics.get("gpu_usage_pct") if isinstance(edge_metrics, dict) else None,
            "edge_gpu_free_mb_at_decision": edge_metrics.get("gpu_free_mb") if isinstance(edge_metrics, dict) else None,
            "edge_gpu_total_mb_at_decision": edge_metrics.get("gpu_total_mb") if isinstance(edge_metrics, dict) else None,
            "edge_tps_at_decision": edge_metrics.get("current_tps") if isinstance(edge_metrics, dict) else None,
            "cloud_cpu_pct_at_decision": cloud_metrics.get("cpu_usage_pct") if isinstance(cloud_metrics, dict) else None,
            "cloud_memory_available_mb_at_decision": cloud_metrics.get("memory_available_mb") if isinstance(cloud_metrics, dict) else None,
            "cloud_memory_total_mb_at_decision": cloud_metrics.get("memory_total_mb") if isinstance(cloud_metrics, dict) else None,
            "cloud_gpu_usage_pct_at_decision": cloud_metrics.get("gpu_usage_pct") if isinstance(cloud_metrics, dict) else None,
            "cloud_gpu_free_mb_at_decision": cloud_metrics.get("gpu_free_mb") if isinstance(cloud_metrics, dict) else None,
            "cloud_gpu_total_mb_at_decision": cloud_metrics.get("gpu_total_mb") if isinstance(cloud_metrics, dict) else None,
            "cloud_tps_at_decision": cloud_metrics.get("current_tps") if isinstance(cloud_metrics, dict) else None,
            "edge_model_id": (env.edge_result or {}).get("model") if isinstance(env.edge_result, dict) else None,
            "edge_endpoint": (env.edge_result or {}).get("endpoint") if isinstance(env.edge_result, dict) else None,
            "edge_prompt_bytes": edge_profile.get("request_prompt_bytes"),
            "edge_prompt_tokens": edge_profile.get("prompt_tokens"),
            "edge_completion_tokens": edge_profile.get("completion_tokens"),
            "edge_total_tokens": edge_profile.get("total_tokens"),
            "edge_cached_tokens": edge_profile.get("cached_tokens"),
            "edge_reasoning_tokens": edge_profile.get("reasoning_tokens"),
            "edge_prompt_ms": edge_profile.get("prompt_ms"),
            "edge_completion_ms": edge_profile.get("completion_ms"),
            "edge_prompt_tps": edge_profile.get("prompt_tokens_per_s"),
            "edge_completion_tps": edge_profile.get("completion_tokens_per_s"),
            "edge_ttft_ms": edge_profile.get("ttft_ms"),
            "cloud_model_id": (env.cloud_result or {}).get("model") if isinstance(env.cloud_result, dict) else None,
            "cloud_endpoint": (env.cloud_result or {}).get("endpoint") if isinstance(env.cloud_result, dict) else None,
            "cloud_prompt_bytes": cloud_profile.get("request_prompt_bytes"),
            "cloud_prompt_tokens": cloud_profile.get("prompt_tokens"),
            "cloud_completion_tokens": cloud_profile.get("completion_tokens"),
            "cloud_total_tokens": cloud_profile.get("total_tokens"),
            "cloud_cached_tokens": cloud_profile.get("cached_tokens"),
            "cloud_reasoning_tokens": cloud_profile.get("reasoning_tokens"),
            "cloud_prompt_ms": cloud_profile.get("prompt_ms"),
            "cloud_completion_ms": cloud_profile.get("completion_ms"),
            "cloud_prompt_tps": cloud_profile.get("prompt_tokens_per_s"),
            "cloud_completion_tps": cloud_profile.get("completion_tokens_per_s"),
            "cloud_ttft_ms": cloud_profile.get("ttft_ms"),
            "edge_model_profile": edge_profile,
            "cloud_model_profile": cloud_profile,
            "resource_snapshots": env.resource_snapshots,
            "observation_24d": env.observation,
            "final_answer": env.final_answer,
        }
