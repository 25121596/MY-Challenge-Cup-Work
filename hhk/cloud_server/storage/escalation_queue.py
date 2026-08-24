"""Admission queue for edge-to-cloud review requests.

It is intentionally explicit that this store is in-memory. Production must
replace it with a durable queue before relying on it across cloud restarts.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from config import config
from storage.models import EscalationRequest, ReviewFeedback


class EscalationQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[str, str] = {}
        self._order: list[str] = []

    def submit(self, request: EscalationRequest) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        now_text = now.isoformat().replace("+00:00", "Z")
        with self._lock:
            existing_id = self._idempotency.get(request.idempotency_key)
            if existing_id:
                record = self._records[existing_id]
                return {**record, "status": "duplicate"}
            try:
                deadline = datetime.fromisoformat(request.deadline_at.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError("deadline_at must be RFC 3339")
            if deadline <= now:
                return {"review_id": request.review_id, "trace_id": request.trace_id,
                        "status": "expired", "queue_position": None, "accepted_at": now_text,
                        "deadline_at": request.deadline_at, "execution_status": "not_queued"}
            if len(self._order) >= config.max_escalation_queue:
                return {"review_id": request.review_id, "trace_id": request.trace_id,
                        "status": "rejected", "queue_position": None, "accepted_at": now_text,
                        "deadline_at": request.deadline_at, "execution_status": "not_queued"}
            record = {"review_id": request.review_id, "trace_id": request.trace_id,
                      "status": "accepted", "safety_priority": request.safety_priority,
                      "failure_probability": request.failure_probability,
                      "deadline_at": request.deadline_at, "accepted_at": now_text,
                      "execution_status": "queued", "request": request.model_dump()}
            self._records[request.review_id] = record
            self._idempotency[request.idempotency_key] = request.review_id
            self._order.append(request.review_id)
            safety_rank = {"P0": 0, "P1": 1, "P2": 2}
            self._order.sort(key=lambda item: (safety_rank[self._records[item]["safety_priority"]],
                                                self._records[item]["deadline_at"],
                                                self._records[item]["accepted_at"]))
            record["queue_position"] = self._order.index(request.review_id) + 1
            return record

    def get(self, escalation_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(escalation_id)
            if record is None:
                return None
            result = dict(record)
            if escalation_id in self._order:
                result["queue_position"] = self._order.index(escalation_id) + 1
            return result

    def add_feedback(self, escalation_id: str, feedback: ReviewFeedback) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(escalation_id)
            if record is None:
                return None
            if feedback.trace_id and feedback.trace_id != record["trace_id"]:
                raise ValueError("feedback trace_id does not match review")
            record["feedback"] = feedback.model_dump()
            return {"sample_id": feedback.sample_id, "trace_id": feedback.trace_id,
                    "status": "accepted", "storage": "in_memory"}


escalation_queue = EscalationQueue()
