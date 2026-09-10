"""Background jobs with pollable progress.

Indexing a library takes minutes and scoring takes seconds; neither can block an
HTTP response. Jobs run on a worker thread and the UI polls for progress.

Only one job runs at a time. The model, the SQLite connection and the on-disk
renames are all single-writer resources, and serializing here is far simpler
than making each of them concurrent for a single-user desktop app.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    id: str
    kind: str
    state: str = "running"  # running | done | failed | cancelled
    message: str = ""
    current: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    """Runs one job at a time and keeps a bounded history of finished ones."""

    def __init__(self, history: int = 20):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._active: str | None = None
        self._history = history

    @property
    def active(self) -> Job | None:
        with self._lock:
            return self._jobs.get(self._active) if self._active else None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, kind: str, work: Callable[[Job], Any]) -> Job:
        """Start ``work`` on a thread. Raises if a job is already running."""
        with self._lock:
            if self._active and self._jobs[self._active].state == "running":
                raise RuntimeError(f"a {self._jobs[self._active].kind} job is already running")
            job = Job(id=uuid.uuid4().hex[:12], kind=kind)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._active = job.id
            self._prune()

        def run() -> None:
            try:
                job.result = work(job)
                job.state = "cancelled" if job.cancelled else "done"
            except Exception as exc:
                job.state = "failed"
                # Keep the traceback server-side; the UI shows only the message.
                job.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

        threading.Thread(target=run, name=f"siftr-{kind}", daemon=True).start()
        return job

    def _prune(self) -> None:
        while len(self._order) > self._history:
            stale = self._order.pop(0)
            if self._jobs.get(stale) and self._jobs[stale].state != "running":
                self._jobs.pop(stale, None)
