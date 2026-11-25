"""Single-consumer latest-value work lanes for inference plugins."""
from __future__ import annotations

import logging
import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
log = logging.getLogger("cctv_ai.scheduler")


class LatestTaskLane(Generic[T]):
    """Run work off the capture loop and retain at most one pending task.

    A task waiting behind slow inference is replaced by the newest task. This
    bounds memory and avoids spending CPU on stale frames.
    """

    def __init__(self, handler: Callable[[T], None], name: str):
        self.handler = handler
        self.name = name
        self._condition = threading.Condition()
        self._pending: T | None = None
        self._has_pending = False
        self._closed = False
        self.submitted = 0
        self.completed = 0
        self.dropped = 0
        self.last_ms = 0.0
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, task: T) -> None:
        with self._condition:
            if self._closed:
                return
            self.submitted += 1
            if self._has_pending:
                self.dropped += 1
            self._pending = task
            self._has_pending = True
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._has_pending or self._closed)
                if self._closed and not self._has_pending:
                    return
                task = self._pending
                self._pending = None
                self._has_pending = False
            started = threading_time()
            try:
                self.handler(task)  # type: ignore[arg-type]
            except Exception:
                log.exception("work lane %s failed", self.name)
            finally:
                self.last_ms = (threading_time() - started) * 1000
                self.completed += 1

    def stats(self) -> dict[str, int | float]:
        with self._condition:
            return {"queued": int(self._has_pending), "submitted": self.submitted,
                    "completed": self.completed, "dropped": self.dropped,
                    "last_ms": round(self.last_ms, 1)}

    def close(self, timeout: float = 1.0) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self.thread.join(timeout=timeout)


def threading_time() -> float:
    import time
    return time.perf_counter()
