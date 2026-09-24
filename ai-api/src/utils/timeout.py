from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from threading import BoundedSemaphore, Lock
from typing import Any


class InferenceTimeoutError(Exception):
    """Raised when ML inference exceeds the configured timeout."""


class InferenceBusyError(Exception):
    """Raised when both native inference slots are occupied."""


class RequestWork:
    """Release request admission only after its background work has exited."""

    def __init__(self, release: Callable[[], None]):
        self._release = release
        self._lock = Lock()
        self._pending = 0
        self._closed = False
        self._released = False

    def begin(self) -> None:
        with self._lock:
            if self._closed and self._pending == 0:
                raise RuntimeError("Request work started after admission closed")
            self._pending += 1

    def done(self) -> None:
        with self._lock:
            self._pending -= 1
            release = self._closed and self._pending == 0 and not self._released
            if release:
                self._released = True
        if release:
            self._release()

    def track(self, future: Future) -> None:
        self.begin()
        future.add_done_callback(lambda _future: self.done())

    def close(self) -> None:
        with self._lock:
            self._closed = True
            release = self._pending == 0 and not self._released
            if release:
                self._released = True
        if release:
            self._release()


request_work: ContextVar[RequestWork | None] = ContextVar("request_work", default=None)


def track_request_work(future: Future) -> None:
    tracker = request_work.get()
    if tracker is not None:
        tracker.track(future)


async def tracked_to_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Cancel queued work, but retain request admission while a thread runs."""
    tracker = request_work.get()
    if tracker is None:
        return await asyncio.to_thread(fn, *args, **kwargs)

    tracker.begin()
    state = Lock()
    started = cancelled = False

    def work():
        nonlocal started
        with state:
            if cancelled:
                return None
            started = True
        try:
            return fn(*args, **kwargs)
        finally:
            tracker.done()

    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        return await task
    except BaseException:
        nonlocal_cancel = False
        with state:
            if not started:
                cancelled = True
                nonlocal_cancel = True
        if nonlocal_cancel:
            tracker.done()
        task.cancel()
        raise


# Two native calls per process, matching the blur profile's request/decode
# budget. A timed-out call keeps its slot until the native thread exits.
_timeout_executor: ThreadPoolExecutor | None = None
_capacity = BoundedSemaphore(2)
_state_lock = Lock()
_active: set[Future] = set()
_timed_out: set[Future] = set()
_stopping = False


def inference_healthy() -> bool:
    with _state_lock:
        return not _timed_out and not _stopping


def start_timeout_executor() -> None:
    global _timeout_executor, _stopping
    with _state_lock:
        if _active:
            raise RuntimeError("Previous native inference has not stopped")
        old = _timeout_executor
        _timeout_executor = None
        _stopping = False
    if old is not None:
        old.shutdown(wait=False, cancel_futures=True)


def shutdown_timeout_executor() -> bool:
    global _stopping
    with _state_lock:
        _stopping = True
        executor = _timeout_executor
    if executor is not None:
        # A stuck native call cannot be killed by Python; the process supervisor
        # replaces an unhealthy instance after its normal termination grace.
        executor.shutdown(wait=False, cancel_futures=True)
    with _state_lock:
        return not _active


def run_with_timeout(
    fn: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: int = 120,
) -> Any:
    """Run *fn* in a reusable thread pool with a timeout.

    Admission has no waiting queue. The timeout bounds caller wait, while a
    running native call remains counted until the thread actually finishes.

    Use this for batch ONNX calls or any inference path that genuinely needs a
    wall-clock timeout guard independent of Celery's task-level limits.

    For per-image calls inside Celery tasks, prefer :func:`run_direct` instead
    — those are already covered by ``task_soft_time_limit``.
    """
    if kwargs is None:
        kwargs = {}

    if not _capacity.acquire(blocking=False):
        raise InferenceBusyError("Inference capacity exhausted")
    try:
        global _timeout_executor
        with _state_lock:
            if _stopping:
                raise InferenceBusyError("Inference is shutting down")
            if _timeout_executor is None:
                _timeout_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="infer-timeout"
                )
            future = _timeout_executor.submit(fn, *args, **kwargs)
            _active.add(future)
    except BaseException:
        _capacity.release()
        raise

    def release(finished: Future) -> None:
        with _state_lock:
            _active.discard(finished)
            _timed_out.discard(finished)
        _capacity.release()

    future.add_done_callback(release)
    track_request_work(future)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        from src.utils.blur_metrics import timeouts

        timeouts.labels("native").inc()
        if not future.cancel():
            with _state_lock:
                if not future.done():
                    _timed_out.add(future)
        raise InferenceTimeoutError(f"Inference timed out after {timeout_seconds}s") from None


def run_direct(
    fn: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: int = 0,  # ignored, kept for call-site compat
) -> Any:
    """Call *fn* directly in the current thread — zero overhead.

    Use this for per-image ML inference inside Celery tasks where
    ``task_soft_time_limit`` already provides an outer safety net.
    The *timeout_seconds* parameter is accepted but ignored so that
    call sites can switch from ``run_with_timeout`` without changing args.
    """
    if kwargs is None:
        kwargs = {}
    return fn(*args, **kwargs)
