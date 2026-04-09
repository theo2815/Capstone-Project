from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class InferenceTimeoutError(Exception):
    """Raised when ML inference exceeds the configured timeout."""


# Single reusable thread for timeout-wrapped calls — avoids spawning a new
# daemon thread per inference call (which caused 60k+ thread creates for 20k
# images).  The executor is lazily initialised so import alone is free.
_timeout_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _timeout_executor
    if _timeout_executor is None:
        _timeout_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer-timeout")
    return _timeout_executor


def run_with_timeout(
    fn: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: int = 120,
) -> Any:
    """Run *fn* in a reusable thread pool with a timeout.

    Unlike the previous implementation that spawned a **new** daemon thread per
    call, this reuses a single-thread ``ThreadPoolExecutor``.  The thread stays
    alive between calls, eliminating per-call thread creation/join overhead.

    Use this for batch ONNX calls or any inference path that genuinely needs a
    wall-clock timeout guard independent of Celery's task-level limits.

    For per-image calls inside Celery tasks, prefer :func:`run_direct` instead
    — those are already covered by ``task_soft_time_limit``.
    """
    if kwargs is None:
        kwargs = {}

    future = _get_executor().submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        raise InferenceTimeoutError(
            f"Inference timed out after {timeout_seconds}s"
        )


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
