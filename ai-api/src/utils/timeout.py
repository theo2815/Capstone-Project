from __future__ import annotations

import threading
from typing import Any, Callable


class InferenceTimeoutError(Exception):
    """Raised when ML inference exceeds the configured timeout."""


def run_with_timeout(
    fn: Callable[..., Any],
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: int = 120,
) -> Any:
    """Run a callable in a daemon thread with a timeout.

    Args:
        fn: The function to execute.
        args: Positional arguments for fn.
        kwargs: Keyword arguments for fn.
        timeout_seconds: Maximum wall-clock seconds to wait.

    Returns:
        The return value of fn.

    Raises:
        InferenceTimeoutError: If fn does not complete within timeout_seconds.
        Exception: Any exception raised by fn is re-raised.
    """
    if kwargs is None:
        kwargs = {}

    result: list[Any] = []
    error: list[BaseException] = []

    def _target():
        try:
            result.append(fn(*args, **kwargs))
        except BaseException as e:
            error.append(e)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise InferenceTimeoutError(
            f"Inference timed out after {timeout_seconds}s"
        )

    if error:
        raise error[0]

    return result[0] if result else None
