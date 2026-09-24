from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.utils import timeout as inference


def test_timed_out_queued_work_never_starts(monkeypatch):
    inference.start_timeout_executor()
    started = threading.Event()
    release = threading.Event()
    marker = threading.Event()

    def slow():
        started.set()
        release.wait(2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(inference, "_timeout_executor", executor)
        try:
            with pytest.raises(inference.InferenceTimeoutError):
                inference.run_with_timeout(slow, timeout_seconds=0.02)
            assert started.is_set()
            with pytest.raises((inference.InferenceTimeoutError, inference.InferenceBusyError)):
                inference.run_with_timeout(marker.set, timeout_seconds=0.02)
        finally:
            release.set()
    assert not marker.is_set()


def test_timed_out_running_work_keeps_capacity_and_unhealthy_state(monkeypatch):
    inference.start_timeout_executor()
    release = threading.Event()
    started = threading.Event()

    def slow():
        started.set()
        release.wait(2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(inference, "_timeout_executor", executor)
        monkeypatch.setattr(inference, "_capacity", threading.BoundedSemaphore(1))
        try:
            with pytest.raises(inference.InferenceTimeoutError):
                inference.run_with_timeout(slow, timeout_seconds=0.02)
            assert started.is_set()
            assert not inference.inference_healthy()
            with pytest.raises(inference.InferenceBusyError):
                inference.run_with_timeout(lambda: None, timeout_seconds=0.02)
        finally:
            release.set()
    assert inference.inference_healthy()


def test_queued_future_is_cancelled_on_timeout(monkeypatch):
    inference.start_timeout_executor()
    release = threading.Event()
    started = threading.Event()
    marker = threading.Event()

    def slow():
        started.set()
        release.wait(2)

    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(inference, "_timeout_executor", executor)
        monkeypatch.setattr(inference, "_capacity", threading.BoundedSemaphore(1))
        executor.submit(slow)
        assert started.wait(1)
        try:
            with pytest.raises(inference.InferenceTimeoutError):
                inference.run_with_timeout(marker.set, timeout_seconds=0.02)
        finally:
            release.set()
    assert not marker.is_set()


def test_shutdown_does_not_create_a_replacement_executor(monkeypatch):
    inference.start_timeout_executor()
    release = threading.Event()
    marker = threading.Event()
    monkeypatch.setattr(inference, "_stopping", False, raising=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(inference, "_timeout_executor", executor)
        monkeypatch.setattr(inference, "_capacity", threading.BoundedSemaphore(2))
        try:
            with pytest.raises(inference.InferenceTimeoutError):
                inference.run_with_timeout(lambda: release.wait(2), timeout_seconds=0.02)
            inference.shutdown_timeout_executor()
            with pytest.raises(inference.InferenceBusyError):
                inference.run_with_timeout(marker.set, timeout_seconds=0.02)
        finally:
            release.set()
    assert not marker.is_set()


def test_shutdown_rejects_new_work_after_native_calls_drain():
    inference.start_timeout_executor()
    assert inference.shutdown_timeout_executor()
    with pytest.raises(inference.InferenceBusyError):
        inference.run_with_timeout(lambda: None, timeout_seconds=0.02)
    inference.start_timeout_executor()


def test_native_work_registered_after_request_close_keeps_admission(monkeypatch):
    inference.start_timeout_executor()
    started = threading.Event()
    release = threading.Event()
    admission_released = threading.Event()
    tracker = inference.RequestWork(admission_released.set)
    tracker.begin()  # An already-running tracked_to_thread outer call.
    tracker.close()  # Request cancellation races with native future submission.
    token = inference.request_work.set(tracker)

    def slow():
        started.set()
        release.wait(2)

    try:
        with pytest.raises(inference.InferenceTimeoutError):
            inference.run_with_timeout(slow, timeout_seconds=0.02)
        assert started.is_set()
        tracker.done()  # The canceled outer call exits, but native work has not.
        assert not admission_released.is_set()
    finally:
        release.set()
        inference.request_work.reset(token)
    assert admission_released.wait(1)


async def test_cancelled_thread_call_stays_accounted_until_it_exits():
    started = threading.Event()
    release = threading.Event()
    admission_released = threading.Event()
    tracker = inference.RequestWork(admission_released.set)
    token = inference.request_work.set(tracker)

    def slow():
        started.set()
        release.wait(2)

    try:
        task = asyncio.create_task(inference.tracked_to_thread(slow))
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        tracker.close()
        assert not admission_released.is_set()
    finally:
        release.set()
        inference.request_work.reset(token)
    assert await asyncio.to_thread(admission_released.wait, 1)


async def test_cancelled_queued_thread_call_never_runs():
    blocker = threading.Event()
    marker = threading.Event()
    admission_released = threading.Event()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.set_default_executor(executor)
        first = asyncio.create_task(asyncio.to_thread(blocker.wait, 2))
        await asyncio.sleep(0.01)
        tracker = inference.RequestWork(admission_released.set)
        token = inference.request_work.set(tracker)
        try:
            queued = asyncio.create_task(inference.tracked_to_thread(marker.set))
            await asyncio.sleep(0.01)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            tracker.close()
        finally:
            blocker.set()
            inference.request_work.reset(token)
        await first
    assert admission_released.is_set()
    assert not marker.is_set()
