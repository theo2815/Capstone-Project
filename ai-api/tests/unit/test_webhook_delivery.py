from __future__ import annotations

import contextlib
import hashlib
import hmac
import ipaddress
import socket
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from src.workers.tasks import webhook_tasks

PUBLIC_V4 = "93.184.216.34"
PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"


def _answers(*ips):
    return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
            for ip in ips]


def _deliver(url, *ips, **task_kwargs):
    """Run one delivery with a fake resolver and a fake network; return the requests sent."""
    seen = []
    real_client = httpx.AsyncClient

    def handler(request):
        seen.append(request)
        return httpx.Response(200)

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    with patch.object(webhook_tasks.socket, "getaddrinfo", return_value=_answers(*ips)), \
            patch.object(webhook_tasks.httpx, "AsyncClient", client):
        webhook_tasks.deliver_webhook.run(url, "job.completed", {"job_id": "j1"}, **task_kwargs)
    return seen


def test_userinfo_url_is_never_delivered():
    assert _deliver("http://a.test@a.test/x", PUBLIC_V4) == []


def test_one_internal_answer_blocks_delivery():
    assert _deliver("http://a.test/x", PUBLIC_V4, "127.0.0.1") == []


# Python 3.12.0's ipaddress still calls 192.0.0.8 public; later patch releases (CI, Docker) do not.
_OLD_IPADDRESS = ipaddress.ip_address("192.0.0.8").is_global


@pytest.mark.parametrize("ip", ["100.64.0.1", "198.18.0.1",
                                pytest.param("192.0.0.8", marks=pytest.mark.skipif(
                                    _OLD_IPADDRESS, reason="this Python's ipaddress is too old")),
                                "240.0.0.1", "10.0.0.1", "169.254.169.254", "::ffff:127.0.0.1",
                                "fd00::1"])
def test_non_public_answers_are_blocked(ip):
    assert _deliver("http://a.test/x", ip) == []


def test_https_goes_to_the_checked_ip_with_the_hostname_for_tls():
    [request] = _deliver("https://hooks.example.com:8443/x?y=1", PUBLIC_V4)
    assert (request.url.host, request.url.port, request.url.path, request.url.query) == \
        (PUBLIC_V4, 8443, "/x", b"y=1")
    assert request.headers["host"] == "hooks.example.com:8443"
    assert request.extensions["sni_hostname"] == "hooks.example.com"


def test_ipv6_answer_is_used_correctly():
    [request] = _deliver("https://hooks.example.com/x", PUBLIC_V6)
    assert request.url.host == PUBLIC_V6


def test_ipv6_literal_url_keeps_its_port_and_host():
    [request] = _deliver(f"https://[{PUBLIC_V6}]:8443/x", PUBLIC_V6)
    assert (request.url.host, request.url.port) == (PUBLIC_V6, 8443)
    assert request.headers["host"] == f"[{PUBLIC_V6}]:8443"


def test_nat64_answer_is_checked_by_the_ipv4_inside_it():
    # 64:ff9b::/96 wraps an IPv4 address in its last 32 bits (10.0.0.1, then 93.184.216.34).
    assert _deliver("http://a.test/x", "64:ff9b::a00:1") == []
    assert len(_deliver("http://a.test/x", "64:ff9b::5db8:d822")) == 1


def test_port_zero_is_not_dropped():
    [request] = _deliver("https://hooks.example.com:0/x", PUBLIC_V4)
    assert request.url.port == 0


@pytest.mark.parametrize("url,ok", [
    ("http://a@hooks.example.com/x", False),
    ("http://100.64.0.1/x", False),
    ("http://127.0.0.1/x", False),
    ("https://hooks.example.com/x", True),
])
def test_registration_check(url, ok):
    from src.api.v1.webhooks import _validate_webhook_url_basic

    assert (_validate_webhook_url_basic(url) is None) is ok


def test_slow_receiver_is_cut_off_at_the_deadline(monkeypatch):
    import threading
    import time

    from src.config import get_settings

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5)  # never leak the socket if delivery does not connect
    port = server.getsockname()[1]
    stop = threading.Event()

    def trickle():  # answers one byte every 0.2 s, forever
        conn, _ = server.accept()
        conn.recv(65536)
        while not stop.wait(0.2):
            try:
                conn.sendall(b"H")
            except OSError:
                break
        conn.close()
        server.close()

    threading.Thread(target=trickle, daemon=True).start()
    monkeypatch.setattr(webhook_tasks, "_validate_webhook_url",
                        lambda url: ("127.0.0.1", "127.0.0.1"))
    monkeypatch.setattr(get_settings(), "WEBHOOK_DELIVERY_DEADLINE_SECONDS", 1.0, raising=False)
    outcome = {}

    def deliver():
        started = time.monotonic()
        try:
            webhook_tasks.deliver_webhook.run(
                f"http://127.0.0.1:{port}/x", "job.completed", {}, timeout=0.5)
        except Exception as error:  # the timeout (or Celery's retry wrapper around it)
            outcome["error"] = error
        outcome["elapsed"] = time.monotonic() - started

    worker = threading.Thread(target=deliver, daemon=True)
    worker.start()
    worker.join(5)
    stop.set()
    assert not worker.is_alive(), "delivery still running after 5 s"
    assert outcome["elapsed"] < 3
    # the deadline, not the 0.5 s per-read timeout, ended it
    assert "deadline" in str(outcome.get("error"))


@pytest.fixture
def fernet(monkeypatch):
    """A real WEBHOOK_SECRET_KEY, so stored secrets are ciphertext as in production."""
    from cryptography.fernet import Fernet

    from src.config import get_settings
    from src.utils import crypto

    monkeypatch.setattr(get_settings(), "WEBHOOK_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(crypto, "_fernet", None)
    return crypto


def _signature(request, secret):
    return "sha256=" + hmac.new(secret, request.content, hashlib.sha256).hexdigest()


def test_dispatch_queues_webhook_id_not_secret(fernet):
    from src.workers import helpers

    webhook = SimpleNamespace(id=uuid.uuid4(), url="https://hooks.example.com/x",
                              secret=fernet.encrypt_secret("s3cret"))

    class Repo:
        def __init__(self, session):
            pass

        def list_by_event(self, event, api_key_id=None):
            return [webhook]

    with patch.object(helpers, "get_sync_session", contextlib.nullcontext), \
            patch.object(helpers, "SyncWebhookRepository", Repo), \
            patch.object(webhook_tasks.deliver_webhook, "delay") as delay:
        assert helpers.dispatch_webhook_sync("job.completed", {"job_id": "j1"},
                                             api_key_id="k1") == 1
    kwargs = delay.call_args.kwargs
    assert kwargs.get("secret") is None
    assert "s3cret" not in repr(kwargs)
    assert kwargs["webhook_id"] == str(webhook.id)


def test_worker_signs_with_secret_loaded_by_id(fernet, monkeypatch):
    webhook_id = str(uuid.uuid4())
    loaded = []

    class Repo:
        def __init__(self, session):
            pass

        def get(self, wid):
            loaded.append(wid)
            return SimpleNamespace(secret=fernet.encrypt_secret("s3cret"))

    monkeypatch.setattr(webhook_tasks, "get_sync_session", contextlib.nullcontext, raising=False)
    monkeypatch.setattr(webhook_tasks, "SyncWebhookRepository", Repo, raising=False)
    [request] = _deliver("https://hooks.example.com/x", PUBLIC_V4, webhook_id=webhook_id)
    assert loaded == [uuid.UUID(webhook_id)]
    assert request.headers["X-QuickPitik-Signature"] == _signature(request, b"s3cret")


def test_message_queued_before_the_change_still_signs():
    [request] = _deliver("https://hooks.example.com/x", PUBLIC_V4, secret="s3cret")
    assert request.headers["X-QuickPitik-Signature"] == _signature(request, b"s3cret")
