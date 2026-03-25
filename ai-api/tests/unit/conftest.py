"""Unit-test conftest that stubs heavy third-party dependencies.

Celery workers depend on packages that may not be installed in the lightweight
CI / local-dev environment used only for unit tests (sqlalchemy, celery,
structlog, pydantic_settings, pgvector, cv2, etc.).

We insert lightweight mocks into ``sys.modules`` **only when the real package
is not importable**, so that every ``from X import Y`` in the production modules
resolves without error.

This file is auto-loaded by pytest before any ``tests/unit/*.py`` module.
"""
from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock


def _needs_stub(name: str) -> bool:
    """Return True if *name* is not importable (i.e. we need to stub it)."""
    if name in sys.modules:
        return False
    try:
        importlib.import_module(name)
        return False
    except Exception:
        return True


def _make_module(name: str, attrs: dict | None = None) -> types.ModuleType:
    """Create a module stub and register it in *sys.modules*."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    # Allow arbitrary attribute access by default.
    mod.__getattr__ = lambda self_name: MagicMock()  # noqa: ARG005
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


def _install_stubs() -> None:  # noqa: C901
    """Pre-register stubs for every missing third-party package."""

    # ------------------------------------------------------------------
    # pgvector
    # ------------------------------------------------------------------
    if _needs_stub("pgvector"):
        _make_module("pgvector")
        _make_module("pgvector.sqlalchemy", {"Vector": MagicMock()})

    # ------------------------------------------------------------------
    # sqlalchemy
    # ------------------------------------------------------------------
    if _needs_stub("sqlalchemy"):
        _make_module("sqlalchemy", {
            "select": MagicMock(),
            "create_engine": MagicMock(),
            "cast": MagicMock(),
            "DateTime": MagicMock(),
            "Float": MagicMock(),
            "Integer": MagicMock(),
            "String": MagicMock(),
            "Text": MagicMock(),
            "Boolean": MagicMock(),
            "ForeignKey": MagicMock(),
            "func": MagicMock(),
        })
        _make_module("sqlalchemy.orm", {
            "DeclarativeBase": type("DeclarativeBase", (), {}),
            "Mapped": MagicMock(),
            "mapped_column": MagicMock(return_value=MagicMock()),
            "relationship": MagicMock(return_value=MagicMock()),
            "Session": MagicMock(),
            "sessionmaker": MagicMock(),
        })
        _make_module("sqlalchemy.dialects")
        _make_module("sqlalchemy.dialects.postgresql", {
            "UUID": MagicMock(),
            "JSONB": MagicMock(),
        })

    # ------------------------------------------------------------------
    # celery — the @celery_app.task decorator must be a pass-through
    # so the underlying function can still be called directly in tests.
    # ------------------------------------------------------------------
    if _needs_stub("celery"):
        mock_celery_app = MagicMock()
        mock_celery_app.task = lambda *a, **kw: (lambda fn: fn)
        mock_celery_app.autodiscover_tasks = MagicMock()
        mock_celery_app.conf = MagicMock()

        _make_module("celery", {"Celery": MagicMock(return_value=mock_celery_app)})
        _make_module("celery.signals", {
            "worker_process_init": MagicMock(connect=lambda fn=None, **kw: fn),
            "worker_process_shutdown": MagicMock(connect=lambda fn=None, **kw: fn),
        })

    # ------------------------------------------------------------------
    # structlog (src.utils.logging)
    # ------------------------------------------------------------------
    if _needs_stub("structlog"):
        mock_logger = MagicMock()
        _make_module("structlog", {
            "get_logger": MagicMock(return_value=mock_logger),
            "configure": MagicMock(),
        })
        _make_module("structlog.contextvars")
        _make_module("structlog.processors")
        _make_module("structlog.dev")

    # ------------------------------------------------------------------
    # pydantic_settings (src.config.Settings)
    # ------------------------------------------------------------------
    if _needs_stub("pydantic_settings"):
        _make_module("pydantic_settings", {
            "BaseSettings": type("BaseSettings", (), {"model_config": {}}),
        })

    # ------------------------------------------------------------------
    # cv2 (src.workers.helpers)
    # ------------------------------------------------------------------
    if _needs_stub("cv2"):
        _make_module("cv2", {
            "imdecode": MagicMock(),
            "IMREAD_COLOR": 1,
        })


# Run stubs immediately when conftest is loaded (before test collection of
# modules that import application code).
_install_stubs()
