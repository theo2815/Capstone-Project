from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DESKTOP_SCOPES = '["blur:read", "jobs:read"]'


def test_gen_api_key_prints_least_privilege_desktop_scopes(capsys):
    runpy.run_path(str(ROOT / "gen_api_key.py"))
    out = capsys.readouterr().out
    assert DESKTOP_SCOPES in out and '["*"]' not in out


def test_insert_key_uses_least_privilege_desktop_scopes():
    text = (ROOT / "insert_key.py").read_text(encoding="utf-8")  # never import it: it runs docker
    assert DESKTOP_SCOPES in text and '["*"]' not in text


@pytest.mark.parametrize("url", [
    "postgresql+asyncpg://u:p@db.railway.internal:5432/quickpitik",
    "postgresql+asyncpg://u:p@10.0.0.5:5432/quickpitik",
    "postgresql+asyncpg://u:p@/quickpitik?host=/var/run/postgresql",  # no host: refuse
])
def test_seed_refuses_a_non_local_database(url):
    from scripts.seed_db import _require_local_database

    with pytest.raises(SystemExit):
        _require_local_database(url)


@pytest.mark.parametrize("url", [
    "postgresql+asyncpg://postgres:x@localhost:5433/quickpitik",
    "postgresql+asyncpg://postgres:x@127.0.0.1:5432/quickpitik",
    "postgresql+asyncpg://postgres:x@db:5432/quickpitik",
    "postgresql+asyncpg://postgres:x@[::1]:5432/quickpitik",
])
def test_seed_allows_a_local_database(url):
    from scripts.seed_db import _require_local_database

    _require_local_database(url)
