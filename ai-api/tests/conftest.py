from __future__ import annotations

import os

# Face/bib tests need the old profile until AS-8 deletes them (fix plan 2026-10-16).
os.environ.setdefault("MODEL_PROFILE", "all")

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def app():
    """Create a test FastAPI application."""
    from src.main import create_app

    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Create a test HTTP client."""
    with TestClient(app) as c:
        yield c
