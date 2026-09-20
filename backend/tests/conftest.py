"""
Forces mock mode + an isolated temp DB/upload dir before the app is ever
imported (Settings() is @lru_cache'd, so this must happen first) -- these
are orchestrator input-validation tests, not model-quality tests, so they
shouldn't need the GPU or trained checkpoints, and shouldn't touch the
real dev database or uploads folder. All four task modes need forcing here
(not just VQA/change/fusion) -- app/main.py's startup lifespan eager-loads
whichever services are set to "real", so leaving any one of them on
whatever a real .env happens to say would make the test suite pay that
model's full load time on every run.
"""
import os
import tempfile
from pathlib import Path

import pytest

_TMP_DIR = tempfile.mkdtemp(prefix="satquery_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DIR}/test.db"
os.environ["UPLOAD_DIR"] = str(Path(_TMP_DIR) / "uploads")
os.environ["RESULTS_DIR"] = str(Path(_TMP_DIR) / "results")
os.environ["VQA_MODE"] = "mock"
os.environ["CHANGE_MODE"] = "mock"
os.environ["FUSION_MODE"] = "mock"
os.environ["GROUNDING_MODE"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
