"""Paths and settings shared by conftest.py, pages.py and the tests."""

import os
from datetime import datetime
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[2]
TEST_DATA = TESTS_DIR.parent / "test-data"
SAMPLES = REPO / "data" / "samples"
APP_PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
PORT = int(os.environ.get("E2E_PORT", "5055"))
BASE_URL = f"http://127.0.0.1:{PORT}"
RUN_DIR = Path(os.environ.get("E2E_RUN_DIR") or TESTS_DIR / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S"))
SHOTS = RUN_DIR / "screenshots"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
