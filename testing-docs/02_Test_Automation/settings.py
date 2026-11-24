"""Paths and settings shared by conftest.py, pages.py and the tests.

Folder layout (see TESTING_README.md at the repository root):
    testing-docs/02_Test_Automation/          this framework (tests/ holds the test scripts)
    testing-docs/03_Test_Data/                TD-01 to TD-14 input files
    testing-docs/04_Test_Execution_Results/   <run>/ evidence folders, one per run (kept at this
                                              depth so screenshot paths stay under 260 characters)
"""

import os
import sys
from datetime import datetime
from pathlib import Path

AUTOMATION_DIR = Path(__file__).resolve().parent
TESTING_DOCS = AUTOMATION_DIR.parent
REPO = TESTING_DOCS.parent
TEST_DATA = TESTING_DOCS / "03_Test_Data"
RUNS_DIR = TESTING_DOCS / "04_Test_Execution_Results"
SAMPLES = REPO / "data" / "samples"
_windows_venv_python = REPO / ".venv" / "Scripts" / "python.exe"
_linux_venv_python = REPO / ".venv" / "bin" / "python"
APP_PYTHON = (
    _windows_venv_python
    if _windows_venv_python.exists()
    else _linux_venv_python
    if _linux_venv_python.exists()
    else Path(sys.executable)
)
PORT = int(os.environ.get("E2E_PORT", "5055"))
BASE_URL = f"http://127.0.0.1:{PORT}"
RUN_DIR = Path(os.environ.get("E2E_RUN_DIR") or RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S"))
SHOTS = RUN_DIR / "screenshots"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
