"""Run settings. Every value can be overridden by an environment variable (CON-6)."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
ALLOWED_EXTENSIONS = (".csv", ".xlsx")
MIN_ROWS = 50
MIN_COLUMNS = 2

DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.2
ALLOWED_TEST_RATIOS = (0.1, 0.2, 0.3)

PORT = int(os.environ.get("PORT", "5000"))
