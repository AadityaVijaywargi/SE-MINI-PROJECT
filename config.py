"""Run settings. Every value can be overridden by an environment variable (CON-6).

A local ``.env`` file is read first, if present; ``.env.example`` lists every variable.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def env_int(name, default):
    """Integer environment variable, or ``default`` when it is unset."""
    return int(os.environ.get(name, str(default)))


# "production" restricts CORS to CORS_ORIGINS and hides unexpected job errors (SEC-3, SEC-5).
APP_ENV = os.environ.get("APP_ENV", "development")
PRODUCTION = APP_ENV == "production"
CORS_ORIGINS = [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "").split(",")]
CORS_ORIGINS = [origin for origin in CORS_ORIGINS if origin]

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
SAMPLES_DIR = BASE_DIR / "data" / "samples"
MAX_UPLOAD_MB = env_int("MAX_UPLOAD_MB", 100)
ALLOWED_EXTENSIONS = (".csv", ".xlsx")
MIN_ROWS = 50
MIN_COLUMNS = 2
# SEC-7: uploads older than this are deleted, and the folder is kept under a size cap.
UPLOAD_TTL_SECONDS = env_int("UPLOAD_TTL_SECONDS", 3600)
UPLOAD_DIR_MAX_MB = env_int("UPLOAD_DIR_MAX_MB", 500)

# "auto" trains XGBoost and the DNN on an NVIDIA GPU when CUDA is usable and on the CPU
# otherwise; "cpu" never uses the GPU (core/gpu.py, docs/deviations.md).
COMPUTE_DEVICE = os.environ.get("COMPUTE_DEVICE", "auto").strip().lower()

DEFAULT_SEED = 42
DEFAULT_TEST_RATIO = 0.2
ALLOWED_TEST_RATIOS = (0.1, 0.2, 0.3)

# REQ-JB-7: finished jobs are forgotten after this many seconds.
JOB_TTL_SECONDS = env_int("JOB_TTL_SECONDS", 3600)

# REQ-TR-2, REQ-MS-4: SVM trains on a stratified sample above this many training rows.
SVM_MAX_ROWS = env_int("SVM_MAX_ROWS", 20000)

# REQ-TR-3, REQ-TR-4, REQ-MS-5: the deep neural network. Early stopping usually ends
# training long before the epoch limit.
DNN_EPOCHS = env_int("DNN_EPOCHS", 100)
DNN_MAX_EPOCHS = 500
DNN_BATCH_SIZE = 32

# REQ-TR-5, REQ-TR-6, CON-4, CON-5: the variational quantum classifier.
QML_MAX_ROWS = env_int("QML_MAX_ROWS", 2000)
QML_EPOCHS = env_int("QML_EPOCHS", 20)
QML_MAX_EPOCHS = 100
QML_QUBITS = env_int("QML_QUBITS", 4)
QML_MIN_QUBITS = 2
QML_MAX_QUBITS = 8
QML_LAYERS = 2
QML_BATCH_SIZE = 32
QML_LEARNING_RATE = 0.05
# PennyLane simulator. default.qubit simulates a whole batch of rows at once and is about
# 5x faster than lightning.qubit for 4 qubits, which keeps a default run inside PERF-4's
# 1 to 3 minutes; set QML_DEVICE=lightning.qubit to use Lightning (docs/deviations.md).
# lightning.gpu (Linux, requirements-gpu.txt) simulates on a CUDA GPU, but at 2 to 8
# qubits copying to the GPU costs more than it saves, so it is never picked automatically.
QML_DEVICE = os.environ.get("QML_DEVICE", "default.qubit")

PORT = env_int("PORT", 5000)
