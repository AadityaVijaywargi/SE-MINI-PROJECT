"""Domain layer: storage, profiling, target analysis, preprocessing, training, evaluation, jobs."""

import sys

if sys.platform == "win32":
    # On Windows, scikit-learn bundles an older msvcp140.dll. If it loads first, creating a
    # PennyLane lightning.qubit device crashes the interpreter, so the Lightning extension
    # is loaded before anything imports scikit-learn. Linux (Docker, CI) is unaffected.
    try:
        import pennylane_lightning.lightning_qubit_ops  # noqa: F401  pylint: disable=unused-import
    except ImportError:
        pass
