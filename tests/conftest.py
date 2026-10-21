"""Shared pytest fixtures (SM-87): the five bundled sample datasets and a Flask client."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES_DIR = ROOT / "data" / "samples"
DOMAIN_SAMPLES = {
    "banking": "Exited",
    "healthcare": "Diagnosis",
    "telecom": "Churn",
    "insurance": "Fraud",
}


def load_sample(name):
    """Read one bundled sample CSV as a DataFrame."""
    return pd.read_csv(SAMPLES_DIR / f"{name}.csv")


@pytest.fixture(params=sorted(DOMAIN_SAMPLES))
def domain_sample(request):
    """Each of the four domain datasets with its target column: ``(name, frame, target)``."""
    return request.param, load_sample(request.param), DOMAIN_SAMPLES[request.param]


@pytest.fixture
def banking():
    """Banking churn data: missing values, an ID column, a binary 0/1 target."""
    return load_sample("banking")


@pytest.fixture
def three_class():
    """Customer segments with three classes."""
    return load_sample("three_class")


@pytest.fixture
def client(tmp_path):
    """Flask test client with uploads written to a temporary folder."""
    from app import create_app

    app = create_app({"UPLOAD_DIR": tmp_path / "uploads", "TESTING": True})
    return app.test_client()
