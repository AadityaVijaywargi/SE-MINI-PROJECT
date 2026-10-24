"""CUDA detection and device selection (core/gpu.py)."""

import pytest

import config
from core import gpu
from core.trainers import build_trainers, xgboost_classifier


@pytest.fixture(autouse=True)
def fresh_detection():
    """Each test sees its own driver lookup instead of the cached answer."""
    gpu.cuda_device_count.cache_clear()
    yield
    gpu.cuda_device_count.cache_clear()


class FakeDriver:
    """Stands in for the CUDA driver library."""

    def __init__(self, init_status=0, count=1):
        self.init_status = init_status
        self.count = count

    def cuInit(self, _flags):  # pylint: disable=invalid-name
        return self.init_status

    def cuDeviceGetCount(self, pointer):  # pylint: disable=invalid-name
        pointer._obj.value = self.count  # pylint: disable=protected-access
        return 0


def test_no_driver_means_no_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "load_driver", lambda: None)
    assert gpu.cuda_device_count() == 0
    assert not gpu.use_cuda()


def test_driver_that_cannot_start_means_no_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "load_driver", lambda: FakeDriver(init_status=999))
    assert gpu.cuda_device_count() == 0


def test_working_driver_reports_its_gpus(monkeypatch):
    monkeypatch.setattr(gpu, "load_driver", lambda: FakeDriver(count=2))
    monkeypatch.setattr(config, "COMPUTE_DEVICE", "auto")
    assert gpu.cuda_device_count() == 2
    assert gpu.use_cuda()


def test_cpu_setting_ignores_the_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "load_driver", lambda: FakeDriver(count=1))
    monkeypatch.setattr(config, "COMPUTE_DEVICE", "cpu")
    assert not gpu.use_cuda()


@pytest.mark.parametrize("cuda, device", [(True, "cuda"), (False, "cpu")])
def test_xgboost_follows_the_detected_device(monkeypatch, cuda, device):
    monkeypatch.setattr(gpu, "use_cuda", lambda: cuda)
    assert xgboost_classifier(42).get_params()["device"] == device


def test_ml_models_report_their_device(monkeypatch, banking):
    from core.preprocessing import Preprocessor  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(gpu, "use_cuda", lambda: False)
    split = Preprocessor().fit_split(banking, "Exited")
    for trainer in build_trainers(["naive_bayes", "xgboost"], seed=42):
        trainer.train(split)
        assert trainer.details["Device"] == "CPU"
