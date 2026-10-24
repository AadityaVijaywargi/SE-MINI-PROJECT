"""CUDA detection shared by the trainers.

Asks the NVIDIA driver directly (through ctypes), so the check needs no extra package and
simply answers "no GPU" on a machine without the driver, such as CI, Docker or Render.
``COMPUTE_DEVICE=cpu`` keeps every model on the CPU even when a GPU is present.
"""

import ctypes
import ctypes.util
import functools
import logging
import sys

import config

logger = logging.getLogger(__name__)


def load_driver():
    """The CUDA driver library, or None when it is not installed."""
    if sys.platform == "win32":
        name = "nvcuda.dll"
    else:
        name = ctypes.util.find_library("cuda") or "libcuda.so.1"
    try:
        return ctypes.CDLL(name)
    except OSError:
        return None


@functools.lru_cache(maxsize=None)
def cuda_device_count():
    """Number of usable CUDA GPUs; 0 when there is no driver or the GPU cannot start."""
    driver = load_driver()
    if driver is None:
        return 0
    count = ctypes.c_int(0)
    try:
        if driver.cuInit(0) != 0 or driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
            logger.warning("CUDA driver found but no GPU could be started; using the CPU")
            return 0
    except AttributeError:
        return 0
    return count.value


def use_cuda():
    """True when the models should train on the GPU (COMPUTE_DEVICE is not ``cpu``)."""
    return config.COMPUTE_DEVICE != "cpu" and cuda_device_count() > 0


def device_label(cuda):
    """The ``Device`` value shown in a model's details."""
    return "GPU (CUDA)" if cuda else "CPU"
