"""Runtime detection and GPU configuration utilities.

Provides helper functions to:
- Detect whether TensorFlow sees a GPU
- Display runtime type (CPU, GPU-Colab, GPU-local)
- Configure GPU memory growth when a GPU is available
- Work without error on CPU-only machines

Mixed precision is NOT enabled here — it will be studied during
the training phase.

Author: Yassine
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def detect_gpu() -> list[str]:
    """Return a list of GPU device names visible to TensorFlow.

    Returns an empty list when no GPU is found or TensorFlow is not
    installed.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return [gpu.name for gpu in gpus]
    except ImportError:
        logger.info("TensorFlow is not installed — GPU detection skipped.")
        return []
    except Exception as exc:
        logger.warning("GPU detection failed: %s", exc)
        return []


def configure_gpu_memory_growth() -> None:
    """Enable memory growth on all visible GPUs.

    By default TensorFlow allocates ALL GPU memory upfront. Memory
    growth tells it to allocate only as much as needed, which is
    essential in Colab where the GPU is shared.

    Does nothing if no GPU is found or TensorFlow is not installed.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth enabled for %s", gpu.name)
        if not gpus:
            logger.info("No GPU found — memory growth configuration skipped.")
    except ImportError:
        logger.info("TensorFlow is not installed — skipping GPU config.")
    except RuntimeError as exc:
        # Memory growth must be set before GPUs are initialized.
        logger.warning("Could not set memory growth: %s", exc)


def get_runtime_info() -> dict[str, str | bool]:
    """Return a dictionary describing the current runtime environment.

    Keys:
        - ``runtime_type``: ``"colab_gpu"``, ``"colab_cpu"``,
          ``"local_gpu"``, or ``"local_cpu"``.
        - ``gpu_available``: Boolean.
        - ``gpu_devices``: Comma-separated device names.
        - ``tf_version``: TensorFlow version or ``"not installed"``.
    """
    info: dict[str, str | bool] = {}

    # Check if running inside Google Colab.
    in_colab = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

    # TensorFlow version.
    try:
        import tensorflow as tf
        info["tf_version"] = tf.__version__
    except ImportError:
        info["tf_version"] = "not installed"

    gpus = detect_gpu()
    info["gpu_available"] = len(gpus) > 0
    info["gpu_devices"] = ", ".join(gpus) if gpus else "none"

    if in_colab:
        info["runtime_type"] = "colab_gpu" if gpus else "colab_cpu"
    else:
        info["runtime_type"] = "local_gpu" if gpus else "local_cpu"

    return info


def print_runtime_info() -> None:
    """Print a human-readable summary of the runtime environment."""
    info = get_runtime_info()
    print("=" * 50)
    print("RUNTIME INFORMATION")
    print("=" * 50)
    print(f"  Runtime type  : {info['runtime_type']}")
    print(f"  TF version    : {info['tf_version']}")
    print(f"  GPU available : {info['gpu_available']}")
    print(f"  GPU devices   : {info['gpu_devices']}")
    print("=" * 50)
