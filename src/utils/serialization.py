"""Serialization utilities for JSON export.

Converts NumPy types, PyTorch/TensorFlow scalars, and arrays to native Python types.
"""

from typing import Any
import numpy as np


def convert_numpy_to_python(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and TF tensors to native Python objects.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable Python object.
    """
    if isinstance(obj, dict):
        return {convert_numpy_to_python(k): convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(elem) for elem in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "numpy"):
        return convert_numpy_to_python(obj.numpy())
    return obj
