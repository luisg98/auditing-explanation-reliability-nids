"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def effective_seed(config: dict[str, Any], default: int = 42) -> int:
    """Return the configured seed, overridden by PIPELINE_SEED when present."""
    return int(os.environ.get("PIPELINE_SEED", config.get("random_seed", default)))


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds when TensorFlow is available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
