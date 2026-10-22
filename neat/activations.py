from collections.abc import Callable

import numpy as np

ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))),
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0.0, x),
    "sin": np.sin,
    "gauss": lambda x: np.exp(-np.clip(x, -60, 60) ** 2),
    "abs": np.abs,
    "step": lambda x: (x > 0).astype(float),
}
