"""Shared model and data utilities (NumPy only, ML-framework agnostic)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


N_FEATURES = 10


def load_data(
    partition_id: int,
    n_samples: int = 300,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate synthetic binary-classification data for one node.

    Each node gets different data via ``partition_id`` as the random seed,
    so the federation is genuinely heterogeneous.
    """
    rng = np.random.default_rng(seed=partition_id)

    # Two Gaussian clusters shifted by ±1 along every feature
    X0 = rng.standard_normal((n_samples // 2, N_FEATURES)) + 1.0
    X1 = rng.standard_normal((n_samples // 2, N_FEATURES)) - 1.0
    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.float64)

    idx = rng.permutation(n_samples)
    X, y = X[idx], y[idx]

    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def get_initial_parameters() -> list[NDArray]:
    """Return zero-initialised weights and bias."""
    return [np.zeros(N_FEATURES), np.zeros(1)]


def _sigmoid(z: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train(
    parameters: list[NDArray],
    X: NDArray,
    y: NDArray,
    lr: float = 0.05,
    epochs: int = 10,
) -> tuple[list[NDArray], int, dict]:
    """One FL round of local training via gradient descent."""
    w, b = parameters[0].copy(), parameters[1].copy()
    n = len(y)
    for _ in range(epochs):
        pred = _sigmoid(X @ w + b)
        err = pred - y
        w -= lr * (X.T @ err) / n
        b -= lr * np.sum(err) / n
    return [w, b], n, {}


def evaluate(
    parameters: list[NDArray],
    X: NDArray,
    y: NDArray,
) -> tuple[float, int, dict]:
    """Return cross-entropy loss and accuracy."""
    w, b = parameters[0], parameters[1]
    pred = _sigmoid(X @ w + b)
    loss = float(
        -np.mean(y * np.log(pred + 1e-9) + (1 - y) * np.log(1 - pred + 1e-9))
    )
    acc = float(np.mean((pred >= 0.5) == y))
    return loss, len(X), {"accuracy": acc}
