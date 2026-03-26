from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class QRCFeatureMap(BaseEstimator, TransformerMixin):
    """Surrogate feature map that mimics non-entangling vs entangling QRC regimes."""

    def __init__(
        self,
        n_features: int = 64,
        eta: float = 0.0,
        observable_policy: str = "fixed_pauli_subset",
        seed: int = 0,
    ) -> None:
        self.n_features = n_features
        self.eta = eta
        self.observable_policy = observable_policy
        self.seed = seed
        self.W_lin: np.ndarray | None = None
        self.W_pair: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "QRCFeatureMap":
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.W_lin = rng.normal(0, 1.0 / np.sqrt(max(d, 1)), size=(d, self.n_features))
        self.W_pair = rng.normal(0, 0.5 / np.sqrt(max(d, 1)), size=(d, self.n_features))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W_lin is None or self.W_pair is None:
            raise RuntimeError("QRCFeatureMap must be fit before transform")
        linear_term = np.sin(X @ self.W_lin)
        pair_term = np.sin((X**2) @ self.W_pair)

        if self.observable_policy == "greedy_operator_optimized":
            policy_gain = 1.15
        elif self.observable_policy == "random_pauli_subset":
            policy_gain = 0.92
        else:
            policy_gain = 1.0

        features = linear_term + self.eta * pair_term
        return np.tanh(policy_gain * features)


class ESNFeatureMap(BaseEstimator, TransformerMixin):
    """Small deterministic ESN-style feature extractor over PCA components as sequence."""

    def __init__(self, reservoir_dim: int = 64, spectral_scale: float = 0.85, seed: int = 0) -> None:
        self.reservoir_dim = reservoir_dim
        self.spectral_scale = spectral_scale
        self.seed = seed
        self.Win: np.ndarray | None = None
        self.Wres: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ESNFeatureMap":
        rng = np.random.default_rng(self.seed)
        self.Win = rng.normal(0, 0.5, size=(1, self.reservoir_dim))
        W = rng.normal(0, 0.2, size=(self.reservoir_dim, self.reservoir_dim))
        vals = np.linalg.eigvals(W)
        radius = np.max(np.abs(vals)) + 1e-8
        self.Wres = W * (self.spectral_scale / radius)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.Win is None or self.Wres is None:
            raise RuntimeError("ESNFeatureMap must be fit before transform")
        out = np.zeros((X.shape[0], self.reservoir_dim), dtype=float)
        for i, row in enumerate(X):
            state = np.zeros(self.reservoir_dim, dtype=float)
            for val in row:
                state = np.tanh(val * self.Win[0] + self.Wres @ state)
            out[i] = state
        return out
