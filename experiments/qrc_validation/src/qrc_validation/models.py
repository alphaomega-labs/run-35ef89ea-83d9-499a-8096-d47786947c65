from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ESNFeatureMap, QRCFeatureMap


def _probs_from_model(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    if probs.shape[1] != n_classes:
        out = np.full((X.shape[0], n_classes), 1.0 / n_classes)
        out[:, : probs.shape[1]] = probs
        return out
    return probs


def train_and_predict_proba(model_key: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int, eta: float = 0.0, observable_policy: str = "fixed_pauli_subset") -> np.ndarray:
    n_classes = int(np.unique(y_train).size)

    if model_key == "qrc_ent":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("qrc", QRCFeatureMap(n_features=96, eta=max(eta, 0.5), observable_policy=observable_policy, seed=seed)),
            ("clf", LogisticRegression(max_iter=250, random_state=seed)),
        ])
    elif model_key == "qrc_nonent":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("qrc", QRCFeatureMap(n_features=96, eta=0.0, observable_policy=observable_policy, seed=seed)),
            ("clf", LogisticRegression(max_iter=250, random_state=seed)),
        ])
    elif model_key == "esn":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("esn", ESNFeatureMap(reservoir_dim=96, seed=seed)),
            ("clf", LogisticRegression(max_iter=250, random_state=seed)),
        ])
    elif model_key == "random_feature_reservoir":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("rff", RBFSampler(gamma=0.3, n_components=128, random_state=seed)),
            ("clf", LogisticRegression(max_iter=250, random_state=seed)),
        ])
    elif model_key == "rbf_kernel":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("nys", Nystroem(gamma=0.25, n_components=128, random_state=seed)),
            ("clf", LogisticRegression(max_iter=250, random_state=seed)),
        ])
    elif model_key == "rff_ridge":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("rff", RBFSampler(gamma=0.5, n_components=192, random_state=seed)),
            ("clf", LogisticRegression(max_iter=250, random_state=seed)),
        ])
    elif model_key == "logistic_pca":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=250, random_state=seed)),
        ])
    elif model_key == "mlp":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, learning_rate_init=1e-3, max_iter=60, random_state=seed)),
        ])
    else:
        raise ValueError(f"Unsupported model_key={model_key}")

    model.fit(X_train, y_train)
    return _probs_from_model(model, X_test, n_classes=n_classes)
