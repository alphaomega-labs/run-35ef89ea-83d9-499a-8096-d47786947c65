from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def _safe_probs(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    p = _safe_probs(probs)
    conf = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if not np.any(mask):
            continue
        ece += np.abs(np.mean(acc[mask]) - np.mean(conf[mask])) * (np.sum(mask) / len(y_true))
    return float(ece)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    p = _safe_probs(probs)
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((p - y_onehot) ** 2, axis=1)))


def evaluate_metrics(y_true: np.ndarray, probs: np.ndarray, runtime_seconds: float) -> dict:
    p = _safe_probs(probs)
    pred = np.argmax(p, axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "negative_log_likelihood": float(log_loss(y_true, p)),
        "ece_15bin": float(ece_score(y_true, p, n_bins=15)),
        "brier_score": float(multiclass_brier(y_true, p, n_classes=p.shape[1])),
        "runtime_seconds": float(runtime_seconds),
    }
    return out


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 300, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, values.size, size=values.size)
        boots.append(float(np.mean(values[idx])))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi
