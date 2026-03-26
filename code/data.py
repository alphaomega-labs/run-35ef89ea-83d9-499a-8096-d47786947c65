from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray


def _balanced_subset(X: np.ndarray, y: np.ndarray, max_samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        keep_idx.extend(cls_idx[:max_samples_per_class].tolist())
    keep = np.array(sorted(keep_idx))
    return X[keep], y[keep]


def _fetch_openml_candidates(candidates: list[str], version: int, data_home: Path) -> tuple[np.ndarray, np.ndarray, str]:
    last_err: Exception | None = None
    for name in candidates:
        try:
            ds = fetch_openml(
                name=name,
                version=version,
                as_frame=False,
                parser="auto",
                data_home=str(data_home),
            )
            X = np.asarray(ds.data, dtype=np.float64)
            y_raw = np.asarray(ds.target)
            _, y = np.unique(y_raw, return_inverse=True)
            return X, y.astype(int), name
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if last_err is None:
        raise RuntimeError("No OpenML dataset candidates were provided.")
    raise RuntimeError(
        f"Failed OpenML candidates {candidates}: {type(last_err).__name__}: {last_err}"
    )


def load_canonical_dataset(name: str, max_samples_per_class: int, seed: int, data_home: Path) -> DatasetBundle:
    lookup: dict[str, tuple[list[str], int]] = {
        "MNIST": (["mnist_784"], 1),
        "FASHION_MNIST": (["Fashion-MNIST", "Fashion-MNIST-784"], 1),
        "KMNIST": (["Kuzushiji-MNIST", "KMNIST"], 1),
        "EMNIST_LETTERS": (["EMNIST_Balanced", "EMNIST-Letters"], 1),
    }
    if name not in lookup:
        raise ValueError(f"Unsupported canonical dataset name: {name}")
    candidates, version = lookup[name]
    X, y, resolved = _fetch_openml_candidates(candidates, version=version, data_home=data_home)
    X, y = _balanced_subset(X, y, max_samples_per_class=max_samples_per_class, seed=seed)
    return DatasetBundle(name=f"{name}::{resolved}", X=X, y=y)


def write_dataset_manifest(path: Path, config: dict) -> None:
    data_home = Path(config["openml_data_home"])
    records: list[dict] = []
    for ds in config["datasets"]:
        bundle = load_canonical_dataset(
            ds,
            max_samples_per_class=config["max_samples_per_class"],
            seed=0,
            data_home=data_home,
        )
        records.append(
            {
                "requested_dataset": ds,
                "resolved_dataset": bundle.name,
                "n_samples_subset": int(bundle.X.shape[0]),
                "n_features": int(bundle.X.shape[1]),
                "n_classes_subset": int(np.unique(bundle.y).size),
                "sha256_subset": hashlib.sha256(bundle.X.tobytes() + bundle.y.tobytes()).hexdigest(),
            }
        )
    blob = {
        "source": "openml via sklearn.fetch_openml",
        "openml_data_home": str(data_home),
        "datasets": records,
        "max_samples_per_class": config["max_samples_per_class"],
    }
    path.write_text(json.dumps(blob, indent=2) + "\n")
