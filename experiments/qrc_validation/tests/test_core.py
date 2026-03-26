from __future__ import annotations

from pathlib import Path

import numpy as np

from qrc_validation.features import QRCFeatureMap
from qrc_validation.symbolic import run_sympy_checks


def test_qrc_feature_shape() -> None:
    X = np.random.default_rng(0).normal(size=(12, 8))
    fmap = QRCFeatureMap(n_features=24, eta=0.5, observable_policy="fixed_pauli_subset", seed=0)
    fmap.fit(X)
    Z = fmap.transform(X)
    assert Z.shape == (12, 24)


def test_sympy_outputs(tmp_path: Path) -> None:
    report = tmp_path / "sympy.txt"
    table = tmp_path / "table.csv"
    out = run_sympy_checks(report, table)
    assert "did_expression" in out
    assert report.exists()
    assert table.exists()
