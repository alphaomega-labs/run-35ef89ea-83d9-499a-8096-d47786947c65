# QRC Validation Experiments

This package executes the `validation_simulation` phase for the QRC open-question study.

## Goal
Run comparator-parity and mechanism-focused experiments for three manuscript claims: rank-conditioned parity advantage, entanglement-observable interaction, and non-entangling kernel-null behavior, plus channel-transfer stress with full per-channel retraining and symbolic checks tied to `phase_outputs/SYMPY.md`.

## Layout
- `run_experiments.py`: CLI entrypoint.
- `src/qrc_validation/`: reusable simulation and analysis modules.
- `configs/default.json`: reproducible run configuration.
- `tests/test_core.py`: smoke and symbolic-check tests.
- `iter_1/`: iteration-specific outputs (`results_summary.json`, manifest, SymPy report, preview PNGs).

## Commands
From workspace root:

```bash
python -m venv experiments/.venv
experiments/.venv/bin/python -m pip install -U pip
uv pip install --python experiments/.venv/bin/python numpy pandas scipy scikit-learn matplotlib seaborn sympy pytest ruff mypy jinja2
PYTHONPATH=experiments/qrc_validation/src experiments/.venv/bin/python experiments/qrc_validation/run_experiments.py --config experiments/qrc_validation/configs/default.json --output-dir experiments/qrc_validation/iter_1
PYTHONPATH=experiments/qrc_validation/src experiments/.venv/bin/ruff check experiments/qrc_validation
PYTHONPATH=experiments/qrc_validation/src experiments/.venv/bin/mypy --ignore-missing-imports experiments/qrc_validation/src
PYTHONPATH=experiments/qrc_validation/src experiments/.venv/bin/pytest -q experiments/qrc_validation/tests
```

## Notes
- Canonical datasets are fetched via `sklearn.fetch_openml` and cached under `experiments/.cache/openml`.
- Figures are exported as PDF vector graphics for LaTeX compatibility and rasterized to PNG for readability checks.
- Negative and contradictory slices are persisted to `experiments/negative_results/*.jsonl` and corresponding appendix audit files under `paper/appendix/`.
