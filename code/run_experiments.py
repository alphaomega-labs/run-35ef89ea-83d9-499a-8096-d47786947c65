from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrc_validation.runner import RunnerConfig, run_all


def _iter_tag_from_output(path: Path) -> str:
    for part in path.parts:
        if part.startswith("iter_"):
            return part
    return "iter_1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QRC validation experiments")
    parser.add_argument("--config", type=Path, default=Path("experiments/qrc_validation/configs/default.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/qrc_validation/iter_1"))
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    iter_tag = _iter_tag_from_output(args.output_dir)
    rc = RunnerConfig(
        root=Path("."),
        output_dir=args.output_dir,
        paper_fig_dir=Path("paper/figures") / iter_tag,
        paper_tbl_dir=Path("paper/tables") / iter_tag,
        paper_data_dir=Path("paper/data") / iter_tag,
        appendix_dir=Path("paper/appendix"),
        negative_dir=Path("experiments/negative_results"),
    )
    out = run_all(cfg, rc)
    print(f"progress: 100% completed {len(out['figure_paths'])} figures, {len(out['table_paths'])} tables")


if __name__ == "__main__":
    main()
