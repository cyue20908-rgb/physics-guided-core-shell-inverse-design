"""Verify the raw M4/M5 exports reproduce the archived solver comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_comsol_tables_reproduce_archived_metrics(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verification" / "compare_comsol_pymiediff.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads((tmp_path / "comparison_metrics.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "validation" / "validation_data.json").read_text(encoding="utf-8"))[
        "metrics"
    ]
    mapping = {
        "M4_Qsca_MAE": "M4_Qsca_MAE",
        "M4_Qsca_RMSE": "M4_Qsca_RMSE",
        "M4_max_abs_relative_pct": "M4_Qsca_max_abs_relative_pct",
        "M5_Qsca_MAE": "M5_Qsca_MAE",
        "M5_Qsca_RMSE": "M5_Qsca_RMSE",
        "M5_max_abs_relative_pct": "M5_Qsca_max_abs_relative_pct",
        "M4_M5_max_abs_relative_pct_of_M5": "M4_M5_Qsca_max_abs_relative_pct_of_M5",
    }
    for current_key, frozen_key in mapping.items():
        assert np.isclose(actual[current_key], frozen[frozen_key], rtol=1e-11, atol=1e-12)
