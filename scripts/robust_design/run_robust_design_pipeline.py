# -*- coding: utf-8 -*-
"""Run the complete goal-free robust-design workflow.

The runner executes calibration, screening, refinement, and finalization with
the current Python interpreter. Passed tasks are skipped unless --overwrite is
supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from core_shell_optical_design.robust_design.robust_design_core import RESULT_ROOT

CODE_DIR = Path(__file__).resolve().parent
STAGES = (
    (
        "calibration",
        CODE_DIR / "calibrate_absorption_constraints.py",
        RESULT_ROOT / "absorption_constraint_calibration" / "absorption_calibration_run_summary.json",
    ),
    (
        "screening",
        CODE_DIR / "screen_robust_material_candidates.py",
        RESULT_ROOT / "material_candidate_screening" / "material_screening_run_summary.json",
    ),
    (
        "refinement",
        CODE_DIR / "refine_robust_geometries.py",
        RESULT_ROOT / "continuous_geometry_refinement" / "robust_refinement_run_summary.json",
    ),
    (
        "finalization",
        CODE_DIR / "finalize_robust_integer_design.py",
        RESULT_ROOT / "full_factorial_robustness_evaluation" / "robust_design_run_summary.json",
    ),
)


def passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle).get("status") == "passed"
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    task_names = tuple(item[0] for item in STAGES)
    parser.add_argument("--start", choices=task_names, default="calibration")
    parser.add_argument("--stop", choices=task_names, default="finalization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined.")
    labels = [item[0] for item in STAGES]
    start = labels.index(args.start)
    stop = labels.index(args.stop)
    if start > stop:
        raise ValueError("--start must not be after --stop.")

    for label, script, summary in STAGES[start : stop + 1]:
        if passed(summary) and not args.overwrite:
            print(f"{label}: already PASSED; skipping {summary}")
            continue
        command = [sys.executable, str(script), "--device", args.device]
        if args.overwrite:
            command.append("--overwrite")
        elif args.resume:
            command.append("--resume")
        print("=" * 96)
        print(f"Running {label}: {' '.join(command)}")
        result = subprocess.run(command, cwd=str(CODE_DIR.parent), check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        if not passed(summary):
            raise RuntimeError(f"{label} exited without a passed summary: {summary}")

    print("=" * 96)
    print("Goal-free robust-design pipeline PASSED")
    print(
        "Final structure: "
        f"{RESULT_ROOT / 'full_factorial_robustness_evaluation' / 'selected_robust_design.json'}"
    )


if __name__ == "__main__":
    main()
