"""Recreate the three COMSOL/PyMieDiff validation panels from archived CSVs.

These are reproducibility plots, not the author's final typeset Figure 8 art.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "validation"


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return {
        key: np.array([float(row[key]) if row[key] else np.nan for row in rows])
        for key in rows[0]
    }


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "reproduced" / "figures" / "comsol_validation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = read_numeric_csv(VALIDATION / "COMSOL_PyMieDiff_full_spectrum.csv")
    ten = read_numeric_csv(VALIDATION / "COMSOL_PyMieDiff_10_wavelengths.csv")
    wl, grid = full["wavelength_nm"], ten["wavelength_nm"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(wl, full["PyMieDiff_Qsca"], color="#008c95", label="PyMieDiff")
    ax.plot(grid, ten["M4_Qsca"], "s", color="#eb8a70", mfc="none", label="COMSOL M4")
    ax.plot(grid, ten["M5_Qsca"], "D", color="#547db5", mfc="none", label="COMSOL M5")
    ax.set(xlabel="Wavelength (nm)", ylabel=r"Scattering efficiency, $Q_{\mathrm{sca}}$")
    ax.legend(frameon=False)
    save(fig, args.output_dir / "Figure8a_recreated.svg")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0, color="0.55", linewidth=0.8)
    ax.plot(grid, ten["M4_minus_Py_Qsca_pct"], "s", color="#547db5", mfc="none", label="M4")
    ax.plot(grid, ten["M5_minus_Py_Qsca_pct"], "o", color="#eb8a70", mfc="none", label="M5")
    ax.set(xlabel="Wavelength (nm)", ylabel=r"Signed relative $Q_{\mathrm{sca}}$ difference (%)")
    ax.legend(frameon=False)
    save(fig, args.output_dir / "Figure8b_recreated.svg")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(
        np.arange(len(grid)),
        ten["M4_minus_M5_Qsca_abs_pct_of_M5"],
        color="#e8a277",
        width=0.65,
    )
    ax.set_xticks(np.arange(len(grid)), [str(int(value)) for value in grid])
    ax.set(
        xlabel="Wavelength (nm)",
        ylabel=r"$|Q_{\mathrm{sca}}^{\mathrm{M4}}-Q_{\mathrm{sca}}^{\mathrm{M5}}|/Q_{\mathrm{sca}}^{\mathrm{M5}}$ (%)",
    )
    save(fig, args.output_dir / "Figure8c_recreated.svg")
    print(f"Wrote three SVG panels to {args.output_dir}")


if __name__ == "__main__":
    main()
