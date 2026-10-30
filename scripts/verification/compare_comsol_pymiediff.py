"""Check the independent COMSOL M4/M5 ten-wavelength comparison.

The script reads the frozen selected-design spectrum and the raw COMSOL table
exports. It does not require COMSOL or modify any frozen input file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "validation"
SPECTRA = (
    ROOT
    / "results"
    / "robust_design"
    / "full_factorial_robustness_evaluation"
    / "selected_design_spectra.npz"
)
WAVELENGTHS_NM = np.array([400, 406, 502, 598, 700, 748, 778, 850, 928, 1000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "reproduced" / "comsol_validation",
    )
    args = parser.parse_args()

    m4 = np.loadtxt(VALIDATION / "M4.txt", comments="%", encoding="utf-8")
    m5 = np.loadtxt(VALIDATION / "M5.txt", comments="%", encoding="utf-8")
    if m4.shape != (10, 8) or m5.shape != (10, 8):
        raise ValueError("Both COMSOL exports must have ten rows and eight columns.")
    for label, table in (("M4", m4), ("M5", m5)):
        if not np.allclose(table[:, 0], WAVELENGTHS_NM, rtol=0, atol=1e-8):
            raise ValueError(f"{label} wavelengths do not match the locked grid.")
        if not np.allclose(table[:, 0], table[:, 1], rtol=0, atol=1e-8):
            raise ValueError(f"{label} wavelength columns disagree.")
        if np.max(np.abs(table[:, 5] + table[:, 6] - table[:, 7])) > 1e-10:
            raise ValueError(f"{label} fails Qext = Qsca + Qabs closure.")

    with np.load(SPECTRA, allow_pickle=False) as archive:
        nominal_rows = np.flatnonzero(np.all(archive["perturbations_nm"] == 0, axis=1))
        if len(nominal_rows) != 1:
            raise ValueError("Expected exactly one unperturbed selected design.")
        nominal = int(nominal_rows[0])
        if not np.array_equal(archive["nominal_geometry_nm"].reshape(-1), [142, 64, 57, 21]):
            raise ValueError("The selected geometry differs from the COMSOL model.")
        materials = [str(x) for x in archive["material_names"].reshape(-1)]
        if materials != ["TiO2", "PDMS", "BaSO4", "TiO2"]:
            raise ValueError("The selected materials differ from the COMSOL model.")
        wavelengths = np.asarray(archive["wavelengths_nm"], dtype=float)
        indices = np.array([np.argmin(np.abs(wavelengths - x)) for x in WAVELENGTHS_NM])
        if np.max(np.abs(wavelengths[indices] - WAVELENGTHS_NM)) > 1e-8:
            raise ValueError("The PyMieDiff spectrum lacks a requested wavelength.")
        py = {key: np.asarray(archive[key][nominal][indices], dtype=float)
              for key in ("qsca", "qabs", "qext")}

    rows = []
    for i, wavelength in enumerate(WAVELENGTHS_NM):
        rows.append({
            "wavelength_nm": int(wavelength),
            "PyMieDiff_Qsca": float(py["qsca"][i]),
            "PyMieDiff_Qabs": float(py["qabs"][i]),
            "PyMieDiff_Qext": float(py["qext"][i]),
            "M4_Qsca": float(m4[i, 5]),
            "M4_Qabs": float(m4[i, 6]),
            "M4_Qext": float(m4[i, 7]),
            "M5_Qsca": float(m5[i, 5]),
            "M5_Qabs": float(m5[i, 6]),
            "M5_Qext": float(m5[i, 7]),
            "M4_minus_Py_Qsca_pct": float(100 * (m4[i, 5] - py["qsca"][i]) / py["qsca"][i]),
            "M5_minus_Py_Qsca_pct": float(100 * (m5[i, 5] - py["qsca"][i]) / py["qsca"][i]),
            "M4_minus_M5_Qsca_abs_pct_of_M5": float(100 * abs(m4[i, 5] - m5[i, 5]) / m5[i, 5]),
        })

    with (VALIDATION / "COMSOL_PyMieDiff_10_wavelengths.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        published = list(csv.DictReader(handle))
    if len(published) != len(rows):
        raise ValueError("The frozen comparison CSV is not ten rows long.")
    for current, frozen in zip(rows, published, strict=True):
        for key, value in current.items():
            if not np.isclose(value, float(frozen[key]), rtol=1e-10, atol=1e-12):
                raise ValueError(f"Frozen comparison mismatch at {current['wavelength_nm']} nm: {key}")

    m4_delta = m4[:, 5] - py["qsca"]
    m5_delta = m5[:, 5] - py["qsca"]
    metrics = {
        "M4_Qsca_MAE": float(np.mean(np.abs(m4_delta))),
        "M4_Qsca_RMSE": float(np.sqrt(np.mean(m4_delta**2))),
        "M4_max_abs_relative_pct": float(np.max(np.abs(100 * m4_delta / py["qsca"]))),
        "M5_Qsca_MAE": float(np.mean(np.abs(m5_delta))),
        "M5_Qsca_RMSE": float(np.sqrt(np.mean(m5_delta**2))),
        "M5_max_abs_relative_pct": float(np.max(np.abs(100 * m5_delta / py["qsca"]))),
        "M4_M5_max_abs_relative_pct_of_M5": float(
            np.max(100 * np.abs(m4[:, 5] - m5[:, 5]) / m5[:, 5])
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "COMSOL_PyMieDiff_recomputed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
