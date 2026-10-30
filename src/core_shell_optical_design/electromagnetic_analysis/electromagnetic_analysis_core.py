# -*- coding: utf-8 -*-
"""Shared utilities for post-hoc electromagnetic and solar analyses.

The code reads only the frozen robust design and the original material
dispersion table. It does not modify frozen design artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator
import numpy as np

from core_shell_optical_design.project_paths import (
    DATASET_PATH,
    FROZEN_ROBUST_RESULTS_DIR,
    REFERENCE_DATA_DIR,
    REPOSITORY_ROOT,
    REPRODUCED_POSTHOC_RESULTS_DIR,
)


PROJECT_ROOT = REPOSITORY_ROOT

DEFAULT_DATASET = DATASET_PATH
DEFAULT_FINAL_DIR = FROZEN_ROBUST_RESULTS_DIR
DEFAULT_FINAL_JSON = DEFAULT_FINAL_DIR / "selected_robust_design.json"
DEFAULT_FINAL_FREEZE = DEFAULT_FINAL_DIR / "selected_design_freeze_record.json"
DEFAULT_FINAL_SPECTRA = DEFAULT_FINAL_DIR / "selected_design_spectra.npz"

DEFAULT_OUTPUT_ROOT = REPRODUCED_POSTHOC_RESULTS_DIR

DPI = 600
BACKEND = "pena"
PRECISION = "double"
NEGATIVE_TOL = 1.0e-7
ENERGY_TOL = 1.0e-7
SINGULAR_TOL = 1.0e-10


def require_times_new_roman() -> None:
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    if "Times New Roman" not in names:
        raise RuntimeError(
            "Times New Roman is not available to Matplotlib. "
            "Install/enable the font before generating manuscript figures."
        )


def setup_publication_style() -> None:
    require_times_new_roman()
    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "font.size": 11.0,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.6,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.55,
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.width": 1.35,
            "ytick.major.width": 1.35,
            "xtick.minor.width": 0.95,
            "ytick.minor.width": 0.95,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "legend.fontsize": 10.0,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "legend.framealpha": 0.98,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.unicode_minus": False,
        }
    )


def style_axis(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    x_major: float | None = None,
    y_major: float | None = None,
    minor: bool = True,
    grid: bool = False,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight="bold")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.55)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
        labeltop=False,
        labelright=False,
        width=1.35,
        length=5.0,
        colors="black",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
        width=0.95,
        length=2.5,
        colors="black",
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
    if y_major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))

    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color("black")

    if grid:
        ax.grid(True, ls="--", lw=0.55, color="0.82", alpha=0.55)
        ax.set_axisbelow(True)


def style_legend(legend) -> None:
    if legend is None:
        return
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.9)
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_color("black")


def save_figure(fig, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, complex):
            return {"real": float(value.real), "imag": float(value.imag)}
        raise TypeError(type(value).__name__)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=default)


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npz_first(data: Any, names: Sequence[str]):
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"None of these NPZ keys were found: {names}")


def load_dataset(path: Path = DEFAULT_DATASET) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as data:
        wavelengths = np.asarray(
            npz_first(data, ("wavelengths", "wavelengths_nm", "wavelength_nm", "wl_nm")),
            dtype=np.float64,
        )
        material_names = np.asarray(
            npz_first(data, ("material_names", "materials", "material_list"))
        ).astype(str)
        material_n = np.asarray(
            npz_first(data, ("material_n", "n", "n_data")),
            dtype=np.float64,
        )
        material_k = np.asarray(
            npz_first(data, ("material_k", "k", "k_data")),
            dtype=np.float64,
        )

    n_materials = len(material_names)
    if material_n.shape[0] == len(wavelengths) and material_n.shape[1] == n_materials:
        material_n = material_n.T
    if material_k.shape[0] == len(wavelengths) and material_k.shape[1] == n_materials:
        material_k = material_k.T

    if material_n.shape != material_k.shape:
        raise ValueError("material_n and material_k shapes differ.")
    if material_n.shape != (n_materials, len(wavelengths)):
        raise ValueError(
            f"Unexpected material-table shape: {material_n.shape}; "
            f"expected {(n_materials, len(wavelengths))}."
        )
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("Dataset wavelength grid is not strictly increasing.")

    return {
        "wavelengths_nm": wavelengths,
        "material_names": material_names,
        "material_n": material_n,
        "material_k": material_k,
    }


def load_final_design(
    final_json: Path = DEFAULT_FINAL_JSON,
    final_spectra: Path = DEFAULT_FINAL_SPECTRA,
    dataset_path: Path = DEFAULT_DATASET,
) -> Dict[str, Any]:
    for path in (final_json, final_spectra, dataset_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with open(final_json, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    structure = summary.get("structure")
    if not isinstance(structure, dict):
        raise KeyError("selected_robust_design.json has no structure object.")

    material_indices = np.asarray(structure["material_indices"], dtype=np.int64)
    material_names = np.asarray(structure["material_names_core_to_outer"], dtype=str)
    geometry = np.asarray(
        [
            structure["r_core_nm"],
            structure["t1_nm"],
            structure["t2_nm"],
            structure["t3_nm"],
        ],
        dtype=np.float64,
    )
    radii = np.asarray(structure["interface_radii_nm"], dtype=np.float64)

    if material_indices.shape != (4,) or material_names.shape != (4,):
        raise ValueError("Expected four layers in Stage-08 structure.")
    if geometry.shape != (4,) or radii.shape != (4,):
        raise ValueError("Expected four geometry variables and four interface radii.")
    if np.any(geometry <= 0.0):
        raise ValueError("Geometry contains non-positive values.")
    if not np.allclose(np.cumsum(geometry), radii, atol=1e-12, rtol=0.0):
        raise ValueError("Stage-08 geometry and interface radii are inconsistent.")

    dataset = load_dataset(dataset_path)
    if material_indices.min() < 0 or material_indices.max() >= len(dataset["material_names"]):
        raise ValueError("Final design has an invalid material index.")
    expected_names = dataset["material_names"][material_indices]
    if not np.array_equal(expected_names.astype(str), material_names.astype(str)):
        raise ValueError(
            "Stage-08 material names disagree with material_indices and dataset table."
        )

    with np.load(final_spectra, allow_pickle=True) as data:
        wavelengths = np.asarray(
            npz_first(data, ("wavelengths_nm", "wavelengths", "wavelength_nm", "wl_nm")),
            dtype=np.float64,
        )
        perturbations = np.asarray(
            npz_first(
                data,
                (
                    "perturbations_nm",
                    "perturbation_deltas_nm",
                    "deltas_nm",
                    "state_deltas_nm",
                ),
            ),
            dtype=np.int64,
        )
        qsca = np.asarray(
            npz_first(data, ("qsca", "qsca_81", "all_qsca_81", "state_qsca")),
            dtype=np.float64,
        )
        qabs = np.asarray(
            npz_first(data, ("qabs", "qabs_81", "all_qabs_81", "state_qabs")),
            dtype=np.float64,
        )
        qext = np.asarray(
            npz_first(data, ("qext", "qext_81", "all_qext_81", "state_qext")),
            dtype=np.float64,
        )

    def normalize_states(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return values[None, :]
        if values.ndim != 2:
            raise ValueError(f"Expected 2D spectra, got {values.shape}.")
        if values.shape[1] == len(wavelengths):
            return values
        if values.shape[0] == len(wavelengths):
            return values.T
        raise ValueError(f"Cannot infer spectrum orientation: {values.shape}.")

    qsca = normalize_states(qsca)
    qabs = normalize_states(qabs)
    qext = normalize_states(qext)

    if perturbations.ndim != 2:
        raise ValueError(f"Unexpected perturbation shape: {perturbations.shape}.")
    if perturbations.shape[1] != 4 and perturbations.shape[0] == 4:
        perturbations = perturbations.T
    if perturbations.shape != (qsca.shape[0], 4):
        raise ValueError("Perturbation table does not match the 81-state spectra.")

    zero_rows = np.flatnonzero(np.all(perturbations == 0, axis=1))
    if zero_rows.size != 1:
        raise ValueError(
            f"Expected exactly one nominal perturbation [0,0,0,0], found {zero_rows.size}."
        )
    nominal_index = int(zero_rows[0])

    nominal_qsca = qsca[nominal_index].copy()
    nominal_qabs = qabs[nominal_index].copy()
    nominal_qext = qext[nominal_index].copy()

    if not (
        np.isfinite(nominal_qsca).all()
        and np.isfinite(nominal_qabs).all()
        and np.isfinite(nominal_qext).all()
    ):
        raise ValueError("Nominal Stage-08 spectrum contains NaN/Inf.")
    if float(np.min(nominal_qsca)) < -NEGATIVE_TOL:
        raise ValueError("Nominal Qsca contains significant negative values.")
    if float(np.min(nominal_qabs)) < -NEGATIVE_TOL:
        raise ValueError("Nominal Qabs contains significant negative values.")
    residual = float(np.max(np.abs(nominal_qext - nominal_qsca - nominal_qabs)))
    if residual > ENERGY_TOL:
        raise ValueError(
            f"Stage-08 nominal spectrum violates Qext=Qsca+Qabs: {residual:.3e}"
        )

    return {
        "summary": summary,
        "structure": structure,
        "material_indices": material_indices,
        "material_names": material_names,
        "geometry_nm": geometry,
        "radii_nm": radii,
        "outer_radius_nm": float(radii[-1]),
        "wavelengths_nm": wavelengths,
        "nominal_qsca": np.maximum(nominal_qsca, 0.0),
        "nominal_qabs": np.maximum(nominal_qabs, 0.0),
        "nominal_qext": np.maximum(nominal_qext, 0.0),
        "perturbations_nm": perturbations,
        "all_qsca": qsca,
        "all_qabs": qabs,
        "all_qext": qext,
        "dataset": dataset,
        "input_hashes": {
            "final_json_sha256": sha256_file(final_json),
            "final_spectra_sha256": sha256_file(final_spectra),
            "dataset_sha256": sha256_file(dataset_path),
        },
    }


def interpolate_material_tables(
    design: Dict[str, Any],
    query_wavelengths_nm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = design["dataset"]
    src_wl = dataset["wavelengths_nm"]
    query = np.asarray(query_wavelengths_nm, dtype=np.float64)

    if query.min() < src_wl.min() or query.max() > src_wl.max():
        raise ValueError("Requested wavelengths lie outside the material dispersion table.")

    n_interp = np.empty((len(dataset["material_names"]), len(query)), dtype=np.float64)
    k_interp = np.empty_like(n_interp)
    for index in range(len(dataset["material_names"])):
        n_interp[index] = np.interp(query, src_wl, dataset["material_n"][index])
        k_interp[index] = np.interp(query, src_wl, dataset["material_k"][index])
    return n_interp, k_interp


def build_eps_layers(
    design: Dict[str, Any],
    query_wavelengths_nm: np.ndarray,
) -> np.ndarray:
    n_interp, k_interp = interpolate_material_tables(design, query_wavelengths_nm)
    indices = design["material_indices"]
    refractive_index = n_interp[indices] + 1j * k_interp[indices]
    return (refractive_index ** 2)[None, :, :].astype(np.complex128)


def singular_measure(outer_radius_nm: float, wavelengths_nm: np.ndarray) -> np.ndarray:
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    phase = 2.0 * np.pi * float(outer_radius_nm) / wavelengths_nm
    return np.abs(np.sin(phase))


def select_representative_wavelengths(design: Dict[str, Any]) -> np.ndarray:
    """
    Choose one high-scattering wavelength from each of three broad bands:
    400–550, 550–750, and 750–1000 nm.

    Exact Peña singular points are excluded from selection.
    """
    wavelengths = design["wavelengths_nm"]
    qsca = design["nominal_qsca"]
    outer = design["outer_radius_nm"]
    bands = ((400.0, 550.0), (550.0, 750.0), (750.0, 1000.000001))
    selected: List[float] = []

    safe = singular_measure(outer, wavelengths) > 1.0e-8
    for band_index, (low, high) in enumerate(bands):
        mask = (wavelengths >= low) & (wavelengths < high) & safe
        candidate_indices = np.flatnonzero(mask)
        if candidate_indices.size == 0:
            raise RuntimeError(f"No singular-safe wavelength in band {low:g}–{high:g} nm.")
        best_local = candidate_indices[int(np.argmax(qsca[candidate_indices]))]
        selected.append(float(wavelengths[best_local]))

    if len(set(selected)) != 3:
        raise RuntimeError(f"Representative wavelengths are not unique: {selected}")
    return np.asarray(selected, dtype=np.float64)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {path}\n"
                "Use --overwrite for an intentional rebuild."
            )
        for item in path.iterdir():
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
            else:
                item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def cumulative_trapezoid_normalized(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    areas = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    cumulative = np.concatenate(([0.0], np.cumsum(areas)))
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(cumulative)
    return cumulative / total
