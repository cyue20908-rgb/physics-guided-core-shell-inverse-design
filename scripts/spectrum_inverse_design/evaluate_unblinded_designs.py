# -*- coding: utf-8 -*-
"""Post-unblinding comparison of frozen designs and hidden references.

This stage is deliberately read-only with respect to the frozen inverse-design
results. It first verifies the Stage-06A freeze record and its SHA256 snapshot,
then opens the hidden generating references exactly once for analysis.

The stage performs no optimization, search, material reranking, geometry
adjustment, or model inference. Its purpose is to distinguish:

1. optical-response equivalence (the primary inverse-design objective), and
2. recovery of the particular hidden generating structure (a secondary,
   non-unique inverse-problem diagnostic).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from core_shell_optical_design.project_paths import (
    DATA_DIR,
    REPRODUCED_INVERSE_RESULTS_DIR,
)

STAGE = "post_unblinding_evaluation"
VERSION = "tandem-mie-inverse-2026-07-25-stage06b-v1"

RESULT_ROOT = REPRODUCED_INVERSE_RESULTS_DIR
DEFAULT_FREEZE_DIR = RESULT_ROOT / "pre_unblinding_freeze"
DEFAULT_PRIVATE_DIR = DATA_DIR / "blind_benchmark" / "private_reference"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "post_unblinding_evaluation"

MATERIAL_NAMES = np.asarray(
    ["Al2O3", "BaSO4", "SiO2", "TiO2", "PDMS", "Air"], dtype="U16"
)
LOWER_BOUNDS = np.asarray([120.0, 20.0, 20.0, 20.0], dtype=np.float64)
UPPER_BOUNDS = np.asarray([220.0, 80.0, 80.0, 80.0], dtype=np.float64)
MAX_OUTER_RADIUS = 300.0

# Publication-quality plotting parameters.  Internal benchmark task IDs remain
# unchanged; only their display names in figures are normalized to
# Target1-Target10.
PLOT_DPI = 600
FONT_FAMILY = "Times New Roman"
AXIS_LINE_WIDTH = 1.25
TICK_LINE_WIDTH = 1.10
TARGET_COLOR = "#202020"
INVERSE_COLOR = "#D62728"
REFERENCE_COLOR = "#4C78A8"
DESIGN_COLOR = "#F58518"

MATERIAL_PLOT_LABELS = {
    "Al2O3": r"$\mathbf{Al_2O_3}$",
    "BaSO4": r"$\mathbf{BaSO_4}$",
    "SiO2": r"$\mathbf{SiO_2}$",
    "TiO2": r"$\mathbf{TiO_2}$",
    "PDMS": r"$\mathbf{PDMS}$",
    "Air": r"$\mathbf{Air}$",
}
GEOMETRY_PLOT_LABELS = [
    r"$\mathbf{r}_{\mathbf{core}}$",
    r"$\mathbf{t}_{1}$",
    r"$\mathbf{t}_{2}$",
    r"$\mathbf{t}_{3}$",
]
Q_SCA_LABEL = r"$\mathbf{Q}_{\mathbf{sca}}$"
Q_ABS_LABEL = r"$\mathbf{Q}_{\mathbf{abs}}$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the pre-unblind freeze, then compare frozen designs with hidden references"
    )
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE_DIR)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--spectrum-atol", type=float, default=1.0e-12)
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--spectral-success-nmae", type=float, default=0.02)
    parser.add_argument("--spectral-success-nrmse", type=float, default=0.02)
    parser.add_argument("--spectral-success-r2", type=float, default=0.99)
    parser.add_argument("--spectral-success-peak-nm", type=float, default=6.0)
    parser.add_argument("--structure-close-tolerance-nm", type=float, default=2.0)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=json_default)
    os.replace(temp, path)


def save_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def spectrum_hash(wavelengths: np.ndarray, qsca: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(wavelengths, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(qsca, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def verify_artifact_record(record: Mapping[str, Any], path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(record.get("sha256", ""))
    expected_size = int(record.get("size_bytes", -1))
    actual_hash = sha256_file(path)
    actual_size = int(path.stat().st_size)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected_hash}, received {actual_hash}"
        )
    if expected_size >= 0 and actual_size != expected_size:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size}, received {actual_size}"
        )


def verify_freeze_record(freeze: Mapping[str, Any], freeze_path: Path) -> str:
    if str(freeze.get("status", "")).lower() != "passed":
        raise RuntimeError("Stage 06A freeze record does not have passed status.")
    if bool(freeze.get("private_reference_accessed", True)):
        raise RuntimeError("Stage 06A freeze record says the private reference was accessed.")
    expected = str(freeze.get("freeze_record_sha256", ""))
    if not expected:
        raise RuntimeError("Stage 06A freeze record lacks freeze_record_sha256.")
    payload = dict(freeze)
    payload.pop("freeze_record_sha256", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != expected:
        raise RuntimeError(
            "pre_unblinding_freeze_record.json content hash is invalid; frozen facts may have changed."
        )
    if int(freeze.get("counts", {}).get("frozen_results", -1)) != 10:
        raise RuntimeError("Stage 06A freeze record must contain ten frozen results.")
    return sha256_file(freeze_path)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Stage 06B output already exists: {path}\n"
            "Use --overwrite only to reproduce the same post-hoc analysis. "
            "Stage 03-05 frozen designs must never be modified or reranked."
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)


def require_keys(values: Mapping[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in values]
    if missing:
        raise KeyError(f"{label} is missing required keys: {missing}")


def legal_geometry_mask(geometry: np.ndarray, tolerance: float = 1.0e-8) -> np.ndarray:
    geometry = np.asarray(geometry, dtype=np.float64)
    return (
        np.all(geometry >= LOWER_BOUNDS[None, :] - tolerance, axis=1)
        & np.all(geometry <= UPPER_BOUNDS[None, :] + tolerance, axis=1)
        & (np.sum(geometry, axis=1) <= MAX_OUTER_RADIUS + tolerance)
    )


def legal_material_mask(materials: np.ndarray) -> np.ndarray:
    materials = np.asarray(materials, dtype=np.int64)
    return (
        np.all((materials >= 0) & (materials < len(MATERIAL_NAMES)), axis=1)
        & np.all(materials[:, 1:] != 5, axis=1)
        & np.all(materials[:, 1:] != materials[:, :-1], axis=1)
    )


def energy_residual(qsca: np.ndarray, qabs: np.ndarray, qext: np.ndarray) -> np.ndarray:
    numerator = np.max(np.abs(qext - (qsca + qabs)), axis=1)
    denominator = np.maximum(np.max(np.abs(qext), axis=1), 1.0)
    return numerator / denominator


def spectral_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    wavelengths: np.ndarray,
    *,
    scale_floor: float = 1.0e-12,
) -> Dict[str, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError(
            f"Reference and prediction must share a 2D shape, got {reference.shape} and {prediction.shape}."
        )
    error = prediction - reference
    scale = np.max(np.abs(reference), axis=1)
    usable_scale = scale > scale_floor
    nmae = np.full(len(reference), np.nan, dtype=np.float64)
    nrmse = np.full(len(reference), np.nan, dtype=np.float64)
    nmae[usable_scale] = np.mean(np.abs(error[usable_scale]), axis=1) / scale[usable_scale]
    nrmse[usable_scale] = (
        np.sqrt(np.mean(error[usable_scale] ** 2, axis=1)) / scale[usable_scale]
    )
    mae = np.mean(np.abs(error), axis=1)
    rmse = np.sqrt(np.mean(error**2, axis=1))
    maximum_abs_error = np.max(np.abs(error), axis=1)

    centered = reference - np.mean(reference, axis=1, keepdims=True)
    ss_tot = np.sum(centered**2, axis=1)
    ss_res = np.sum(error**2, axis=1)
    r2 = np.where(
        ss_tot > scale_floor**2,
        1.0 - ss_res / ss_tot,
        np.where(ss_res <= scale_floor**2, 1.0, np.nan),
    )

    ref_norm = np.linalg.norm(reference, axis=1)
    pred_norm = np.linalg.norm(prediction, axis=1)
    cosine = np.sum(reference * prediction, axis=1) / np.maximum(
        ref_norm * pred_norm, scale_floor
    )

    ref_center = reference - np.mean(reference, axis=1, keepdims=True)
    pred_center = prediction - np.mean(prediction, axis=1, keepdims=True)
    pearson = np.sum(ref_center * pred_center, axis=1) / np.maximum(
        np.linalg.norm(ref_center, axis=1) * np.linalg.norm(pred_center, axis=1),
        scale_floor,
    )

    ref_peak_index = np.argmax(reference, axis=1)
    pred_peak_index = np.argmax(prediction, axis=1)
    peak_error_nm = np.abs(wavelengths[pred_peak_index] - wavelengths[ref_peak_index])

    return {
        "scale": scale,
        "reference_near_zero": ~usable_scale,
        "mae": mae,
        "rmse": rmse,
        "nmae": nmae,
        "nrmse": nrmse,
        "r2": r2,
        "cosine_similarity": cosine,
        "pearson_r": pearson,
        "maximum_abs_error": maximum_abs_error,
        "reference_peak_nm": wavelengths[ref_peak_index],
        "prediction_peak_nm": wavelengths[pred_peak_index],
        "peak_error_nm": peak_error_nm,
    }


def nearest_legal_integer_geometry(reference: np.ndarray) -> np.ndarray:
    """Return a deterministic nearest legal 1-nm geometry.

    For each component only floor/ceil can be nearest. All legal combinations
    are evaluated and ranked by L2 distance, then L1 distance, then
    lexicographic order. This avoids an arbitrary half-to-even rounding rule.
    """

    reference = np.asarray(reference, dtype=np.float64)
    output = np.empty_like(reference, dtype=np.int64)
    for row_index, row in enumerate(reference):
        choices = []
        for value in row:
            floor_value = int(math.floor(float(value)))
            ceil_value = int(math.ceil(float(value)))
            choices.append(sorted(set((floor_value, ceil_value))))
        candidates = np.asarray(list(itertools.product(*choices)), dtype=np.int64)
        legal = legal_geometry_mask(candidates.astype(np.float64))
        candidates = candidates[legal]
        if len(candidates) == 0:
            raise RuntimeError(
                f"No legal nearest integer projection exists for reference geometry {row.tolist()}."
            )
        difference = candidates.astype(np.float64) - row[None, :]
        l2 = np.sum(difference**2, axis=1)
        l1 = np.sum(np.abs(difference), axis=1)
        order = sorted(
            range(len(candidates)),
            key=lambda i: (
                float(l2[i]),
                float(l1[i]),
                tuple(int(v) for v in candidates[i]),
            ),
        )
        output[row_index] = candidates[order[0]]
    return output


def material_string(indices: np.ndarray) -> str:
    return "|".join(MATERIAL_NAMES[np.asarray(indices, dtype=np.int64)].tolist())


def material_plot_label(name: Any) -> str:
    """Return a bold chemistry-formatted material name for figure text."""
    key = str(name).strip()
    return MATERIAL_PLOT_LABELS.get(key, rf"$\mathbf{{{key}}}$")


def material_sequence_plot_string(indices: np.ndarray) -> str:
    """Return a publication-formatted four-layer material sequence."""
    names = MATERIAL_NAMES[np.asarray(indices, dtype=np.int64)].tolist()
    return " / ".join(material_plot_label(name) for name in names)


def target_display_label(index: int) -> str:
    """Return the standardized visible target name without changing task IDs."""
    if not 0 <= int(index) < 10:
        raise ValueError(f"Target index must lie in 0-9, received {index}.")
    return f"Target{int(index) + 1}"


def configure_plot_style() -> None:
    """Configure all Stage-06B figures for publication-quality output."""
    from matplotlib import font_manager

    try:
        font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
    except Exception as exc:
        raise RuntimeError(
            f"Required plotting font '{FONT_FAMILY}' was not found. "
            "Install/enable Times New Roman before generating final figures."
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY, "DejaVu Serif"],
            "font.size": 10.5,
            "font.weight": "bold",
            "text.color": "black",
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": AXIS_LINE_WIDTH,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": TICK_LINE_WIDTH,
            "ytick.major.width": TICK_LINE_WIDTH,
            "xtick.minor.width": 0.85,
            "ytick.minor.width": 0.85,
            "xtick.major.size": 4.6,
            "ytick.major.size": 4.6,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
            "legend.fontsize": 9.2,
            "legend.frameon": True,
            "legend.framealpha": 0.97,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "legend.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": PLOT_DPI,
            "figure.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "mathtext.fallback": "stix",
            "mathtext.default": "bf",
            "axes.unicode_minus": False,
        }
    )


def style_axis(
    ax: Any,
    *,
    grid: bool = True,
    minor_x: bool = False,
    minor_y: bool = True,
) -> None:
    """Keep four black spines but show ticks only on the left and bottom."""
    if minor_x and ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if minor_y and ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(AXIS_LINE_WIDTH)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
        labeltop=False,
        labelright=False,
        colors="black",
        labelcolor="black",
        width=TICK_LINE_WIDTH,
        length=4.6,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
        colors="black",
        width=0.85,
        length=2.4,
    )

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontweight("bold")
        label.set_color("black")
    for text_item in (ax.xaxis.label, ax.yaxis.label, ax.title):
        text_item.set_fontname(FONT_FAMILY)
        text_item.set_fontweight("bold")
        text_item.set_color("black")
    for offset_text in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
        offset_text.set_fontname(FONT_FAMILY)
        offset_text.set_fontweight("bold")
        offset_text.set_color("black")

    ax.grid(False)
    if grid:
        ax.yaxis.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.60,
            color="0.60",
            alpha=0.25,
        )
    ax.set_axisbelow(True)


def style_legend(owner: Any, *args: Any, **kwargs: Any) -> Any:
    """Create an opaque Times New Roman bold legend."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("framealpha", 0.97)
    kwargs.setdefault("fancybox", False)
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("facecolor", "white")
    legend = owner.legend(*args, **kwargs)
    legend.get_frame().set_linewidth(0.9)
    for item in legend.get_texts():
        item.set_fontname(FONT_FAMILY)
        item.set_fontweight("bold")
        item.set_color("black")
    title = legend.get_title()
    if title is not None:
        title.set_fontname(FONT_FAMILY)
        title.set_fontweight("bold")
        title.set_color("black")
    return legend


def style_colorbar(colorbar: Any, label: str) -> None:
    colorbar.set_label(label, fontname=FONT_FAMILY, fontweight="bold")
    colorbar.outline.set_edgecolor("black")
    colorbar.outline.set_linewidth(AXIS_LINE_WIDTH)
    colorbar.ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        colors="black",
        width=TICK_LINE_WIDTH,
        length=4.0,
    )
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontname(FONT_FAMILY)
        tick_label.set_fontweight("bold")
        tick_label.set_color("black")


def save_figure_pair(fig: Any, png_path: Path) -> None:
    """Save one figure as a 600-dpi PNG and an editable SVG."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(
        png_path,
        format="png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    plt.close(fig)


def plot_target(
    output_path_png: Path,
    *,
    target_label: str,
    difficulty: str,
    wavelengths: np.ndarray,
    ref_qsca: np.ndarray,
    pred_qsca: np.ndarray,
    ref_qabs: np.ndarray,
    pred_qabs: np.ndarray,
    ref_geometry: np.ndarray,
    pred_geometry: np.ndarray,
    ref_materials: np.ndarray,
    pred_materials: np.ndarray,
    qsca_nmae: float,
) -> None:
    """Plot one standardized target/inverse-design post-hoc comparison."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.6, 7.7),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.plot(
        wavelengths,
        ref_qsca,
        color=TARGET_COLOR,
        linewidth=2.25,
        linestyle="-",
        label=target_label,
        zorder=4,
    )
    ax.plot(
        wavelengths,
        pred_qsca,
        color=INVERSE_COLOR,
        linewidth=1.85,
        linestyle="--",
        label=f"Inverse design for {target_label}",
        zorder=3,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(Q_SCA_LABEL)
    ax.set_title(rf"(a) Scattering-spectrum equivalence; NMAE = {qsca_nmae:.4g}")
    ax.set_xlim(float(np.min(wavelengths)), float(np.max(wavelengths)))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axis(ax, grid=True, minor_x=True, minor_y=True)
    style_legend(ax, loc="best")

    ax = axes[0, 1]
    ax.plot(
        wavelengths,
        ref_qabs,
        color=REFERENCE_COLOR,
        linewidth=2.25,
        linestyle="-",
        label=rf"Reference {Q_ABS_LABEL} of {target_label}",
        zorder=4,
    )
    ax.plot(
        wavelengths,
        pred_qabs,
        color=DESIGN_COLOR,
        linewidth=1.85,
        linestyle="--",
        label=rf"Inverse-designed {Q_ABS_LABEL} of {target_label}",
        zorder=3,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(Q_ABS_LABEL)
    ax.set_title(r"(b) Post-hoc absorption comparison (not optimized)")
    ax.set_xlim(float(np.min(wavelengths)), float(np.max(wavelengths)))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axis(ax, grid=True, minor_x=True, minor_y=True)
    style_legend(ax, loc="best")

    x = np.arange(4)
    width = 0.36
    ax = axes[1, 0]
    ax.bar(
        x - width / 2,
        ref_geometry,
        width=width,
        color=REFERENCE_COLOR,
        edgecolor="black",
        linewidth=0.8,
        label="Reference structure",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        pred_geometry,
        width=width,
        color=DESIGN_COLOR,
        edgecolor="black",
        linewidth=0.8,
        label="Inverse-designed structure",
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(GEOMETRY_PLOT_LABELS)
    ax.set_ylabel("Geometry parameter (nm)")
    ax.set_title("(c) Geometry-parameter comparison")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axis(ax, grid=True, minor_x=False, minor_y=True)
    style_legend(ax, loc="best")

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"Target: {target_label}",
        f"Difficulty: {difficulty}",
        "",
        "Reference material sequence:",
        material_sequence_plot_string(ref_materials),
        "",
        "Inverse-designed material sequence:",
        material_sequence_plot_string(pred_materials),
        "",
        f"Material Hamming distance: {int(np.sum(ref_materials != pred_materials))}/4",
        f"Geometry MAE: {float(np.mean(np.abs(pred_geometry - ref_geometry))):.3f} nm",
        rf"$\mathbf{{R}}_{{\mathbf{{out}}}}$ error: {float(np.sum(pred_geometry) - np.sum(ref_geometry)):+.3f} nm",
    ]
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.2,
        fontname=FONT_FAMILY,
        fontweight="bold",
        color="black",
        linespacing=1.25,
        transform=ax.transAxes,
    )
    ax.set_title("(d) Structural recovery summary", fontweight="bold")

    fig.suptitle(
        f"{target_label}: target and inverse-design comparison",
        fontsize=15,
        fontweight="bold",
        fontname=FONT_FAMILY,
    )
    save_figure_pair(fig, output_path_png)


def plot_aggregate(output_dir: Path, frame: pd.DataFrame, confusion: np.ndarray) -> None:
    """Plot aggregate target metrics and the material-recovery matrix."""
    x = np.arange(len(frame))
    target_labels = [target_display_label(index) for index in range(len(frame))]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.8, 8.0),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    ax.bar(x, frame["qsca_nmae"], color="#B279A2", edgecolor="black", linewidth=0.75)
    ax.axhline(
        0.02,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="High-accuracy threshold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=35, ha="right")
    ax.set_ylabel(r"$\mathbf{Q}_{\mathbf{sca}}$ NMAE")
    ax.set_title("(a) Frozen inverse-design optical error")
    style_axis(ax, grid=True, minor_x=False, minor_y=True)
    style_legend(ax, loc="best")

    ax = axes[0, 1]
    ax.bar(
        x,
        frame["geometry_parameter_mae_nm"],
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.75,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=35, ha="right")
    ax.set_ylabel("Geometry MAE (nm)")
    ax.set_title("(b) Distance from the reference geometry")
    style_axis(ax, grid=True, minor_x=False, minor_y=True)

    ax = axes[1, 0]
    ax.bar(
        x,
        frame["material_hamming_distance"],
        color="#54A24B",
        edgecolor="black",
        linewidth=0.75,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=35, ha="right")
    ax.set_ylabel("Incorrect layers (0-4)")
    ax.set_ylim(0, 4.25)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.set_title("(c) Material-sequence recovery")
    style_axis(ax, grid=True, minor_x=False, minor_y=False)

    ax = axes[1, 1]
    for difficulty, group in frame.groupby("difficulty", sort=False):
        ax.scatter(
            group["geometry_parameter_mae_nm"],
            group["qsca_nmae"],
            s=58,
            edgecolor="black",
            linewidth=0.55,
            label=str(difficulty).capitalize(),
            zorder=3,
        )
    ax.set_xlabel("Geometry MAE from reference (nm)")
    ax.set_ylabel(r"$\mathbf{Q}_{\mathbf{sca}}$ NMAE")
    ax.set_title("(d) Spectral equivalence vs structural similarity")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axis(ax, grid=True, minor_x=True, minor_y=True)
    style_legend(ax, title="Difficulty", loc="best")

    fig.suptitle(
        "Post-hoc comparison for Target1-Target10",
        fontsize=15,
        fontweight="bold",
        fontname=FONT_FAMILY,
    )
    save_figure_pair(fig, output_dir / "aggregate_unblind_comparison.png")

    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    image = ax.imshow(confusion, cmap="Blues", interpolation="nearest")
    material_labels = [material_plot_label(name) for name in MATERIAL_NAMES]
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(material_labels, rotation=32, ha="right")
    ax.set_yticklabels(material_labels)
    ax.set_xlabel("Inverse-designed material")
    ax.set_ylabel("Reference material")
    ax.set_title("Layer-wise material recovery confusion matrix")
    threshold = 0.55 * float(np.max(confusion)) if np.size(confusion) else 0.0
    for row in range(6):
        for column in range(6):
            value = int(confusion[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontname=FONT_FAMILY,
                fontweight="bold",
            )
    style_axis(ax, grid=False, minor_x=False, minor_y=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    style_colorbar(colorbar, "Layer count")
    save_figure_pair(fig, output_dir / "material_recovery_confusion.png")


def main() -> None:
    args = parse_args()
    for name in (
        "spectrum_atol",
        "energy_tolerance",
        "spectral_success_nmae",
        "spectral_success_nrmse",
        "structure_close_tolerance_nm",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")

    started = time.time()
    freeze_dir = Path(args.freeze_dir)
    snapshot_dir = freeze_dir / "snapshot"
    private_dir = Path(args.private_dir)
    output_dir = Path(args.output_dir)

    freeze_path = freeze_dir / "pre_unblinding_freeze_record.json"
    public_path = snapshot_dir / "blind_target_spectra.npz"
    manifest_path = snapshot_dir / "blind_benchmark_manifest.json"
    lock_path = snapshot_dir / "blind_benchmark_lock.json"
    frozen_npz_path = snapshot_dir / "frozen_blind_design_results.npz"
    frozen_csv_path = snapshot_dir / "frozen_blind_design_results.csv"
    private_path = private_dir / "hidden_reference_structures.npz"

    print("Stage 06B post-hoc hidden-reference unblind comparison")
    print("Verifying Stage 06A freeze before opening the private reference.")

    freeze = load_json(freeze_path)
    freeze_file_sha256 = verify_freeze_record(freeze, freeze_path)
    benchmark_id = str(freeze.get("benchmark_id", ""))
    if not benchmark_id:
        raise RuntimeError("Stage 06A freeze record lacks benchmark_id.")

    snapshot_records = freeze.get("snapshot_artifacts", {})
    if not isinstance(snapshot_records, dict):
        raise RuntimeError("Stage 06A freeze record has invalid snapshot_artifacts.")
    required_snapshot_paths = {
        "blind_target_spectra.npz": public_path,
        "blind_benchmark_manifest.json": manifest_path,
        "blind_benchmark_lock.json": lock_path,
        "frozen_blind_design_results.npz": frozen_npz_path,
        "frozen_blind_design_results.csv": frozen_csv_path,
    }
    for name, path in required_snapshot_paths.items():
        if name not in snapshot_records:
            raise RuntimeError(f"Stage 06A freeze record lacks snapshot hash for {name}.")
        verify_artifact_record(snapshot_records[name], path, f"Frozen snapshot {name}")

    lock = load_json(lock_path)
    manifest = load_json(manifest_path)
    if str(lock.get("benchmark_id", "")) != benchmark_id:
        raise RuntimeError("Benchmark lock ID differs from the Stage 06A freeze ID.")
    if str(manifest.get("benchmark_id", "")) != benchmark_id:
        raise RuntimeError("Public manifest ID differs from the Stage 06A freeze ID.")
    if sha256_file(public_path) != str(lock.get("public_target_sha256", "")):
        raise RuntimeError("Frozen public-target hash differs from blind_benchmark_lock.json.")
    if sha256_file(manifest_path) != str(lock.get("manifest_sha256", "")):
        raise RuntimeError("Frozen public-manifest hash differs from blind_benchmark_lock.json.")

    private_record = manifest.get("outputs", {}).get("private_reference", {})
    if not isinstance(private_record, dict) or not private_record.get("sha256"):
        raise RuntimeError("The frozen public manifest lacks the private-reference artifact hash.")
    verify_artifact_record(private_record, private_path, "Hidden private reference")

    # The hash chain is now verified. Opening the private file below is the
    # formal unblind point. No search, optimization, or reranking follows.
    print("Freeze chain PASSED. Opening hidden reference for post-hoc analysis only.")
    public = load_npz(public_path)
    frozen = load_npz(frozen_npz_path)
    frozen_csv = pd.read_csv(frozen_csv_path, encoding="utf-8-sig")
    private = load_npz(private_path)

    require_keys(
        public,
        ["task_id", "wavelengths_nm", "target_qsca", "target_hash", "benchmark_id"],
        "Public target snapshot",
    )
    require_keys(
        private,
        [
            "task_id",
            "benchmark_id",
            "reference_geometry",
            "reference_radii",
            "reference_material_indices",
            "reference_material_names",
            "reference_qsca",
            "reference_qabs",
            "reference_qext",
            "target_hash",
        ],
        "Private reference",
    )
    require_keys(
        frozen,
        [
            "task_id",
            "wavelengths_nm",
            "target_qsca",
            "target_hash",
            "candidate_task_index",
            "candidate_integer_rank",
            "candidate_design_uid",
            "candidate_material_indices",
            "candidate_integer_geometry",
            "candidate_mie_qsca",
            "candidate_mie_qabs",
            "candidate_mie_qext",
        ],
        "Frozen Stage 05 snapshot",
    )

    public_task_id = public["task_id"].astype(str)
    wavelengths = np.asarray(public["wavelengths_nm"], dtype=np.float64)
    public_qsca = np.asarray(public["target_qsca"], dtype=np.float64)
    public_hash = public["target_hash"].astype(str)
    if len(public_task_id) != 10 or public_qsca.shape != (10, len(wavelengths)):
        raise RuntimeError("Expected ten public target spectra on one shared wavelength grid.")
    public_benchmark_id = str(np.asarray(public["benchmark_id"]).item())
    private_benchmark_id = str(np.asarray(private["benchmark_id"]).item())
    if public_benchmark_id != benchmark_id or private_benchmark_id != benchmark_id:
        raise RuntimeError("Public/private benchmark IDs do not match the Stage 06A freeze.")

    private_task_id = private["task_id"].astype(str)
    private_position = {task: index for index, task in enumerate(private_task_id.tolist())}
    if set(private_position) != set(public_task_id.tolist()):
        raise RuntimeError("Private and public task ID sets differ.")
    private_order = np.asarray([private_position[task] for task in public_task_id], dtype=np.int64)

    reference_geometry = np.asarray(private["reference_geometry"], dtype=np.float64)[private_order]
    reference_radii = np.asarray(private["reference_radii"], dtype=np.float64)[private_order]
    reference_materials = np.asarray(private["reference_material_indices"], dtype=np.int64)[private_order]
    reference_material_names = np.asarray(private["reference_material_names"]).astype(str)[private_order]
    reference_qsca = np.asarray(private["reference_qsca"], dtype=np.float64)[private_order]
    reference_qabs = np.asarray(private["reference_qabs"], dtype=np.float64)[private_order]
    reference_qext = np.asarray(private["reference_qext"], dtype=np.float64)[private_order]
    private_hash = private["target_hash"].astype(str)[private_order]

    if not np.array_equal(private_hash, public_hash):
        raise RuntimeError("Private target hashes do not match the frozen public target hashes.")
    recomputed_hash = np.asarray(
        [spectrum_hash(wavelengths, row) for row in reference_qsca], dtype="U64"
    )
    if not np.array_equal(recomputed_hash, public_hash):
        raise RuntimeError("Hidden reference Qsca does not reproduce the frozen target hashes.")
    if not np.array_equal(reference_qsca, public_qsca):
        worst = float(np.max(np.abs(reference_qsca - public_qsca)))
        raise RuntimeError(
            f"Hidden reference Qsca is not bit-identical to the public benchmark; max difference={worst:.3e}."
        )
    if not np.allclose(reference_radii, np.cumsum(reference_geometry, axis=1), atol=0.0, rtol=0.0):
        raise RuntimeError("Private reference radii do not equal cumulative geometry parameters.")
    if not np.all(legal_geometry_mask(reference_geometry)):
        raise RuntimeError("A hidden generating geometry is outside the declared design domain.")
    if not np.all(legal_material_mask(reference_materials)):
        raise RuntimeError("A hidden generating material sequence is illegal.")
    expected_reference_names = MATERIAL_NAMES[reference_materials]
    if not np.array_equal(reference_material_names, expected_reference_names):
        raise RuntimeError("Private material names disagree with the frozen material-index convention.")

    candidate_task_index = np.asarray(frozen["candidate_task_index"], dtype=np.int64)
    if len(candidate_task_index) != 10 or sorted(candidate_task_index.tolist()) != list(range(10)):
        raise RuntimeError("Frozen snapshot must contain exactly one candidate for each task index 0-9.")
    candidate_order = np.argsort(candidate_task_index, kind="mergesort")
    prediction_geometry = np.asarray(frozen["candidate_integer_geometry"], dtype=np.int64)[candidate_order]
    prediction_materials = np.asarray(frozen["candidate_material_indices"], dtype=np.int64)[candidate_order]
    prediction_qsca = np.asarray(frozen["candidate_mie_qsca"], dtype=np.float64)[candidate_order]
    prediction_qabs = np.asarray(frozen["candidate_mie_qabs"], dtype=np.float64)[candidate_order]
    prediction_qext = np.asarray(frozen["candidate_mie_qext"], dtype=np.float64)[candidate_order]
    design_uid = frozen["candidate_design_uid"].astype(str)[candidate_order]
    integer_rank = np.asarray(frozen["candidate_integer_rank"], dtype=np.int64)[candidate_order]
    if not np.all(integer_rank == 1):
        raise RuntimeError("Stage 06B may compare only the pre-unblind frozen rank-1 designs.")
    if not np.all(legal_geometry_mask(prediction_geometry.astype(np.float64))):
        raise RuntimeError("A frozen inverse geometry is illegal.")
    if not np.all(legal_material_mask(prediction_materials)):
        raise RuntimeError("A frozen inverse material sequence is illegal.")

    if prediction_qsca.shape != reference_qsca.shape:
        raise RuntimeError("Frozen and hidden spectra have incompatible shapes.")
    negative_minimum = float(
        min(
            reference_qsca.min(),
            reference_qabs.min(),
            reference_qext.min(),
            prediction_qsca.min(),
            prediction_qabs.min(),
            prediction_qext.min(),
        )
    )
    if negative_minimum < -float(args.spectrum_atol):
        raise RuntimeError(f"A saved physical spectrum is materially negative: {negative_minimum:.3e}")
    reference_energy = energy_residual(reference_qsca, reference_qabs, reference_qext)
    prediction_energy = energy_residual(prediction_qsca, prediction_qabs, prediction_qext)
    if float(np.max(reference_energy)) > float(args.energy_tolerance):
        raise RuntimeError("Hidden reference spectra fail energy closure.")
    if float(np.max(prediction_energy)) > float(args.energy_tolerance):
        raise RuntimeError("Frozen design spectra fail energy closure.")

    qsca = spectral_metrics(reference_qsca, prediction_qsca, wavelengths)
    qabs = spectral_metrics(reference_qabs, prediction_qabs, wavelengths)
    qext = spectral_metrics(reference_qext, prediction_qext, wavelengths)

    geometry_error = prediction_geometry.astype(np.float64) - reference_geometry
    abs_geometry_error = np.abs(geometry_error)
    prediction_radii = np.cumsum(prediction_geometry, axis=1).astype(np.float64)
    radii_error = prediction_radii - reference_radii
    abs_radii_error = np.abs(radii_error)
    nearest_reference_integer = nearest_legal_integer_geometry(reference_geometry)

    material_match = prediction_materials == reference_materials
    material_hamming = np.sum(~material_match, axis=1)
    material_exact = material_hamming == 0
    geometry_max_error = np.max(abs_geometry_error, axis=1)
    geometry_close = geometry_max_error <= float(args.structure_close_tolerance_nm)
    geometry_grid_exact = np.all(prediction_geometry == nearest_reference_integer, axis=1)
    structure_close = material_exact & geometry_close
    structure_grid_exact = material_exact & geometry_grid_exact

    difficulty = (
        public["difficulty"].astype(str)
        if "difficulty" in public
        else np.asarray(["unspecified"] * 10, dtype="U16")
    )
    peak_applicable = (
        np.asarray(public["peak_criterion_applicable"], dtype=bool)
        if "peak_criterion_applicable" in public
        else np.ones(10, dtype=bool)
    )
    peak_ok = (~peak_applicable) | (
        qsca["peak_error_nm"] <= float(args.spectral_success_peak_nm)
    )
    spectrally_equivalent = (
        (qsca["nmae"] <= float(args.spectral_success_nmae))
        & (qsca["nrmse"] <= float(args.spectral_success_nrmse))
        & (qsca["r2"] >= float(args.spectral_success_r2))
        & peak_ok
    )
    alternative_equivalent = spectrally_equivalent & ~structure_close

    rows: Dict[str, Any] = {
        "task_index": np.arange(10, dtype=np.int64),
        "task_id": public_task_id,
        "difficulty": difficulty,
        "benchmark_id": [benchmark_id] * 10,
        "target_hash": public_hash,
        "frozen_design_uid": design_uid,
        "predicted_materials": [material_string(row) for row in prediction_materials],
        "reference_materials": [material_string(row) for row in reference_materials],
        "pred_m0": prediction_materials[:, 0],
        "pred_m1": prediction_materials[:, 1],
        "pred_m2": prediction_materials[:, 2],
        "pred_m3": prediction_materials[:, 3],
        "ref_m0": reference_materials[:, 0],
        "ref_m1": reference_materials[:, 1],
        "ref_m2": reference_materials[:, 2],
        "ref_m3": reference_materials[:, 3],
        "material_hamming_distance": material_hamming,
        "layer_material_accuracy": np.mean(material_match, axis=1),
        "exact_material_sequence_recovery": material_exact,
        "pred_r_core_nm": prediction_geometry[:, 0],
        "pred_t1_nm": prediction_geometry[:, 1],
        "pred_t2_nm": prediction_geometry[:, 2],
        "pred_t3_nm": prediction_geometry[:, 3],
        "ref_r_core_nm": reference_geometry[:, 0],
        "ref_t1_nm": reference_geometry[:, 1],
        "ref_t2_nm": reference_geometry[:, 2],
        "ref_t3_nm": reference_geometry[:, 3],
        "nearest_grid_ref_r_core_nm": nearest_reference_integer[:, 0],
        "nearest_grid_ref_t1_nm": nearest_reference_integer[:, 1],
        "nearest_grid_ref_t2_nm": nearest_reference_integer[:, 2],
        "nearest_grid_ref_t3_nm": nearest_reference_integer[:, 3],
        "geometry_parameter_mae_nm": np.mean(abs_geometry_error, axis=1),
        "geometry_parameter_rmse_nm": np.sqrt(np.mean(geometry_error**2, axis=1)),
        "geometry_parameter_l2_nm": np.linalg.norm(geometry_error, axis=1),
        "geometry_parameter_max_abs_error_nm": geometry_max_error,
        "interface_radius_mae_nm": np.mean(abs_radii_error, axis=1),
        "interface_radius_max_abs_error_nm": np.max(abs_radii_error, axis=1),
        "pred_outer_radius_nm": prediction_radii[:, -1],
        "ref_outer_radius_nm": reference_radii[:, -1],
        "outer_radius_signed_error_nm": radii_error[:, -1],
        "outer_radius_abs_error_nm": abs_radii_error[:, -1],
        "geometry_within_1nm": geometry_max_error <= 1.0,
        "geometry_within_2nm": geometry_max_error <= 2.0,
        "geometry_within_5nm": geometry_max_error <= 5.0,
        "nearest_integer_geometry_exact": geometry_grid_exact,
        "structure_recovered_within_tolerance": structure_close,
        "nearest_integer_structure_exact": structure_grid_exact,
        "qsca_nmae": qsca["nmae"],
        "qsca_nrmse": qsca["nrmse"],
        "qsca_r2": qsca["r2"],
        "qsca_mae": qsca["mae"],
        "qsca_max_abs_error": qsca["maximum_abs_error"],
        "qsca_cosine_similarity": qsca["cosine_similarity"],
        "qsca_pearson_r": qsca["pearson_r"],
        "qsca_reference_peak_nm": qsca["reference_peak_nm"],
        "qsca_prediction_peak_nm": qsca["prediction_peak_nm"],
        "qsca_peak_error_nm": qsca["peak_error_nm"],
        "qabs_reference_peak": qabs["scale"],
        "qabs_reference_near_zero": qabs["reference_near_zero"],
        "qabs_mae": qabs["mae"],
        "qabs_nmae": qabs["nmae"],
        "qabs_nrmse": qabs["nrmse"],
        "qabs_r2": qabs["r2"],
        "qext_nmae": qext["nmae"],
        "qext_nrmse": qext["nrmse"],
        "qext_r2": qext["r2"],
        "reference_energy_residual": reference_energy,
        "prediction_energy_residual": prediction_energy,
        "spectrally_equivalent": spectrally_equivalent,
        "alternative_spectrally_equivalent_solution": alternative_equivalent,
    }
    comparison = pd.DataFrame(rows)

    # Verify the snapshot CSV agrees with the NPZ frozen design used here.
    if len(frozen_csv) != 10:
        raise RuntimeError("Frozen Stage 05 CSV snapshot must contain ten rows.")
    frozen_csv = frozen_csv.sort_values("task_index", kind="mergesort").reset_index(drop=True)
    if frozen_csv["task_id"].astype(str).tolist() != public_task_id.tolist():
        raise RuntimeError("Frozen CSV task IDs do not match the benchmark order.")
    csv_geometry = frozen_csv[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.int64)
    csv_materials = frozen_csv[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64)
    if not np.array_equal(csv_geometry, prediction_geometry):
        raise RuntimeError("Frozen CSV and NPZ geometries disagree during unblind.")
    if not np.array_equal(csv_materials, prediction_materials):
        raise RuntimeError("Frozen CSV and NPZ material sequences disagree during unblind.")

    confusion = np.zeros((6, 6), dtype=np.int64)
    for reference_index, prediction_index in zip(reference_materials.ravel(), prediction_materials.ravel()):
        confusion[int(reference_index), int(prediction_index)] += 1
    confusion_frame = pd.DataFrame(confusion, columns=[f"pred_{name}" for name in MATERIAL_NAMES])
    confusion_frame.insert(0, "reference_material", MATERIAL_NAMES)

    by_difficulty_rows = []
    for value, group in comparison.groupby("difficulty", sort=False):
        by_difficulty_rows.append(
            {
                "difficulty": str(value),
                "target_count": int(len(group)),
                "median_qsca_nmae": float(group["qsca_nmae"].median()),
                "maximum_qsca_nmae": float(group["qsca_nmae"].max()),
                "spectrally_equivalent_count": int(group["spectrally_equivalent"].sum()),
                "exact_material_sequence_count": int(group["exact_material_sequence_recovery"].sum()),
                "mean_layer_material_accuracy": float(group["layer_material_accuracy"].mean()),
                "median_geometry_parameter_mae_nm": float(group["geometry_parameter_mae_nm"].median()),
                "structure_recovered_within_tolerance_count": int(
                    group["structure_recovered_within_tolerance"].sum()
                ),
                "alternative_equivalent_solution_count": int(
                    group["alternative_spectrally_equivalent_solution"].sum()
                ),
            }
        )
    by_difficulty = pd.DataFrame(by_difficulty_rows)

    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    comparison_path = output_dir / "post_unblinding_comparison_per_target.csv"
    difficulty_path = output_dir / "post_unblinding_summary_by_difficulty.csv"
    confusion_path = output_dir / "material_recovery_confusion_counts.csv"
    spectra_path = output_dir / "post_unblinding_comparison_spectra.npz"
    save_csv_atomic(comparison_path, comparison)
    save_csv_atomic(difficulty_path, by_difficulty)
    save_csv_atomic(confusion_path, confusion_frame)
    save_npz_atomic(
        spectra_path,
        benchmark_id=np.asarray(benchmark_id),
        task_id=public_task_id,
        difficulty=difficulty,
        wavelengths_nm=wavelengths,
        target_hash=public_hash,
        reference_geometry_nm=reference_geometry,
        reference_radii_nm=reference_radii,
        reference_material_indices=reference_materials,
        nearest_legal_integer_reference_geometry_nm=nearest_reference_integer,
        frozen_geometry_nm=prediction_geometry,
        frozen_radii_nm=prediction_radii,
        frozen_material_indices=prediction_materials,
        reference_qsca=reference_qsca,
        frozen_qsca=prediction_qsca,
        reference_qabs=reference_qabs,
        frozen_qabs=prediction_qabs,
        reference_qext=reference_qext,
        frozen_qext=prediction_qext,
        qsca_nmae=qsca["nmae"],
        qsca_nrmse=qsca["nrmse"],
        qsca_r2=qsca["r2"],
        material_hamming_distance=material_hamming,
        geometry_parameter_error_nm=geometry_error,
        interface_radius_error_nm=radii_error,
        spectrally_equivalent=spectrally_equivalent,
        structure_recovered_within_tolerance=structure_close,
        alternative_spectrally_equivalent_solution=alternative_equivalent,
    )

    if not args.no_plots:
        configure_plot_style()
        figure_dir = output_dir / "figures"
        for index, _task in enumerate(public_task_id):
            target_label = target_display_label(index)
            plot_target(
                figure_dir / f"{target_label}_unblind_comparison.png",
                target_label=target_label,
                difficulty=str(difficulty[index]),
                wavelengths=wavelengths,
                ref_qsca=reference_qsca[index],
                pred_qsca=prediction_qsca[index],
                ref_qabs=reference_qabs[index],
                pred_qabs=prediction_qabs[index],
                ref_geometry=reference_geometry[index],
                pred_geometry=prediction_geometry[index].astype(np.float64),
                ref_materials=reference_materials[index],
                pred_materials=prediction_materials[index],
                qsca_nmae=float(qsca["nmae"][index]),
            )
        plot_aggregate(figure_dir, comparison, confusion)

    output_artifacts = {
        "per_target_comparison": artifact(comparison_path),
        "by_difficulty_summary": artifact(difficulty_path),
        "material_confusion": artifact(confusion_path),
        "comparison_spectra": artifact(spectra_path),
    }
    summary: Dict[str, Any] = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "created_unix_time": time.time(),
        "wall_time_seconds": time.time() - started,
        "python": platform.python_version(),
        "benchmark_id": benchmark_id,
        "formal_unblind_performed": True,
        "analysis_policy": {
            "stage06a_freeze_verified_before_private_read": True,
            "frozen_snapshot_used_instead_of_mutable_stage05_directory": True,
            "private_reference_hash_verified_against_frozen_manifest": True,
            "optimization_performed": False,
            "material_reranking_performed": False,
            "geometry_adjustment_performed": False,
            "model_inference_performed": False,
            "frozen_rank1_designs_only": True,
            "qsca_is_primary_design_objective": True,
            "qabs_qext_are_posthoc_unoptimized_diagnostics": True,
            "generating_structure_recovery_is_secondary_due_to_inverse_nonuniqueness": True,
        },
        "thresholds": {
            "spectrally_equivalent_nmae": float(args.spectral_success_nmae),
            "spectrally_equivalent_nrmse": float(args.spectral_success_nrmse),
            "spectrally_equivalent_r2": float(args.spectral_success_r2),
            "spectrally_equivalent_peak_error_nm_when_applicable": float(
                args.spectral_success_peak_nm
            ),
            "structure_close_max_parameter_error_nm": float(
                args.structure_close_tolerance_nm
            ),
        },
        "counts": {
            "targets": 10,
            "spectrally_equivalent": int(np.sum(spectrally_equivalent)),
            "exact_material_sequence_recovery": int(np.sum(material_exact)),
            "nearest_integer_geometry_exact": int(np.sum(geometry_grid_exact)),
            "nearest_integer_structure_exact": int(np.sum(structure_grid_exact)),
            "structure_recovered_within_tolerance": int(np.sum(structure_close)),
            "alternative_spectrally_equivalent_solutions": int(np.sum(alternative_equivalent)),
        },
        "aggregate_optical_metrics": {
            "median_qsca_nmae": float(np.median(qsca["nmae"])),
            "mean_qsca_nmae": float(np.mean(qsca["nmae"])),
            "maximum_qsca_nmae": float(np.max(qsca["nmae"])),
            "median_qsca_nrmse": float(np.median(qsca["nrmse"])),
            "minimum_qsca_r2": float(np.min(qsca["r2"])),
            "median_qabs_nmae_ignoring_near_zero": float(np.nanmedian(qabs["nmae"])),
            "median_qext_nmae": float(np.nanmedian(qext["nmae"])),
        },
        "aggregate_structure_metrics": {
            "mean_layer_material_accuracy": float(np.mean(material_match)),
            "median_material_hamming_distance": float(np.median(material_hamming)),
            "median_geometry_parameter_mae_nm": float(
                np.median(np.mean(abs_geometry_error, axis=1))
            ),
            "median_geometry_parameter_max_abs_error_nm": float(
                np.median(geometry_max_error)
            ),
            "median_interface_radius_mae_nm": float(
                np.median(np.mean(abs_radii_error, axis=1))
            ),
            "median_outer_radius_abs_error_nm": float(
                np.median(abs_radii_error[:, -1])
            ),
        },
        "integrity_chain": {
            "preunblind_freeze_file": artifact(freeze_path),
            "preunblind_freeze_file_sha256": freeze_file_sha256,
            "preunblind_freeze_content_sha256": str(freeze["freeze_record_sha256"]),
            "private_reference": artifact(private_path),
            "public_target_snapshot": artifact(public_path),
            "frozen_design_snapshot": artifact(frozen_npz_path),
        },
        "outputs": output_artifacts,
    }
    summary_path = output_dir / "post_unblinding_evaluation_summary.json"
    save_json_atomic(summary_path, summary)

    declaration_path = output_dir / "post_unblinding_evaluation_complete.txt"
    with open(declaration_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"Benchmark ID: {benchmark_id}\n"
            "Stage 06A freeze was verified before hidden-reference access.\n"
            "The ten Stage-05 rank-1 designs were compared without optimization, reranking, or adjustment.\n"
            "Stage 03-05 results must not be modified after this formal unblind.\n"
        )

    # Add the final summary/declaration hashes to a separate completion receipt,
    # avoiding self-referential hashing of the summary itself.
    receipt = {
        "benchmark_id": benchmark_id,
        "stage06a_freeze_content_sha256": str(freeze["freeze_record_sha256"]),
        "private_reference_sha256": sha256_file(private_path),
        "stage06b_summary": artifact(summary_path),
        "unblind_declaration": artifact(declaration_path),
        "completed": True,
    }
    save_json_atomic(output_dir / "unblinding_completion_receipt.json", receipt)

    print("Stage 06B post-hoc unblind comparison PASSED")
    print(f"Benchmark ID: {benchmark_id}")
    print(f"Spectrally equivalent: {int(np.sum(spectrally_equivalent))}/10")
    print(f"Exact material-sequence recovery: {int(np.sum(material_exact))}/10")
    print(
        "Structure recovered within "
        f"{float(args.structure_close_tolerance_nm):g} nm and exact materials: "
        f"{int(np.sum(structure_close))}/10"
    )
    print(
        "Alternative spectrally equivalent solutions: "
        f"{int(np.sum(alternative_equivalent))}/10"
    )
    print(f"Median hidden-reference Qsca NMAE: {float(np.median(qsca['nmae'])):.6f}")
    print(f"Results: {output_dir}")
    print("Formal unblind is complete. Do not rerun or modify Stage 03-05.")


if __name__ == "__main__":
    main()
