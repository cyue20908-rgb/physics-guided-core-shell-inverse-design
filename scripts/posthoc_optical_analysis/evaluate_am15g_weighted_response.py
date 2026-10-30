# -*- coding: utf-8 -*-
"""ASTM G173 AM1.5G-weighted optical-response analysis.

The optional visualization uses a consistent publication plotting style:
- Times New Roman + bold typography
- large landscape single-panel canvas
- inward ticks and subtle horizontal major grids
- borderless legends placed INSIDE the axes
- no panel titles
- PNG + SVG export at manuscript quality

All physical calculations, spectral interpolation, integrations, and reported
metrics are unchanged from the original AM1.5G analysis script.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, LogLocator, MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd

from core_shell_optical_design.electromagnetic_analysis.electromagnetic_analysis_core import (
    DEFAULT_DATASET,
    DEFAULT_FINAL_JSON,
    DEFAULT_FINAL_SPECTRA,
    DEFAULT_OUTPUT_ROOT,
    REFERENCE_DATA_DIR,
    cumulative_trapezoid_normalized,
    load_final_design,
    prepare_output_dir,
    save_json,
    trapezoid,
    write_csv,
)


# =============================================================================
# 1. Physical constants and analysis range
# =============================================================================

PLANCK = 6.62607015e-34
LIGHT_SPEED = 299792458.0
WAVELENGTH_MIN_NM = 400.0
WAVELENGTH_MAX_NM = 1000.0


# =============================================================================
# 2. Publication style — aligned with the uploaded manuscript plotting scripts
# =============================================================================

DPI = 600
FIGSIZE = (7.15, 6.35)

FONT = "Times New Roman"

COLOR_SOLAR = "#E69F00"
COLOR_SCATTERING = "#0072B2"
COLOR_ABSORPTION = "#D55E00"

LINEWIDTH_MAIN = 2.70
LINEWIDTH_SECONDARY = 2.50
SPINE_WIDTH = 1.35
TICK_WIDTH = 1.25


def configure_publication_style() -> None:
    """Match the established single-panel manuscript style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT, "DejaVu Serif"],
            "font.weight": "bold",

            # Bold Times-style mathematical text, consistent with the manuscript.
            "mathtext.fontset": "custom",
            "mathtext.rm": f"{FONT}:bold",
            "mathtext.it": f"{FONT}:bold:italic",
            "mathtext.bf": f"{FONT}:bold",

            # Typography matched to the manuscript-wide plotting style.
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 22,
            "axes.labelweight": "bold",
            "legend.fontsize": 19,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,

            # Axes / ticks.
            "axes.linewidth": SPINE_WIDTH,
            "axes.edgecolor": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.minor.width": 0.95,
            "ytick.minor.width": 0.95,
            "xtick.major.size": 4.8,
            "ytick.major.size": 4.8,
            "xtick.minor.size": 2.6,
            "ytick.minor.size": 2.6,

            # Borderless in-panel legends.
            "legend.frameon": False,

            # High-quality output.
            "savefig.dpi": DPI,
            "figure.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _style_text_on_axis(ax: plt.Axes) -> None:
    """Enforce manuscript font styling on axes, ticks, and labels."""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT)
        label.set_fontweight("bold")
        label.set_color("black")

    ax.xaxis.label.set_fontname(FONT)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontname(FONT)
    ax.yaxis.label.set_fontweight("bold")


def polish_axis(
    ax: plt.Axes,
    *,
    horizontal_grid: bool = True,
    right_ticks: bool = False,
) -> None:
    """Apply the established manuscript axis treatment."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        bottom=True,
        left=not right_ticks,
        top=False,
        right=right_ticks,
        colors="black",
        width=TICK_WIDTH,
        pad=6,
    )

    _style_text_on_axis(ax)

    ax.grid(False)
    if horizontal_grid:
        ax.yaxis.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.70,
            alpha=0.26,
            color="0.55",
            zorder=0,
        )
    ax.set_axisbelow(True)


def stylize_legend(legend) -> None:
    """Borderless bold Times New Roman legend."""
    if legend is None:
        return

    frame = legend.get_frame()
    frame.set_linewidth(0.0)
    frame.set_edgecolor((0, 0, 0, 0))
    frame.set_facecolor((1, 1, 1, 0))

    for text in legend.get_texts():
        text.set_fontname(FONT)
        text.set_fontweight("bold")
        text.set_color("black")


def save_figure_pair(fig: plt.Figure, output_path_prefix: Path) -> None:
    """Save PNG and SVG using one frozen tight bounding box."""
    output_path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*FIGSIZE, forward=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    common_bbox = fig.get_tightbbox(renderer).padded(0.045)

    fig.savefig(
        output_path_prefix.with_suffix(".png"),
        format="png",
        dpi=DPI,
        bbox_inches=common_bbox,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        output_path_prefix.with_suffix(".svg"),
        format="svg",
        bbox_inches=common_bbox,
        facecolor="white",
        edgecolor="none",
    )


def save_figure_pair_fixed_canvas(fig: plt.Figure, output_path_prefix: Path) -> None:
    """Save PNG and SVG on the unchanged FIGSIZE canvas.

    Used for paired panels that require exactly the same
    physical dimensions and aspect ratio. No tight-bbox recropping is applied.
    """
    output_path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*FIGSIZE, forward=True)

    fig.savefig(
        output_path_prefix.with_suffix(".png"),
        format="png",
        dpi=DPI,
        bbox_inches=None,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        output_path_prefix.with_suffix(".svg"),
        format="svg",
        bbox_inches=None,
        facecolor="white",
        edgecolor="none",
    )


def set_linear_axis_without_top_tick(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    lower: float = 0.0,
    nbins: int = 5,
    upper_pad: float = 0.10,
) -> None:
    """Set a readable y range and suppress the top-end tick/label."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot determine y limits from non-finite data.")

    vmax = float(np.max(finite))
    raw_high = max(vmax * (1.0 + upper_pad), lower + 1.0e-12)

    locator = MaxNLocator(nbins=nbins, min_n_ticks=4)
    ticks = np.asarray(locator.tick_values(lower, raw_high), dtype=np.float64)
    ticks = ticks[np.isfinite(ticks)]

    upper_candidates = ticks[ticks >= raw_high]
    axis_high = float(upper_candidates[0]) if upper_candidates.size else raw_high
    if axis_high <= vmax:
        axis_high = raw_high

    visible_ticks = ticks[(ticks >= lower - 1.0e-12) & (ticks < axis_high - 1.0e-12)]

    ax.set_ylim(lower, axis_high)
    if visible_ticks.size >= 2:
        ax.set_yticks(visible_ticks)


# =============================================================================
# 3. Input handling
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the ASTM G173 AM1.5G-weighted optical response."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--final-json", type=Path, default=DEFAULT_FINAL_JSON)
    parser.add_argument("--final-spectra", type=Path, default=DEFAULT_FINAL_SPECTRA)
    parser.add_argument("--am15g", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "solar_weighted_evaluation",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_am15g(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    reference_dir = REFERENCE_DATA_DIR
    candidates = [
        reference_dir / "ASTMG173.csv",
        reference_dir / "ASTMG173(2).csv",
        reference_dir / "ASTMG173 (2).csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(reference_dir.glob("ASTMG173*.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple ASTM G173 files found. Use --am15g to choose one:\n"
            + "\n".join(str(path) for path in matches)
        )

    raise FileNotFoundError(
        f"No ASTM G173 CSV found in {reference_dir}. "
        "Expected ASTMG173.csv or use --am15g."
    )


def load_astm(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=1)
    data.columns = [str(column).strip().lower() for column in data.columns]

    required = ["wavelength", "extraterrestrial", "global", "direct"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"ASTM G173 file is missing columns: {missing}")

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required).sort_values("wavelength").reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("ASTM G173 file contains insufficient valid rows.")

    wavelengths = data["wavelength"].to_numpy(dtype=np.float64)
    if np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("ASTM G173 wavelength column is not strictly increasing.")
    if np.any(data["global"].to_numpy(dtype=np.float64) < 0.0):
        raise ValueError("ASTM G173 global irradiance contains negative values.")

    return data


# =============================================================================
# 4. AM1.5G analysis — calculations intentionally unchanged
# =============================================================================


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    wavelengths: np.ndarray,
) -> float:
    denominator = trapezoid(weights, wavelengths)
    if denominator <= 0.0:
        raise ValueError("Weight integral is not positive.")
    return trapezoid(values * weights, wavelengths) / denominator


def percentile_wavelength(
    wavelengths: np.ndarray,
    cumulative: np.ndarray,
    fraction: float,
) -> float:
    return float(np.interp(fraction, cumulative, wavelengths))


def compute_analysis(
    design: Dict[str, Any],
    astm: pd.DataFrame,
) -> Dict[str, Any]:
    astm_wl = astm["wavelength"].to_numpy(dtype=np.float64)
    global_full = astm["global"].to_numpy(dtype=np.float64)

    low = max(
        WAVELENGTH_MIN_NM,
        float(design["wavelengths_nm"].min()),
        float(astm_wl.min()),
    )
    high = min(
        WAVELENGTH_MAX_NM,
        float(design["wavelengths_nm"].max()),
        float(astm_wl.max()),
    )
    if high <= low:
        raise ValueError("Stage-08 spectrum and ASTM G173 data do not overlap.")

    mask = (astm_wl >= low) & (astm_wl <= high)
    wl = astm_wl[mask]
    irradiance = global_full[mask]

    qsca = np.interp(wl, design["wavelengths_nm"], design["nominal_qsca"])
    qabs = np.interp(wl, design["wavelengths_nm"], design["nominal_qabs"])
    qext = qsca + qabs

    photon_flux = irradiance * (wl * 1e-9) / (PLANCK * LIGHT_SPEED)

    energy_qsca = weighted_mean(qsca, irradiance, wl)
    energy_qabs = weighted_mean(qabs, irradiance, wl)
    energy_qext = weighted_mean(qext, irradiance, wl)

    photon_qsca = weighted_mean(qsca, photon_flux, wl)
    photon_qabs = weighted_mean(qabs, photon_flux, wl)
    photon_qext = weighted_mean(qext, photon_flux, wl)

    radius_m = design["outer_radius_nm"] * 1e-9
    area_m2 = math.pi * radius_m**2
    area_um2 = math.pi * (design["outer_radius_nm"] * 1e-3) ** 2

    csca_m2 = qsca * area_m2
    cabs_m2 = qabs * area_m2
    cext_m2 = qext * area_m2

    scattering_power_density = irradiance * csca_m2
    absorption_power_density = irradiance * cabs_m2
    extinction_power_density = irradiance * cext_m2

    cumulative_scattering = cumulative_trapezoid_normalized(scattering_power_density, wl)
    cumulative_absorption = cumulative_trapezoid_normalized(absorption_power_density, wl)
    cumulative_incident = cumulative_trapezoid_normalized(irradiance, wl)

    full_energy = trapezoid(global_full, astm_wl)
    overlap_energy = trapezoid(irradiance, wl)

    metrics = {
        "integration_min_nm": low,
        "integration_max_nm": high,
        "AM15G_energy_coverage_fraction": overlap_energy / max(full_energy, 1e-300),
        "EnergyWeighted_Qsca": energy_qsca,
        "EnergyWeighted_Qabs": energy_qabs,
        "EnergyWeighted_Qext": energy_qext,
        "EnergyWeighted_Albedo": energy_qsca / max(energy_qext, 1e-300),
        "PhotonWeighted_Qsca": photon_qsca,
        "PhotonWeighted_Qabs": photon_qabs,
        "PhotonWeighted_Qext": photon_qext,
        "PhotonWeighted_Albedo": photon_qsca / max(photon_qext, 1e-300),
        "EnergyWeighted_Csca_um2": energy_qsca * area_um2,
        "EnergyWeighted_Cabs_um2": energy_qabs * area_um2,
        "IntegratedScatteringPower_W_per_particle": trapezoid(scattering_power_density, wl),
        "IntegratedAbsorptionPower_W_per_particle": trapezoid(absorption_power_density, wl),
        "IntegratedExtinctionPower_W_per_particle": trapezoid(extinction_power_density, wl),
        "ScatteringContributionPeakWavelength_nm": float(
            wl[int(np.argmax(scattering_power_density))]
        ),
        "ScatteringLambda50_nm": percentile_wavelength(wl, cumulative_scattering, 0.50),
        "ScatteringLambda90_nm": percentile_wavelength(wl, cumulative_scattering, 0.90),
    }

    return {
        "wavelengths_nm": wl,
        "irradiance": irradiance,
        "photon_flux": photon_flux,
        "qsca": qsca,
        "qabs": qabs,
        "qext": qext,
        "csca_m2": csca_m2,
        "cabs_m2": cabs_m2,
        "scattering_power_density": scattering_power_density,
        "absorption_power_density": absorption_power_density,
        "extinction_power_density": extinction_power_density,
        "cumulative_scattering": cumulative_scattering,
        "cumulative_absorption": cumulative_absorption,
        "cumulative_incident": cumulative_incident,
        "metrics": metrics,
    }


# =============================================================================
# 5. Figure panels
# =============================================================================


def plot_figure(output_dir: Path, analysis: Dict[str, Any]) -> None:
    configure_publication_style()

    wl = np.asarray(analysis["wavelengths_nm"], dtype=np.float64)
    irradiance = np.asarray(analysis["irradiance"], dtype=np.float64)
    qsca = np.asarray(analysis["qsca"], dtype=np.float64)
    metrics = analysis["metrics"]

    # -------------------------------------------------------------------------
    # AM1.5G spectral irradiance and unperturbed Qsca.
    # -------------------------------------------------------------------------
    fig_a, ax_left = plt.subplots(figsize=FIGSIZE)
    ax_right = ax_left.twinx()

    solar_line, = ax_left.plot(
        wl,
        irradiance,
        color=COLOR_SOLAR,
        linewidth=LINEWIDTH_MAIN,
        label="AM1.5G global",
        zorder=3,
    )
    qsca_line, = ax_right.plot(
        wl,
        qsca,
        color=COLOR_SCATTERING,
        linewidth=LINEWIDTH_MAIN,
        label=r"$Q_{\mathrm{sca}}$",
        zorder=4,
    )

    ax_left.set_xlim(float(wl.min()), float(wl.max()))
    ax_left.xaxis.set_major_locator(MultipleLocator(100.0))
    ax_left.set_xlabel("Wavelength (nm)", labelpad=10)
    ax_left.set_ylabel(r"Spectral irradiance (W m$^{-2}$ nm$^{-1}$)", labelpad=10)
    ax_right.set_ylabel(r"Scattering efficiency, $Q_{\mathrm{sca}}$", labelpad=10)

    # Use physically meaningful zero baselines and omit the upper-end tick label.
    set_linear_axis_without_top_tick(
        ax_left,
        irradiance,
        lower=0.0,
        nbins=5,
        upper_pad=0.10,
    )
    set_linear_axis_without_top_tick(
        ax_right,
        qsca,
        lower=0.0,
        nbins=5,
        upper_pad=0.10,
    )

    polish_axis(ax_left, horizontal_grid=True, right_ticks=False)

    # Twin-axis styling: right-side ticks only; no second grid.
    for spine in ax_right.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")
    ax_right.tick_params(
        axis="y",
        which="both",
        direction="in",
        left=False,
        right=True,
        top=False,
        colors="black",
        width=TICK_WIDTH,
        pad=4,
    )
    ax_right.tick_params(axis="x", top=False, bottom=False, labelbottom=False)
    _style_text_on_axis(ax_right)
    ax_right.grid(False)

    # Compact borderless legend INSIDE the upper-right region.
    legend_a = ax_left.legend(
        handles=[solar_line, qsca_line],
        labels=["AM1.5G global", r"$Q_{\mathrm{sca}}$"],
        loc="upper right",
        bbox_to_anchor=(0.965, 0.965),
        frameon=False,
        fancybox=False,
        ncol=1,
        handlelength=2.4,
        handletextpad=0.65,
        borderaxespad=0.0,
        labelspacing=0.38,
    )
    stylize_legend(legend_a)

    # Enlarge the axes region within the same physical canvas.
    fig_a.subplots_adjust(left=0.145, right=0.885, bottom=0.165, top=0.970)

    # Panel label with consistent position and typography.
    fig_a.text(
        0.035,
        0.985,
        "(a)",
        ha="left",
        va="top",
        fontsize=22,
        fontname=FONT,
        fontweight="bold",
        color="black",
    )

    save_figure_pair_fixed_canvas(fig_a, output_dir / "am15g_irradiance_and_scattering_efficiency")
    plt.close(fig_a)

    # -------------------------------------------------------------------------
    # AM1.5G-weighted scattering and absorption spectral power.
    # -------------------------------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=FIGSIZE)

    sca = np.asarray(analysis["scattering_power_density"], dtype=np.float64)
    abs_ = np.asarray(analysis["absorption_power_density"], dtype=np.float64)

    positive = np.concatenate((sca[sca > 0.0], abs_[abs_ > 0.0]))
    floor = max(float(np.min(positive)) * 0.5, 1e-30) if positive.size else 1e-30

    ax_b.semilogy(
        wl,
        np.maximum(sca, floor),
        color=COLOR_SCATTERING,
        linewidth=LINEWIDTH_MAIN,
        label="Scattering",
        zorder=3,
    )
    ax_b.semilogy(
        wl,
        np.maximum(abs_, floor),
        color=COLOR_ABSORPTION,
        linewidth=LINEWIDTH_SECONDARY,
        label="Absorption",
        zorder=3,
    )

    ax_b.set_xlim(float(wl.min()), float(wl.max()))
    ax_b.xaxis.set_major_locator(MultipleLocator(100.0))
    ax_b.set_xlabel("Wavelength (nm)", labelpad=10)
    ax_b.set_ylabel(r"Spectral power (W nm$^{-1}$ particle$^{-1}$)", labelpad=10)

    # Data-driven log range with the upper-end tick omitted.
    max_positive = float(np.max(positive)) if positive.size else 1e-12
    min_positive = float(np.min(positive)) if positive.size else 1e-30

    lower_exp = int(np.floor(np.log10(max(min_positive, 1e-300))))
    upper_exp = int(np.ceil(np.log10(max(max_positive, 1e-300))))

    lower_lim = 10.0 ** lower_exp
    upper_lim = 10.0 ** upper_exp
    if upper_lim <= max_positive:
        upper_exp += 1
        upper_lim = 10.0 ** upper_exp

    ax_b.set_ylim(lower_lim, upper_lim)

    # One labeled major tick every three decades; never label the upper limit.
    first_major_exp = int(np.ceil(lower_exp / 3.0) * 3)
    major_exponents = list(range(first_major_exp, upper_exp, 3))
    major_ticks = [10.0 ** exponent for exponent in major_exponents]
    if not major_ticks:
        major_ticks = [lower_lim]

    ax_b.yaxis.set_major_locator(FixedLocator(major_ticks))
    ax_b.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax_b.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))

    polish_axis(ax_b, horizontal_grid=True, right_ticks=False)

    # Borderless legend INSIDE the lower-left region, away from the main curves.
    legend_b = ax_b.legend(
        loc="lower left",
        bbox_to_anchor=(0.045, 0.055),
        frameon=False,
        fancybox=False,
        ncol=1,
        handlelength=2.4,
        handletextpad=0.65,
        borderaxespad=0.0,
        labelspacing=0.38,
    )
    stylize_legend(legend_b)

    fig_b.subplots_adjust(left=0.170, right=0.975, bottom=0.165, top=0.970)
    save_figure_pair(fig_b, output_dir / "am15g_weighted_scattering_and_absorption")
    plt.close(fig_b)

    # -------------------------------------------------------------------------
    # Cumulative AM1.5G-weighted scattering contribution.
    # -------------------------------------------------------------------------
    fig_c, ax_c = plt.subplots(figsize=FIGSIZE)

    cumulative = np.asarray(analysis["cumulative_scattering"], dtype=np.float64)
    incident = np.asarray(analysis["cumulative_incident"], dtype=np.float64)

    l50 = float(metrics["ScatteringLambda50_nm"])
    l90 = float(metrics["ScatteringLambda90_nm"])

    ax_c.plot(
        wl,
        cumulative,
        color=COLOR_SCATTERING,
        linewidth=LINEWIDTH_MAIN,
        label="Weighted scattering contribution",
        zorder=3,
    )
    ax_c.plot(
        wl,
        incident,
        color=COLOR_SOLAR,
        linewidth=LINEWIDTH_SECONDARY,
        linestyle="--",
        label="Incident solar energy",
        zorder=3,
    )

    # Minimal percentile guides only; no box and no extra metric block.
    ax_c.hlines(0.50, float(wl.min()), l50, color="0.50", lw=1.00, ls=":", zorder=1)
    ax_c.vlines(l50, 0.0, 0.50, color=COLOR_SCATTERING, lw=1.00, ls=":", zorder=1)
    ax_c.hlines(0.90, float(wl.min()), l90, color="0.50", lw=1.00, ls=":", zorder=1)
    ax_c.vlines(l90, 0.0, 0.90, color=COLOR_SCATTERING, lw=1.00, ls=":", zorder=1)

    ax_c.set_xlim(float(wl.min()), float(wl.max()))
    ax_c.set_ylim(0.0, 1.03)
    ax_c.xaxis.set_major_locator(MultipleLocator(100.0))
    ax_c.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])  # deliberately omit 1.0
    ax_c.set_xlabel("Wavelength (nm)", labelpad=10)
    ax_c.set_ylabel("Cumulative fraction", labelpad=10)

    polish_axis(ax_c, horizontal_grid=True, right_ticks=False)

    # Upper-left is the cleanest region for the cumulative curves.
    legend_c = ax_c.legend(
        loc="upper left",
        bbox_to_anchor=(0.035, 0.965),
        frameon=False,
        fancybox=False,
        ncol=1,
        handlelength=2.4,
        handletextpad=0.65,
        borderaxespad=0.0,
        labelspacing=0.38,
    )
    stylize_legend(legend_c)

    # Small, borderless annotations aligned to the same text column.
    annotation_x = 0.56
    ax_c.text(
        annotation_x,
        0.055,
        rf"$\lambda_{{50}}={l50:.0f}\ \mathrm{{nm}}$",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        fontname=FONT,
        fontweight="bold",
        color="black",
        zorder=5,
    )
    ax_c.text(
        annotation_x,
        0.145,
        rf"$\lambda_{{90}}={l90:.0f}\ \mathrm{{nm}}$",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=18,
        fontname=FONT,
        fontweight="bold",
        color="black",
        zorder=5,
    )

    fig_c.subplots_adjust(left=0.145, right=0.950, bottom=0.165, top=0.970)

    # Panel label with the same placement and typography as the paired panel.
    fig_c.text(
        0.035,
        0.985,
        "(b)",
        ha="left",
        va="top",
        fontsize=22,
        fontname=FONT,
        fontweight="bold",
        color="black",
    )

    save_figure_pair_fixed_canvas(fig_c, output_dir / "cumulative_am15g_weighted_scattering")
    plt.close(fig_c)


# =============================================================================
# 6. Main and source-data export
# =============================================================================


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)

    design = load_final_design(args.final_json, args.final_spectra, args.dataset)
    am15g_path = resolve_am15g(args.am15g)
    astm = load_astm(am15g_path)
    analysis = compute_analysis(design, astm)

    plot_figure(args.output_dir, analysis)

    rows = []
    for index, wavelength in enumerate(analysis["wavelengths_nm"]):
        rows.append(
            [
                f"{wavelength:.6f}",
                f"{analysis['irradiance'][index]:.12g}",
                f"{analysis['photon_flux'][index]:.12g}",
                f"{analysis['qsca'][index]:.12g}",
                f"{analysis['qabs'][index]:.12g}",
                f"{analysis['qext'][index]:.12g}",
                f"{analysis['scattering_power_density'][index]:.12g}",
                f"{analysis['absorption_power_density'][index]:.12g}",
                f"{analysis['cumulative_scattering'][index]:.12g}",
                f"{analysis['cumulative_absorption'][index]:.12g}",
            ]
        )

    write_csv(
        args.output_dir / "solar_weighted_spectral_data.csv",
        [
            "wavelength_nm",
            "AM15G_global_W_m2_nm",
            "photon_flux_m2_s_nm",
            "Qsca",
            "Qabs",
            "Qext",
            "scattering_power_W_nm_per_particle",
            "absorption_power_W_nm_per_particle",
            "cumulative_scattering_fraction",
            "cumulative_absorption_fraction",
        ],
        rows,
    )

    metric_rows = [
        [key, f"{value:.12g}"]
        for key, value in analysis["metrics"].items()
    ]
    write_csv(
        args.output_dir / "solar_weighted_metrics.csv",
        ["metric", "value"],
        metric_rows,
    )

    summary = {
        "stage": "solar_weighted_evaluation",
        "status": "passed",
        "AM15G_file": str(am15g_path),
        "AM15G_standard": "ASTM G173-03",
        "AM15G_column": "global",
        "metrics": analysis["metrics"],
        "input_hashes": design["input_hashes"],
        "scope_note": (
            "Metrics describe the isolated-particle optical response over the "
            "400–1000 nm overlap. They are not solar-cell EQE, short-circuit "
            "current, or power-conversion efficiency."
        ),
    }
    save_json(args.output_dir / "solar_weighted_evaluation_summary.json", summary)

    print("AM1.5G-weighted optical-response analysis PASSED")
    print("ASTM file:", am15g_path)
    print(
        "Energy-weighted Qsca:",
        f"{analysis['metrics']['EnergyWeighted_Qsca']:.6f}",
    )
    print(
        "Energy-weighted Qabs:",
        f"{analysis['metrics']['EnergyWeighted_Qabs']:.6e}",
    )
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
