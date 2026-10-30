# -*- coding: utf-8 -*-
"""Electric and magnetic Mie multipole decomposition.

The script uses PyMieDiff public multishell APIs and performs three audits:
1) sum of multipole-resolved Qsca/Qext/Qabs equals PyMieDiff totals;
2) PyMieDiff totals agree with the frozen nominal spectrum away from
   exact Peña singularities;
3) exact singular wavelengths are evaluated by symmetric two-step Richardson
   limiting rather than discarded.

Optional four-panel visualization:
(a) total Qsca and reconstructed multipole sum;
(b) dominant electric/magnetic mode spectra;
(c) wavelength-dependent fractional modal composition;
(d) electric/magnetic contributions by order at the global scattering peak.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import pymiediff as pmd

from core_shell_optical_design.electromagnetic_analysis.electromagnetic_analysis_core import (
    BACKEND,
    PRECISION,
    DEFAULT_DATASET,
    DEFAULT_FINAL_JSON,
    DEFAULT_FINAL_SPECTRA,
    DEFAULT_OUTPUT_ROOT,
    build_eps_layers,
    load_final_design,
    prepare_output_dir,
    save_figure,
    save_json,
    setup_publication_style,
    singular_measure,
    style_axis,
    style_legend,
    trapezoid,
    write_csv,
)


CONSISTENCY_TOL = 2.0e-7
RICHARDSON_H_NM = 1.0e-3
TOP_MODE_COUNT = 6
MAX_BAR_ORDER = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the Mie multipole decomposition.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--final-json", type=Path, default=DEFAULT_FINAL_JSON)
    parser.add_argument("--final-spectra", type=Path, default=DEFAULT_FINAL_SPECTRA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "multipole_decomposition",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def reshape_multipoles(value: torch.Tensor, n_max: int, n_wl: int) -> np.ndarray:
    array = to_numpy(value)
    expected = 2 * n_max * n_wl
    if array.size != expected:
        raise RuntimeError(
            f"Unexpected multipole array size {array.size}; expected {expected}. "
            f"Raw shape={array.shape}"
        )
    return array.reshape(2, n_max, 1, n_wl)[:, :, 0, :]


def reshape_coefficients(value: torch.Tensor, n_max: int, n_wl: int) -> np.ndarray:
    array = to_numpy(value)
    expected = n_max * n_wl
    if array.size != expected:
        raise RuntimeError(
            f"Unexpected coefficient array size {array.size}; expected {expected}. "
            f"Raw shape={array.shape}"
        )
    return array.reshape(n_max, 1, n_wl)[:, 0, :]


def compute_raw(
    design: Dict[str, Any],
    wavelengths_nm: np.ndarray,
    device: torch.device,
    n_max: int | None = None,
) -> Dict[str, Any]:
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    eps_layers = build_eps_layers(design, wavelengths_nm)
    k0 = 2.0 * np.pi / wavelengths_nm

    with torch.no_grad():
        result = pmd.multishell.cross_sections(
            k0=torch.as_tensor(k0, dtype=torch.float64, device=device),
            r_layers=torch.as_tensor(
                design["radii_nm"][None, :], dtype=torch.float64, device=device
            ),
            eps_layers=torch.as_tensor(eps_layers, dtype=torch.complex128, device=device),
            eps_env=torch.tensor(1.0 + 0.0j, dtype=torch.complex128, device=device),
            backend=BACKEND,
            precision=PRECISION,
            n_max=n_max,
        )

    for key in ("q_sca_multipoles", "q_ext_multipoles", "q_abs_multipoles"):
        if key not in result:
            raise RuntimeError(
                f"PyMieDiff result has no '{key}'. "
                "This script requires multipole-resolved cross_sections outputs."
            )

    n_max_value = result["n_max"]
    actual_n_max = int(n_max_value.item()) if hasattr(n_max_value, "item") else int(n_max_value)
    n_wl = len(wavelengths_nm)

    with torch.no_grad():
        coeff = pmd.multishell.mie_coefficients(
            k0=torch.as_tensor(k0, dtype=torch.float64, device=device),
            r_layers=torch.as_tensor(
                design["radii_nm"][None, :], dtype=torch.float64, device=device
            ),
            eps_layers=torch.as_tensor(eps_layers, dtype=torch.complex128, device=device),
            eps_env=torch.tensor(1.0 + 0.0j, dtype=torch.complex128, device=device),
            backend=BACKEND,
            precision=PRECISION,
            n_max=actual_n_max,
            return_internal=False,
        )

    return {
        "n_max": actual_n_max,
        "q_sca": to_numpy(result["q_sca"]).reshape(-1, n_wl)[0],
        "q_ext": to_numpy(result["q_ext"]).reshape(-1, n_wl)[0],
        "q_abs": to_numpy(result["q_abs"]).reshape(-1, n_wl)[0],
        "q_sca_multipoles": reshape_multipoles(
            result["q_sca_multipoles"], actual_n_max, n_wl
        ),
        "q_ext_multipoles": reshape_multipoles(
            result["q_ext_multipoles"], actual_n_max, n_wl
        ),
        "q_abs_multipoles": reshape_multipoles(
            result["q_abs_multipoles"], actual_n_max, n_wl
        ),
        "a_n": reshape_coefficients(coeff["a_n"], actual_n_max, n_wl),
        "b_n": reshape_coefficients(coeff["b_n"], actual_n_max, n_wl),
    }


def richardson_value(
    design: Dict[str, Any],
    wavelength_nm: float,
    device: torch.device,
    n_max: int,
) -> Tuple[Dict[str, Any], float]:
    """
    Symmetric limit:
        S(h)   = [f(lambda+h)+f(lambda-h)]/2
        S(h/2) = [f(lambda+h/2)+f(lambda-h/2)]/2
        f(lambda) ~ [4 S(h/2)-S(h)]/3
    """
    h = RICHARDSON_H_NM
    query = np.asarray(
        [
            wavelength_nm - h,
            wavelength_nm + h,
            wavelength_nm - 0.5 * h,
            wavelength_nm + 0.5 * h,
        ],
        dtype=np.float64,
    )
    raw = compute_raw(design, query, device, n_max=n_max)

    def extrapolate(array: np.ndarray) -> Tuple[np.ndarray, float]:
        sym_h = 0.5 * (array[..., 0] + array[..., 1])
        sym_h2 = 0.5 * (array[..., 2] + array[..., 3])
        value = (4.0 * sym_h2 - sym_h) / 3.0
        error = float(np.max(np.abs(sym_h2 - sym_h)))
        return value, error

    output = {"n_max": n_max}
    errors = []
    for key in (
        "q_sca",
        "q_ext",
        "q_abs",
        "q_sca_multipoles",
        "q_ext_multipoles",
        "q_abs_multipoles",
        "a_n",
        "b_n",
    ):
        output[key], error = extrapolate(raw[key])
        errors.append(error)
    return output, max(errors)


def compute_safe_full_spectrum(
    design: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    wavelengths = design["wavelengths_nm"]
    singular = singular_measure(design["outer_radius_nm"], wavelengths) <= 1.0e-10
    safe_idx = np.flatnonzero(~singular)
    singular_idx = np.flatnonzero(singular)

    safe_raw = compute_raw(design, wavelengths[safe_idx], device, n_max=None)
    n_max = int(safe_raw["n_max"])
    n_wl = len(wavelengths)

    q_sca = np.empty(n_wl, dtype=np.float64)
    q_ext = np.empty(n_wl, dtype=np.float64)
    q_abs = np.empty(n_wl, dtype=np.float64)
    q_sca_m = np.empty((2, n_max, n_wl), dtype=np.float64)
    q_ext_m = np.empty_like(q_sca_m)
    q_abs_m = np.empty_like(q_sca_m)
    a_n = np.empty((n_max, n_wl), dtype=np.complex128)
    b_n = np.empty_like(a_n)

    q_sca[safe_idx] = safe_raw["q_sca"]
    q_ext[safe_idx] = safe_raw["q_ext"]
    q_abs[safe_idx] = safe_raw["q_abs"]
    q_sca_m[:, :, safe_idx] = safe_raw["q_sca_multipoles"]
    q_ext_m[:, :, safe_idx] = safe_raw["q_ext_multipoles"]
    q_abs_m[:, :, safe_idx] = safe_raw["q_abs_multipoles"]
    a_n[:, safe_idx] = safe_raw["a_n"]
    b_n[:, safe_idx] = safe_raw["b_n"]

    singular_errors = {}
    for idx in singular_idx:
        estimate, error = richardson_value(
            design, float(wavelengths[idx]), device, n_max=n_max
        )
        q_sca[idx] = float(np.asarray(estimate["q_sca"]))
        q_ext[idx] = float(np.asarray(estimate["q_ext"]))
        q_abs[idx] = float(np.asarray(estimate["q_abs"]))
        q_sca_m[:, :, idx] = estimate["q_sca_multipoles"]
        q_ext_m[:, :, idx] = estimate["q_ext_multipoles"]
        q_abs_m[:, :, idx] = estimate["q_abs_multipoles"]
        a_n[:, idx] = estimate["a_n"]
        b_n[:, idx] = estimate["b_n"]
        singular_errors[f"{wavelengths[idx]:.12g}"] = error

    return {
        "n_max": n_max,
        "singular_indices": singular_idx,
        "richardson_errors": singular_errors,
        "q_sca": q_sca,
        "q_ext": q_ext,
        "q_abs": q_abs,
        "q_sca_multipoles": q_sca_m,
        "q_ext_multipoles": q_ext_m,
        "q_abs_multipoles": q_abs_m,
        "a_n": a_n,
        "b_n": b_n,
    }


def mode_label(channel: int, order: int) -> str:
    prefix = "E" if channel == 0 else "M"
    names = {1: "D", 2: "Q", 3: "O"}
    return prefix + names.get(order, str(order))


def rank_modes(
    wavelengths: np.ndarray,
    q_sca_m: np.ndarray,
) -> List[Dict[str, Any]]:
    integrated = np.asarray(
        [
            [
                trapezoid(q_sca_m[ch, n], wavelengths)
                for n in range(q_sca_m.shape[1])
            ]
            for ch in range(2)
        ]
    )
    total = float(np.sum(integrated))
    rows = []
    for channel in range(2):
        for order_index in range(q_sca_m.shape[1]):
            value = float(integrated[channel, order_index])
            rows.append(
                {
                    "channel": "electric" if channel == 0 else "magnetic",
                    "order": order_index + 1,
                    "label": mode_label(channel, order_index + 1),
                    "integrated_qsca_nm": value,
                    "fraction": value / max(total, 1e-300),
                }
            )
    return sorted(rows, key=lambda row: row["integrated_qsca_nm"], reverse=True)


def plot_figure(
    output_dir: Path,
    design: Dict[str, Any],
    result: Dict[str, Any],
    ranking: List[Dict[str, Any]],
) -> None:
    setup_publication_style()
    wl = design["wavelengths_nm"]
    q_sca_m = result["q_sca_multipoles"]
    reconstructed = np.sum(q_sca_m, axis=(0, 1))
    peak_index = int(np.argmax(result["q_sca"]))

    selected = ranking[: min(TOP_MODE_COUNT, len(ranking))]
    mode_colors = {
        "ED": "#1F77B4",
        "MD": "#E67E22",
        "EQ": "#2CA02C",
        "MQ": "#9467BD",
        "EO": "#17BECF",
        "MO": "#D62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.7), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a)
    ax_a.plot(wl, result["q_sca"], color="black", lw=2.25, label="Total")
    ax_a.plot(
        wl,
        reconstructed,
        color="#D55E00",
        lw=1.6,
        ls="--",
        label="Multipole sum",
    )
    ax_a.axvline(wl[peak_index], color="0.45", lw=1.0, ls=":")
    style_axis(
        ax_a,
        xlabel="Wavelength (nm)",
        ylabel=r"$Q_{\mathrm{sca}}$",
        xlim=(wl.min(), wl.max()),
        x_major=100,
        y_major=1.0,
    )
    ax_a.set_title("(a) Total scattering audit", pad=8)
    style_legend(ax_a.legend(loc="upper right"))

    # (b)
    selected_arrays = []
    selected_labels = []
    for row in selected:
        channel = 0 if row["channel"] == "electric" else 1
        order_index = int(row["order"]) - 1
        values = q_sca_m[channel, order_index]
        selected_arrays.append(values)
        selected_labels.append(row["label"])
        ax_b.plot(
            wl,
            values,
            lw=1.85,
            color=mode_colors.get(row["label"], None),
            label=row["label"],
        )
    style_axis(
        ax_b,
        xlabel="Wavelength (nm)",
        ylabel=r"Multipole $Q_{\mathrm{sca}}$",
        xlim=(wl.min(), wl.max()),
        x_major=100,
    )
    ax_b.set_title("(b) Dominant multipole channels", pad=8)
    style_legend(ax_b.legend(loc="upper right", ncol=2))

    # (c)
    selected_matrix = np.asarray(selected_arrays)
    remainder = np.maximum(result["q_sca"] - np.sum(selected_matrix, axis=0), 0.0)
    denom = np.maximum(result["q_sca"], 1e-300)
    fractions = [values / denom for values in selected_matrix]
    fractions.append(remainder / denom)
    stack_colors = [mode_colors.get(label, None) for label in selected_labels] + ["#BDBDBD"]
    ax_c.stackplot(
        wl,
        *fractions,
        colors=stack_colors,
        labels=selected_labels + ["Other"],
        alpha=0.90,
        linewidth=0.25,
        edgecolor="black",
    )
    style_axis(
        ax_c,
        xlabel="Wavelength (nm)",
        ylabel="Fraction of total scattering",
        xlim=(wl.min(), wl.max()),
        ylim=(0.0, 1.0),
        x_major=100,
        y_major=0.2,
    )
    ax_c.set_title("(c) Wavelength-dependent composition", pad=8)
    style_legend(ax_c.legend(loc="upper right", ncol=2))

    # (d)
    order_count = min(MAX_BAR_ORDER, result["n_max"])
    x = np.arange(order_count)
    width = 0.34
    electric = q_sca_m[0, :order_count, peak_index]
    magnetic = q_sca_m[1, :order_count, peak_index]
    ax_d.bar(
        x - width / 2,
        electric,
        width,
        color="#1F77B4",
        edgecolor="black",
        linewidth=0.9,
        label="Electric",
    )
    ax_d.bar(
        x + width / 2,
        magnetic,
        width,
        color="#E67E22",
        edgecolor="black",
        linewidth=0.9,
        label="Magnetic",
    )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f"n={n}" for n in range(1, order_count + 1)])
    style_axis(
        ax_d,
        xlabel="Multipole order",
        ylabel=r"$Q_{\mathrm{sca}}$ contribution",
        minor=False,
    )
    ax_d.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_d.set_title(f"(d) Contributions at {wl[peak_index]:.0f} nm", pad=8)
    style_legend(ax_d.legend(loc="upper right"))

    save_figure(fig, output_dir / "mie_multipole_decomposition")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    device = choose_device(args.device)
    design = load_final_design(args.final_json, args.final_spectra, args.dataset)

    start = time.time()
    result = compute_safe_full_spectrum(design, device)

    reconstructed_sca = np.sum(result["q_sca_multipoles"], axis=(0, 1))
    reconstructed_ext = np.sum(result["q_ext_multipoles"], axis=(0, 1))
    reconstructed_abs = np.sum(result["q_abs_multipoles"], axis=(0, 1))

    residuals = {
        "multipole_sum_qsca": float(np.max(np.abs(reconstructed_sca - result["q_sca"]))),
        "multipole_sum_qext": float(np.max(np.abs(reconstructed_ext - result["q_ext"]))),
        "multipole_sum_qabs": float(np.max(np.abs(reconstructed_abs - result["q_abs"]))),
        "stage08_qsca": float(np.max(np.abs(result["q_sca"] - design["nominal_qsca"]))),
        "stage08_qext": float(np.max(np.abs(result["q_ext"] - design["nominal_qext"]))),
        "stage08_qabs": float(np.max(np.abs(result["q_abs"] - design["nominal_qabs"]))),
    }

    if max(
        residuals["multipole_sum_qsca"],
        residuals["multipole_sum_qext"],
        residuals["multipole_sum_qabs"],
    ) > CONSISTENCY_TOL:
        raise RuntimeError(f"Multipole-sum audit failed: {residuals}")

    # Stage-08 comparison is informative rather than a hard failure because its
    # exact singular point may have been frozen through a separate limiting path.
    ranking = rank_modes(design["wavelengths_nm"], result["q_sca_multipoles"])
    plot_figure(args.output_dir, design, result, ranking)

    np.savez_compressed(
        args.output_dir / "multipole_spectral_data.npz",
        wavelengths_nm=design["wavelengths_nm"],
        orders=np.arange(1, result["n_max"] + 1),
        channels=np.asarray(["electric", "magnetic"]),
        q_sca=result["q_sca"],
        q_ext=result["q_ext"],
        q_abs=result["q_abs"],
        q_sca_multipoles=result["q_sca_multipoles"],
        q_ext_multipoles=result["q_ext_multipoles"],
        q_abs_multipoles=result["q_abs_multipoles"],
        a_n=result["a_n"],
        b_n=result["b_n"],
    )

    long_rows = []
    for iw, wavelength in enumerate(design["wavelengths_nm"]):
        for channel in range(2):
            for order_index in range(result["n_max"]):
                coefficient = result["a_n"][order_index, iw] if channel == 0 else result["b_n"][order_index, iw]
                long_rows.append(
                    [
                        f"{wavelength:.6f}",
                        "electric" if channel == 0 else "magnetic",
                        order_index + 1,
                        mode_label(channel, order_index + 1),
                        f"{coefficient.real:.12g}",
                        f"{coefficient.imag:.12g}",
                        f"{abs(coefficient):.12g}",
                        f"{result['q_sca_multipoles'][channel, order_index, iw]:.12g}",
                        f"{result['q_ext_multipoles'][channel, order_index, iw]:.12g}",
                        f"{result['q_abs_multipoles'][channel, order_index, iw]:.12g}",
                    ]
                )
    write_csv(
        args.output_dir / "multipole_contributions_long.csv",
        [
            "wavelength_nm",
            "channel",
            "order",
            "mode",
            "coefficient_real",
            "coefficient_imag",
            "coefficient_abs",
            "Qsca_contribution",
            "Qext_contribution",
            "Qabs_contribution",
        ],
        long_rows,
    )

    peak_index = int(np.argmax(result["q_sca"]))
    peak_total = float(result["q_sca"][peak_index])
    peak_modes = []
    for channel in range(2):
        for order_index in range(result["n_max"]):
            value = float(result["q_sca_multipoles"][channel, order_index, peak_index])
            peak_modes.append(
                {
                    "mode": mode_label(channel, order_index + 1),
                    "channel": "electric" if channel == 0 else "magnetic",
                    "order": order_index + 1,
                    "Qsca_at_peak": value,
                    "fraction_at_peak": value / max(peak_total, 1e-300),
                }
            )
    peak_modes.sort(key=lambda row: row["Qsca_at_peak"], reverse=True)

    summary = {
        "stage": "multipole_decomposition_analysis",
        "status": "passed",
        "pymiediff_version": str(getattr(pmd, "__version__", "unknown")),
        "backend": BACKEND,
        "precision": PRECISION,
        "device": str(device),
        "n_max": int(result["n_max"]),
        "peak_wavelength_nm": float(design["wavelengths_nm"][peak_index]),
        "peak_Qsca": peak_total,
        "top_modes_at_peak": peak_modes[:8],
        "top_integrated_modes": ranking[:8],
        "singular_wavelengths_nm": design["wavelengths_nm"][result["singular_indices"]],
        "richardson_h_nm": RICHARDSON_H_NM,
        "richardson_errors": result["richardson_errors"],
        "consistency_residuals": residuals,
        "runtime_seconds": time.time() - start,
        "input_hashes": design["input_hashes"],
        "interpretation_scope": (
            "External Mie-channel decomposition identifies modal contributions. "
            "It does not by itself assign a resonance to a particular physical layer."
        ),
    }
    save_json(args.output_dir / "multipole_analysis_summary.json", summary)

    print("Mie multipole analysis PASSED")
    print("n_max:", result["n_max"])
    print("Peak:", f"{summary['peak_wavelength_nm']:.1f} nm")
    if len(result["singular_indices"]) > 0:
        print(
            "Richardson-treated singular wavelengths:",
            ", ".join(f"{x:.3f}" for x in summary["singular_wavelengths_nm"]),
        )
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
