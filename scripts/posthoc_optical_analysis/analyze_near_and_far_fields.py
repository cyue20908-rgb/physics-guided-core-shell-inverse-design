# -*- coding: utf-8 -*-
"""Far-field directionality and near-field intensity analysis.

Main figure:
    top row: normalized unpolarized angular scattering at three automatically
             selected broadband high-scattering wavelengths
    bottom row: normalized electric-field intensity |E|^2/|E0|^2 at the same
                wavelengths using one common color scale

Supplement:
    magnetic-field intensity maps at the same wavelengths

The representative wavelengths are selected deterministically from three
spectral bands (400–550, 550–750, 750–1000 nm), excluding exact Peña singular
points.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import NullLocator
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
    select_representative_wavelengths,
    setup_publication_style,
    singular_measure,
    style_axis,
    style_legend,
    trapezoid,
    write_csv,
)


N_ANGLES = 361
NEARFIELD_GRID_N = 301
NEARFIELD_DOMAIN_FACTOR = 1.65
NEARFIELD_Y_OFFSET_NM = 1.0
NEARFIELD_CHUNK_SIZE = 18000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze far-field scattering and near-field intensity.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--final-json", type=Path, default=DEFAULT_FINAL_JSON)
    parser.add_argument("--final-spectra", type=Path, default=DEFAULT_FINAL_SPECTRA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "near_and_far_field_analysis",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--grid-n", type=int, default=NEARFIELD_GRID_N)
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


def compute_angular(
    design: Dict[str, Any],
    wavelengths_nm: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    theta = np.linspace(0.0, np.pi, N_ANGLES, dtype=np.float64)
    eps_layers = build_eps_layers(design, wavelengths_nm)
    k0 = 2.0 * np.pi / wavelengths_nm

    if np.any(singular_measure(design["outer_radius_nm"], wavelengths_nm) <= 1.0e-8):
        raise RuntimeError("A selected angular-scattering wavelength is singular-unsafe.")

    with torch.no_grad():
        result = pmd.multishell.angular_scattering(
            k0=torch.as_tensor(k0, dtype=torch.float64, device=device),
            theta=torch.as_tensor(theta, dtype=torch.float64, device=device),
            r_layers=torch.as_tensor(
                design["radii_nm"][None, :], dtype=torch.float64, device=device
            ),
            eps_layers=torch.as_tensor(eps_layers, dtype=torch.complex128, device=device),
            eps_env=torch.tensor(1.0 + 0.0j, dtype=torch.complex128, device=device),
            backend=BACKEND,
            precision=PRECISION,
            n_max=None,
        )

    i_unpol = to_numpy(result["i_unpol"]).reshape(len(wavelengths_nm), len(theta))
    if not np.isfinite(i_unpol).all() or np.min(i_unpol) < -1.0e-9:
        raise RuntimeError("Angular intensity contains invalid values.")
    i_unpol = np.maximum(i_unpol, 0.0)
    i_norm = i_unpol / np.maximum(np.max(i_unpol, axis=1, keepdims=True), 1.0e-300)

    g_values = []
    fb_ratios = []
    for values in i_unpol:
        weight = np.sin(theta)
        denominator = trapezoid(values * weight, theta)
        g = trapezoid(values * np.cos(theta) * weight, theta) / max(denominator, 1e-300)

        forward_mask = theta <= (0.5 * np.pi)
        backward_mask = theta >= (0.5 * np.pi)
        forward = trapezoid(values[forward_mask] * weight[forward_mask], theta[forward_mask])
        backward = trapezoid(values[backward_mask] * weight[backward_mask], theta[backward_mask])
        g_values.append(float(g))
        fb_ratios.append(float(forward / max(backward, 1e-300)))

    return {
        "theta_rad": theta,
        "theta_deg": np.degrees(theta),
        "i_unpolarized": i_unpol,
        "i_unpolarized_norm": i_norm,
        "asymmetry_g": np.asarray(g_values, dtype=np.float64),
        "forward_backward_ratio": np.asarray(fb_ratios, dtype=np.float64),
    }


def compute_nearfield_one(
    design: Dict[str, Any],
    wavelength_nm: float,
    device: torch.device,
    grid_n: int,
) -> Dict[str, Any]:
    outer = design["outer_radius_nm"]
    if singular_measure(outer, np.asarray([wavelength_nm]))[0] <= 1.0e-8:
        raise RuntimeError(f"Near-field wavelength {wavelength_nm:g} nm is singular-unsafe.")

    extent = NEARFIELD_DOMAIN_FACTOR * outer
    axis = np.linspace(-extent, extent, grid_n, dtype=np.float64)
    x_grid, z_grid = np.meshgrid(axis, axis, indexing="xy")
    y_grid = np.full_like(x_grid, NEARFIELD_Y_OFFSET_NM)
    probe = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))

    eps_layers = build_eps_layers(design, np.asarray([wavelength_nm], dtype=np.float64))
    k0 = np.asarray([2.0 * np.pi / wavelength_nm], dtype=np.float64)

    r_layers_t = torch.as_tensor(
        design["radii_nm"][None, :], dtype=torch.float64, device=device
    )
    eps_layers_t = torch.as_tensor(eps_layers, dtype=torch.complex128, device=device)
    k0_t = torch.as_tensor(k0, dtype=torch.float64, device=device)
    eps_env_t = torch.tensor(1.0 + 0.0j, dtype=torch.complex128, device=device)

    e_total_parts: List[np.ndarray] = []
    h_total_parts: List[np.ndarray] = []
    e_incident_parts: List[np.ndarray] = []
    h_incident_parts: List[np.ndarray] = []

    for start in range(0, len(probe), NEARFIELD_CHUNK_SIZE):
        end = min(start + NEARFIELD_CHUNK_SIZE, len(probe))
        with torch.no_grad():
            fields = pmd.multishell.nearfields(
                k0=k0_t,
                r_probe=torch.as_tensor(
                    probe[start:end], dtype=torch.float64, device=device
                ),
                r_layers=r_layers_t,
                eps_layers=eps_layers_t,
                eps_env=eps_env_t,
                E_0=1.0,
                backend=BACKEND,
                precision=PRECISION,
                n_max=None,
            )
        e_total_parts.append(to_numpy(fields["E_t"]).reshape(end - start, 3))
        h_total_parts.append(to_numpy(fields["H_t"]).reshape(end - start, 3))
        e_incident_parts.append(to_numpy(fields["E_i"]).reshape(end - start, 3))
        h_incident_parts.append(to_numpy(fields["H_i"]).reshape(end - start, 3))

    shape = (grid_n, grid_n, 3)
    e_total = np.concatenate(e_total_parts, axis=0).reshape(shape)
    h_total = np.concatenate(h_total_parts, axis=0).reshape(shape)
    e_incident = np.concatenate(e_incident_parts, axis=0).reshape(shape)
    h_incident = np.concatenate(h_incident_parts, axis=0).reshape(shape)

    e0 = float(np.nanmedian(np.linalg.norm(e_incident.reshape(-1, 3), axis=1)))
    h0 = float(np.nanmedian(np.linalg.norm(h_incident.reshape(-1, 3), axis=1)))
    e0 = max(e0, 1e-300)
    h0 = max(h0, 1e-300)

    e_intensity = np.sum(np.abs(e_total / e0) ** 2, axis=-1)
    h_intensity = np.sum(np.abs(h_total / h0) ** 2, axis=-1)

    if not np.isfinite(e_intensity).all() or not np.isfinite(h_intensity).all():
        raise RuntimeError("Near-field map contains NaN/Inf.")

    return {
        "wavelength_nm": float(wavelength_nm),
        "axis_nm": axis,
        "x_nm": x_grid,
        "z_nm": z_grid,
        "E_intensity": e_intensity,
        "H_intensity": h_intensity,
        "E0_abs": e0,
        "H0_abs": h0,
    }


def add_boundaries(ax, radii_nm: np.ndarray) -> None:
    for radius in radii_nm:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                float(radius),
                fill=False,
                edgecolor="white",
                linewidth=0.85,
                alpha=0.95,
            )
        )


def plot_main_figure(
    output_dir: Path,
    design: Dict[str, Any],
    wavelengths_nm: np.ndarray,
    angular: Dict[str, np.ndarray],
    nearfields: List[Dict[str, Any]],
) -> None:
    setup_publication_style()

    fig = plt.figure(figsize=(12.6, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.92, 1.08],
        hspace=0.13,
        wspace=0.10,
    )

    # Top row: three polar angular patterns.
    for index, wavelength in enumerate(wavelengths_nm):
        ax = fig.add_subplot(gs[0, index], projection="polar")
        theta = angular["theta_rad"]
        values = angular["i_unpolarized_norm"][index]

        # Mirror the physical 0–pi scattering pattern for a conventional full polar view.
        theta_full = np.concatenate((theta, 2.0 * np.pi - theta[-2:0:-1]))
        values_full = np.concatenate((values, values[-2:0:-1]))
        ax.plot(theta_full, values_full, color="#1479B8", lw=2.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_rticks([0.5, 1.0])
        ax.set_rlabel_position(135)
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
        ax.grid(True, lw=0.55, alpha=0.38, color="0.45")
        ax.set_title(
            f"({chr(97 + index)}) {wavelength:.0f} nm",
            pad=11,
            fontweight="bold",
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
            label.set_color("black")

    # Bottom row: E-field maps, common color scale.
    all_values = np.concatenate([item["E_intensity"].ravel() for item in nearfields])
    vmax = float(np.percentile(all_values, 99.5))
    vmax = max(vmax, 1e-12)
    images = []

    for index, (wavelength, near) in enumerate(zip(wavelengths_nm, nearfields)):
        ax = fig.add_subplot(gs[1, index])
        axis = near["axis_nm"]
        extent = [axis.min(), axis.max(), axis.min(), axis.max()]
        image = ax.imshow(
            near["E_intensity"],
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
        )
        images.append(image)
        add_boundaries(ax, design["radii_nm"])
        ax.set_aspect("equal")
        style_axis(
            ax,
            xlabel="x (nm)",
            ylabel="z (nm)" if index == 0 else None,
            x_major=300,
            y_major=300,
            minor=False,
        )
        if index > 0:
            ax.set_yticklabels([])
        ax.set_title(
            f"({chr(100 + index)}) $|E|^2/|E_0|^2$ at {wavelength:.0f} nm",
            pad=8,
        )

    cbar = fig.colorbar(
        images[-1],
        ax=[fig.axes[3], fig.axes[4], fig.axes[5]],
        fraction=0.025,
        pad=0.018,
        shrink=0.90,
    )
    cbar.set_label(r"$|E|^2/|E_0|^2$", fontweight="bold")
    cbar.ax.tick_params(direction="in", width=1.0, labelsize=9.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    save_figure(fig, output_dir / "far_field_and_near_field_analysis")
    plt.close(fig)


def plot_h_supplement(
    output_dir: Path,
    design: Dict[str, Any],
    wavelengths_nm: np.ndarray,
    nearfields: List[Dict[str, Any]],
) -> None:
    setup_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), constrained_layout=True)

    all_values = np.concatenate([item["H_intensity"].ravel() for item in nearfields])
    vmax = max(float(np.percentile(all_values, 99.5)), 1e-12)

    last = None
    for index, (ax, wavelength, near) in enumerate(zip(axes, wavelengths_nm, nearfields)):
        axis = near["axis_nm"]
        extent = [axis.min(), axis.max(), axis.min(), axis.max()]
        last = ax.imshow(
            near["H_intensity"],
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
        )
        add_boundaries(ax, design["radii_nm"])
        ax.set_aspect("equal")
        style_axis(
            ax,
            xlabel="x (nm)",
            ylabel="z (nm)" if index == 0 else None,
            x_major=300,
            y_major=300,
            minor=False,
        )
        if index > 0:
            ax.set_yticklabels([])
        ax.set_title(f"({chr(97 + index)}) {wavelength:.0f} nm", pad=8)

    cbar = fig.colorbar(last, ax=axes.tolist(), fraction=0.028, pad=0.018, shrink=0.88)
    cbar.set_label(r"$|H|^2/|H_0|^2$", fontweight="bold")
    cbar.ax.tick_params(direction="in", width=1.0, labelsize=9.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    save_figure(fig, output_dir / "magnetic_near_field_intensity")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.grid_n < 101 or args.grid_n % 2 == 0:
        raise ValueError("--grid-n must be an odd integer >= 101.")

    prepare_output_dir(args.output_dir, args.overwrite)
    device = choose_device(args.device)
    design = load_final_design(args.final_json, args.final_spectra, args.dataset)
    wavelengths = select_representative_wavelengths(design)

    start_time = time.time()
    angular = compute_angular(design, wavelengths, device)
    nearfields = [
        compute_nearfield_one(design, float(wl), device, args.grid_n)
        for wl in wavelengths
    ]

    plot_main_figure(args.output_dir, design, wavelengths, angular, nearfields)
    plot_h_supplement(args.output_dir, design, wavelengths, nearfields)

    # Source data
    angular_rows = []
    for i, wl in enumerate(wavelengths):
        for j, theta_deg in enumerate(angular["theta_deg"]):
            angular_rows.append(
                [
                    f"{wl:.6f}",
                    f"{theta_deg:.6f}",
                    f"{angular['i_unpolarized'][i, j]:.12g}",
                    f"{angular['i_unpolarized_norm'][i, j]:.12g}",
                ]
            )
    write_csv(
        args.output_dir / "far_field_angular_scattering.csv",
        ["wavelength_nm", "theta_deg", "I_unpolarized", "I_unpolarized_normalized"],
        angular_rows,
    )

    np.savez_compressed(
        args.output_dir / "near_field_intensity_maps.npz",
        wavelengths_nm=wavelengths,
        radii_nm=design["radii_nm"],
        x_nm=np.stack([item["x_nm"] for item in nearfields]),
        z_nm=np.stack([item["z_nm"] for item in nearfields]),
        E_intensity=np.stack([item["E_intensity"] for item in nearfields]),
        H_intensity=np.stack([item["H_intensity"] for item in nearfields]),
    )

    summary = {
        "stage": "near_and_far_field_analysis",
        "status": "passed",
        "pymiediff_version": str(getattr(pmd, "__version__", "unknown")),
        "backend": BACKEND,
        "precision": PRECISION,
        "device": str(device),
        "representative_wavelengths_nm": wavelengths,
        "selection_rule": (
            "maximum nominal Qsca in each of 400–550, 550–750 and 750–1000 nm; "
            "exact Pena singular points excluded"
        ),
        "asymmetry_g": angular["asymmetry_g"],
        "forward_backward_ratio": angular["forward_backward_ratio"],
        "nearfield_grid_n": int(args.grid_n),
        "nearfield_domain_factor": NEARFIELD_DOMAIN_FACTOR,
        "nearfield_y_offset_nm": NEARFIELD_Y_OFFSET_NM,
        "runtime_seconds": time.time() - start_time,
        "input_hashes": design["input_hashes"],
    }
    save_json(args.output_dir / "electromagnetic_field_analysis_summary.json", summary)

    print("Far-field and near-field analysis PASSED")
    print("Representative wavelengths:", ", ".join(f"{x:.0f} nm" for x in wavelengths))
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
