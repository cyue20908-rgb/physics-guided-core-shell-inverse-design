# -*- coding: utf-8 -*-
"""Shared utilities for Stage 08 goal-free robust nanoparticle design.

This module deliberately writes to a new result root and never reads Stage 03-06
blind candidates. It reuses only frozen forward physics assets, the Stage-01
legal material registry, and (optionally) the frozen Stage-02 inverse network as
an initialization source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch

from core_shell_optical_design.inverse_design.inverse_design_data import (
    DEFAULT_DATASET,
    DEFAULT_PGFF_DIR,
    DEFAULT_RESULT_ROOT,
    artifact,
    resolve_device,
    save_json_atomic,
    save_npz_atomic,
    set_seed,
    sha256_file,
)
from core_shell_optical_design.inverse_design.forward_model_bridge import (
    pymiediff_forward,
)
from core_shell_optical_design.inverse_design.geometry_parameterization import (
    geometry_bounds_mask,
    material_sequence_mask,
)
from core_shell_optical_design.project_paths import (
    INVERSE_MODEL_DIR,
    INVERSE_TRAINING_DATA,
    REPOSITORY_ROOT,
    REPRODUCED_ROBUST_RESULTS_DIR,
)

PROJECT_ROOT = REPOSITORY_ROOT
RESULT_ROOT = REPRODUCED_ROBUST_RESULTS_DIR
PREPARED_DATA = INVERSE_TRAINING_DATA
STAGE02_DIR = INVERSE_MODEL_DIR

ROBUST_LOWER = np.asarray([121.0, 21.0, 21.0, 21.0], dtype=np.float64)
ROBUST_UPPER = np.asarray([219.0, 79.0, 79.0, 79.0], dtype=np.float64)
ROBUST_MAX_OUTER = 296.0
CONTINUOUS_MAX_OUTER = 295.5
ORIGINAL_LOWER = np.asarray([120.0, 20.0, 20.0, 20.0], dtype=np.float64)
ORIGINAL_UPPER = np.asarray([220.0, 80.0, 80.0, 80.0], dtype=np.float64)
ORIGINAL_MAX_OUTER = 300.0
GEOMETRY_COLUMNS = ("r_core", "t1", "t2", "t3")
MATERIAL_COLUMNS = ("m0", "m1", "m2", "m3")

ABSORPTION_TIERS: tuple[dict[str, float | str], ...] = (
    {
        "name": "ultra_low",
        "band_absorption_fraction_max": 0.005,
        "absorption_fraction_p95_max": 0.010,
        "qabs_max_max": 0.030,
    },
    {
        "name": "low",
        "band_absorption_fraction_max": 0.010,
        "absorption_fraction_p95_max": 0.020,
        "qabs_max_max": 0.050,
    },
    {
        "name": "relaxed_low",
        "band_absorption_fraction_max": 0.020,
        "absorption_fraction_p95_max": 0.050,
        "qabs_max_max": 0.100,
    },
)

METRIC_COLUMNS = (
    "tail_mean_qsca",
    "qsca_p10",
    "mean_qsca",
    "min_qsca",
    "max_qsca",
    "qsca_cv",
    "mean_qabs",
    "qabs_p95",
    "max_qabs",
    "band_absorption_fraction",
    "absorption_fraction_p95",
    "single_scattering_albedo",
    "mean_csca_nm2",
    "energy_residual",
    "minimum_efficiency",
    "minimum_abs_sin",
    "solver_regularized",
    "singularity_limit_error",
    "solver_valid",
    "physics_valid",
)


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def prepare_output_dir(path: Path, *, resume: bool, overwrite: bool, summary_name: str) -> None:
    path = Path(path)
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        summary = path / summary_name
        if summary.is_file() and load_json(summary).get("status") == "passed":
            raise FileExistsError(
                f"Stage is already completed: {summary}\n"
                "Use --overwrite only for a deliberate rebuild."
            )
        if not resume:
            raise FileExistsError(
                f"Output directory is non-empty: {path}\n"
                "Use --resume for a partial run or --overwrite to restart."
            )
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)


def load_physics_assets(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Forward physics asset not found: {path}")
    required = ("wavelengths", "material_names", "material_n", "material_k")
    with np.load(path, allow_pickle=True) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Forward asset is missing fields: {missing}")
        wavelengths = np.asarray(archive["wavelengths"], dtype=np.float64)
        names = np.asarray(archive["material_names"], dtype=str)
        material_n = np.asarray(archive["material_n"], dtype=np.float64)
        material_k = np.asarray(archive["material_k"], dtype=np.float64)
    if wavelengths.ndim != 1 or len(wavelengths) < 2 or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("Wavelength grid must be one-dimensional and strictly increasing.")
    if material_n.shape != material_k.shape or material_n.shape != (len(names), len(wavelengths)):
        raise ValueError("Material optical tables have incompatible dimensions.")
    if not np.isfinite(material_n).all() or not np.isfinite(material_k).all():
        raise ValueError("Material optical tables contain NaN or Inf.")
    if float(material_k.min()) < -1.0e-12:
        raise ValueError("material_k contains materially negative values.")
    return {
        "path": path,
        "wavelengths": wavelengths,
        "material_names": names,
        "material_n": material_n,
        "material_k": material_k,
    }


def load_prepared_registry(path: Path, physics: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    path = Path(path)
    required = (
        "wavelengths_nm",
        "material_names",
        "legal_material_sequences",
        "qsca_mean_train",
        "qsca_std_train",
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Prepared tandem data is missing fields: {missing}")
        output = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if not np.array_equal(
        np.asarray(output["wavelengths_nm"], dtype=np.float64),
        np.asarray(physics["wavelengths"], dtype=np.float64),
    ):
        raise RuntimeError("Prepared-data and physics wavelength grids differ.")
    if not np.array_equal(
        np.asarray(output["material_names"]).astype(str),
        np.asarray(physics["material_names"]).astype(str),
    ):
        raise RuntimeError("Prepared-data and physics material registries differ.")
    sequences = np.asarray(output["legal_material_sequences"], dtype=np.int64)
    if sequences.shape != (400, 4) or len(np.unique(sequences, axis=0)) != 400:
        raise RuntimeError(f"Expected exactly 400 unique legal sequences, received {sequences.shape}.")
    if not np.all(material_sequence_mask(sequences, physics["material_names"])):
        raise RuntimeError("Prepared material registry contains an illegal sequence.")
    return output


def verify_frozen_assets(
    *,
    dataset: Path,
    prepared_data: Path,
    stage02_dir: Path = STAGE02_DIR,
    pgff_dir: Path = DEFAULT_PGFF_DIR,
) -> Dict[str, Any]:
    """Verify frozen assets without reading Stage 03-06 candidate results."""
    summary_path = Path(stage02_dir) / "training_summary.json"
    inverse_path = Path(stage02_dir) / "best_model.pt"
    pgff_path = Path(pgff_dir) / "best_model.pt"
    summary = load_json(summary_path)
    if summary.get("status") != "passed":
        raise RuntimeError("Stage 02 training summary is not marked passed.")
    if not inverse_path.is_file() or not pgff_path.is_file():
        raise FileNotFoundError("Frozen inverse or PGFF checkpoint is missing.")
    outputs = summary.get("outputs", {})
    saved_inverse = outputs.get("best_model", {}).get("sha256")
    if saved_inverse and saved_inverse != sha256_file(inverse_path):
        raise RuntimeError("Stage-02 inverse checkpoint hash differs from training summary.")
    inputs = summary.get("inputs", {})
    saved_prepared = inputs.get("prepared_data", {}).get("sha256")
    if saved_prepared and saved_prepared != sha256_file(prepared_data):
        raise RuntimeError("Prepared-data hash differs from Stage-02 provenance.")
    saved_pgff = inputs.get("pgff_checkpoint", {}).get("sha256")
    if saved_pgff and saved_pgff != sha256_file(pgff_path):
        raise RuntimeError("PGFF checkpoint hash differs from Stage-02 provenance.")
    return {
        "dataset": artifact(dataset),
        "prepared_data": artifact(prepared_data),
        "stage02_training_summary": artifact(summary_path),
        "stage02_inverse_checkpoint": artifact(inverse_path),
        "pgff_checkpoint": artifact(pgff_path),
        "stage02_summary": summary,
    }


def robust_geometry_mask(geometry: np.ndarray, atol: float = 1.0e-8) -> np.ndarray:
    g = np.asarray(geometry, dtype=np.float64)
    if g.ndim == 1:
        g = g[None, :]
    if g.ndim != 2 or g.shape[1] != 4:
        raise ValueError("geometry must have shape (N,4).")
    return (
        np.isfinite(g).all(axis=1)
        & (g >= ROBUST_LOWER[None, :] - atol).all(axis=1)
        & (g <= ROBUST_UPPER[None, :] + atol).all(axis=1)
        & (g.sum(axis=1) <= ROBUST_MAX_OUTER + atol)
    )


def original_geometry_mask(geometry: np.ndarray, atol: float = 1.0e-8) -> np.ndarray:
    g = np.asarray(geometry, dtype=np.float64)
    if g.ndim == 1:
        g = g[None, :]
    return (
        np.isfinite(g).all(axis=1)
        & (g >= ORIGINAL_LOWER[None, :] - atol).all(axis=1)
        & (g <= ORIGINAL_UPPER[None, :] + atol).all(axis=1)
        & (g.sum(axis=1) <= ORIGINAL_MAX_OUTER + atol)
    )


def project_robust_geometry_numpy(geometry: np.ndarray) -> np.ndarray:
    g = np.asarray(geometry, dtype=np.float64).copy()
    one = g.ndim == 1
    if one:
        g = g[None, :]
    g = np.clip(g, ROBUST_LOWER[None, :], ROBUST_UPPER[None, :])
    extra = g - ROBUST_LOWER[None, :]
    # Keep continuous initialisation away from the exact integer boundary.
    # On the 400--1000 nm grid, some exact integer outer radii are removable
    # singular points of the Peña recursion. Integer Stage 08D handles those
    # points with a symmetric limiting evaluation; continuous optimisation is
    # kept half a nanometre inside the robust legal domain.
    budget = CONTINUOUS_MAX_OUTER - float(ROBUST_LOWER.sum())
    totals = extra.sum(axis=1)
    scale = np.minimum(1.0, budget / np.maximum(totals, 1.0e-15))
    g = ROBUST_LOWER[None, :] + extra * scale[:, None]
    if not np.all(robust_geometry_mask(g)):
        raise RuntimeError("Robust geometry projection failed.")
    return g[0] if one else g


def robust_geometry_from_unit(unit: np.ndarray) -> np.ndarray:
    u = np.asarray(unit, dtype=np.float64)
    if u.ndim == 1:
        u = u[None, :]
    if u.ndim != 2 or u.shape[1] != 4:
        raise ValueError("Sobol unit coordinates must have shape (N,4).")
    u = np.clip(u, 0.0, 1.0)
    proposed = ROBUST_LOWER[None, :] + u * (ROBUST_UPPER - ROBUST_LOWER)[None, :]
    return project_robust_geometry_numpy(proposed)


def robust_geometry_from_raw(raw: torch.Tensor) -> torch.Tensor:
    lower = torch.as_tensor(ROBUST_LOWER, dtype=raw.dtype, device=raw.device)
    upper = torch.as_tensor(ROBUST_UPPER, dtype=raw.dtype, device=raw.device)
    fraction = torch.sigmoid(raw)
    extra = fraction * (upper - lower)
    budget = torch.as_tensor(
        CONTINUOUS_MAX_OUTER - float(ROBUST_LOWER.sum()), dtype=raw.dtype, device=raw.device
    )
    totals = extra.sum(dim=1, keepdim=True)
    scale = torch.clamp(budget / torch.clamp(totals, min=1.0e-15), max=1.0)
    return lower[None, :] + extra * scale


def robust_raw_from_geometry(geometry: torch.Tensor) -> torch.Tensor:
    lower = torch.as_tensor(ROBUST_LOWER, dtype=geometry.dtype, device=geometry.device)
    upper = torch.as_tensor(ROBUST_UPPER, dtype=geometry.dtype, device=geometry.device)
    fraction = (geometry - lower[None, :]) / (upper - lower)[None, :]
    fraction = torch.clamp(fraction, 1.0e-6, 1.0 - 1.0e-6)
    return torch.log(fraction) - torch.log1p(-fraction)


def sobol_geometries(count: int, *, seed: int, skip: int = 0) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive.")
    engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=int(seed))
    if skip > 0:
        engine.fast_forward(int(skip))
    unit = engine.draw(int(count)).double().cpu().numpy()
    return robust_geometry_from_unit(unit)


def axial_perturbations(include_nominal: bool = True) -> np.ndarray:
    rows: list[tuple[int, int, int, int]] = []
    if include_nominal:
        rows.append((0, 0, 0, 0))
    for axis in range(4):
        for sign in (-1, 1):
            row = [0, 0, 0, 0]
            row[axis] = sign
            rows.append(tuple(row))
    return np.asarray(rows, dtype=np.float64)


def full_perturbations() -> np.ndarray:
    return np.asarray(list(product((-1, 0, 1), repeat=4)), dtype=np.float64)


def expand_perturbations(
    geometry: np.ndarray,
    materials: np.ndarray,
    perturbations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(geometry, dtype=np.float64)
    m = np.asarray(materials, dtype=np.int64)
    p = np.asarray(perturbations, dtype=np.float64)
    expanded_g = (g[:, None, :] + p[None, :, :]).reshape(-1, 4)
    expanded_m = np.repeat(m, len(p), axis=0)
    parent = np.repeat(np.arange(len(g), dtype=np.int64), len(p))
    state = np.tile(np.arange(len(p), dtype=np.int64), len(g))
    if not np.all(original_geometry_mask(expanded_g)):
        bad = np.flatnonzero(~original_geometry_mask(expanded_g))[:10]
        raise RuntimeError(f"A manufacturing perturbation is geometrically illegal: rows={bad.tolist()}")
    return expanded_g, expanded_m, parent, state


def numpy_min_abs_sin(geometry: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    g = np.asarray(geometry, dtype=np.float64)
    outer = g.sum(axis=1)
    phase = 2.0 * np.pi * outer[:, None] / np.asarray(wavelengths_nm, dtype=np.float64)[None, :]
    return np.abs(np.sin(phase)).min(axis=1)


def torch_min_abs_sin(geometry: torch.Tensor, wavelengths_nm: torch.Tensor) -> torch.Tensor:
    outer = geometry.sum(dim=1)
    phase = 2.0 * math.pi * outer[:, None] / wavelengths_nm[None, :]
    return torch.abs(torch.sin(phase)).amin(dim=1)


def physical_validity(
    qsca: torch.Tensor,
    qabs: torch.Tensor,
    qext: torch.Tensor,
    *,
    negative_tolerance: float,
    energy_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(qsca).all(dim=1) & torch.isfinite(qabs).all(dim=1) & torch.isfinite(qext).all(dim=1)
    minimum = torch.minimum(torch.minimum(qsca.amin(dim=1), qabs.amin(dim=1)), qext.amin(dim=1))
    nonnegative = minimum >= -float(negative_tolerance)
    residual = (qext - qsca - qabs).abs().amax(dim=1) / torch.maximum(
        torch.ones(qext.shape[0], dtype=qext.dtype, device=qext.device),
        qext.abs().amax(dim=1),
    )
    return finite & nonnegative & (residual <= float(energy_tolerance)), residual, minimum


def recursive_forward(
    geometry: torch.Tensor,
    materials: torch.Tensor,
    physics: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    count = int(geometry.shape[0])
    try:
        result = pymiediff_forward(geometry, materials, physics, backend="pena", precision="double")
        return result["q_sca"], result["q_abs"], result["q_ext"], [""] * count
    except Exception as exc:
        if count == 1:
            width = len(physics["wavelengths"])
            nan = torch.full((1, width), torch.nan, dtype=torch.float64, device=geometry.device)
            return nan.clone(), nan.clone(), nan.clone(), [f"{type(exc).__name__}: {exc}"]
        mid = count // 2
        left = recursive_forward(geometry[:mid], materials[:mid], physics)
        right = recursive_forward(geometry[mid:], materials[mid:], physics)
        return (
            torch.cat((left[0], right[0]), dim=0),
            torch.cat((left[1], right[1]), dim=0),
            torch.cat((left[2], right[2]), dim=0),
            left[3] + right[3],
        )




def symmetric_outer_radius_limit_forward(
    geometry: torch.Tensor,
    materials: torch.Tensor,
    physics: Mapping[str, Any],
    *,
    epsilon_nm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Evaluate removable Peña singularities by a symmetric radius limit.

    The exact reported geometry is unchanged.  Four infinitesimally shifted
    outer-shell thicknesses (±h and ±h/2) are evaluated, symmetric averages
    are formed, and a Richardson estimate is returned.  ``limit_error`` is
    the relative discrepancy between the h and h/2 symmetric averages.
    """
    h = float(epsilon_nm)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("epsilon_nm must be finite and positive.")
    geometries = []
    for delta in (h, -h, 0.5 * h, -0.5 * h):
        shifted = geometry.clone()
        shifted[:, 3] = shifted[:, 3] + delta
        geometries.append(shifted)
    stacked_geometry = torch.cat(geometries, dim=0)
    stacked_materials = materials.repeat(4, 1)
    qsca, qabs, qext, errors = recursive_forward(
        stacked_geometry, stacked_materials, physics
    )
    count = geometry.shape[0]
    error_groups = [errors[i * count : (i + 1) * count] for i in range(4)]
    combined_errors: list[str] = []
    for row in range(count):
        values = [group[row] for group in error_groups if group[row]]
        combined_errors.append("; ".join(values))

    def extrapolate(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        plus_h, minus_h, plus_h2, minus_h2 = torch.split(values, count, dim=0)
        average_h = 0.5 * (plus_h + minus_h)
        average_h2 = 0.5 * (plus_h2 + minus_h2)
        estimate = (4.0 * average_h2 - average_h) / 3.0
        scale = torch.maximum(
            torch.ones(count, dtype=values.dtype, device=values.device),
            average_h2.abs().amax(dim=1),
        )
        error = (average_h2 - average_h).abs().amax(dim=1) / scale
        return estimate, error

    qsca_est, qsca_error = extrapolate(qsca)
    qabs_est, qabs_error = extrapolate(qabs)
    qext_est, qext_error = extrapolate(qext)
    limit_error = torch.maximum(torch.maximum(qsca_error, qabs_error), qext_error)
    return qsca_est, qabs_est, qext_est, limit_error, combined_errors


def spectral_metrics_torch(
    qsca: torch.Tensor,
    qabs: torch.Tensor,
    qext: torch.Tensor,
    outer_radius_nm: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    qsca = torch.clamp(qsca, min=0.0)
    qabs = torch.clamp(qabs, min=0.0)
    qext = torch.clamp(qext, min=0.0)
    width = int(qsca.shape[1])
    tail_count = max(1, int(math.ceil(0.10 * width)))
    tail = torch.topk(qsca, k=tail_count, dim=1, largest=False, sorted=False).values.mean(dim=1)
    qsca_p10 = torch.quantile(qsca, 0.10, dim=1)
    mean_qsca = qsca.mean(dim=1)
    std_qsca = qsca.std(dim=1, unbiased=False)
    qsca_cv = std_qsca / torch.clamp(mean_qsca, min=1.0e-12)
    fraction = qabs / torch.clamp(qext, min=1.0e-12)
    band_fraction = qabs.sum(dim=1) / torch.clamp(qext.sum(dim=1), min=1.0e-12)
    area = math.pi * torch.square(outer_radius_nm)
    return {
        "tail_mean_qsca": tail,
        "qsca_p10": qsca_p10,
        "mean_qsca": mean_qsca,
        "min_qsca": qsca.amin(dim=1),
        "max_qsca": qsca.amax(dim=1),
        "qsca_cv": qsca_cv,
        "mean_qabs": qabs.mean(dim=1),
        "qabs_p95": torch.quantile(qabs, 0.95, dim=1),
        "max_qabs": qabs.amax(dim=1),
        "band_absorption_fraction": band_fraction,
        "absorption_fraction_p95": torch.quantile(fraction, 0.95, dim=1),
        "single_scattering_albedo": 1.0 - band_fraction,
        "mean_csca_nm2": mean_qsca * area,
    }


def metrics_frame_from_spectra(
    qsca: torch.Tensor,
    qabs: torch.Tensor,
    qext: torch.Tensor,
    geometry: torch.Tensor,
    *,
    residual: torch.Tensor,
    minimum_efficiency: torch.Tensor,
    minimum_abs_sin: torch.Tensor,
    solver_valid: torch.Tensor,
    physics_valid: torch.Tensor,
) -> pd.DataFrame:
    metrics = spectral_metrics_torch(qsca, qabs, qext, geometry.sum(dim=1))
    metrics.update(
        {
            "energy_residual": residual,
            "minimum_efficiency": minimum_efficiency,
            "minimum_abs_sin": minimum_abs_sin,
            "solver_valid": solver_valid,
            "physics_valid": physics_valid,
        }
    )
    return pd.DataFrame({key: value.detach().cpu().numpy() for key, value in metrics.items()})


def tier_feasible(frame: pd.DataFrame, tier: Mapping[str, Any], prefix: str = "") -> np.ndarray:
    def c(name: str) -> str:
        return f"{prefix}{name}"
    return (
        frame[c("band_absorption_fraction")].to_numpy(dtype=np.float64)
        <= float(tier["band_absorption_fraction_max"])
    ) & (
        frame[c("absorption_fraction_p95")].to_numpy(dtype=np.float64)
        <= float(tier["absorption_fraction_p95_max"])
    ) & (
        frame[c("max_qabs")].to_numpy(dtype=np.float64)
        <= float(tier["qabs_max_max"])
    )


def normalized_absorption_violation_numpy(frame: pd.DataFrame, tier: Mapping[str, Any], prefix: str = "") -> np.ndarray:
    values = []
    for column, limit_key in (
        ("band_absorption_fraction", "band_absorption_fraction_max"),
        ("absorption_fraction_p95", "absorption_fraction_p95_max"),
        ("max_qabs", "qabs_max_max"),
    ):
        ratio = frame[f"{prefix}{column}"].to_numpy(dtype=np.float64) / float(tier[limit_key]) - 1.0
        values.append(np.maximum(ratio, 0.0))
    stacked = np.stack(values, axis=1)
    return np.square(stacked).sum(axis=1)


def absorption_violation_torch(metrics: Mapping[str, torch.Tensor], tier: Mapping[str, Any]) -> torch.Tensor:
    terms = []
    for name, limit_key in (
        ("band_absorption_fraction", "band_absorption_fraction_max"),
        ("absorption_fraction_p95", "absorption_fraction_p95_max"),
        ("max_qabs", "qabs_max_max"),
    ):
        terms.append(torch.relu(metrics[name] / float(tier[limit_key]) - 1.0))
    return torch.stack(terms, dim=1)


def evaluate_design_pool(
    *,
    geometry: np.ndarray,
    materials: np.ndarray,
    physics: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    singularity_tolerance: float,
    negative_tolerance: float,
    energy_tolerance: float,
    singularity_policy: str = "reject",
    singularity_limit_epsilon_nm: float = 1.0e-3,
    singularity_limit_rtol: float = 1.0e-5,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every_batches: int = 10,
    resume: bool = False,
    store_spectra: bool = False,
    print_label: str = "PyMieDiff evaluation",
) -> tuple[pd.DataFrame, Optional[dict[str, np.ndarray]]]:
    g = np.asarray(geometry, dtype=np.float64)
    m = np.asarray(materials, dtype=np.int64)
    if g.shape != (len(m), 4) or m.shape[1:] != (4,):
        raise ValueError("Geometry/material arrays must have shapes (N,4).")
    n = len(g)
    width = len(physics["wavelengths"])
    input_hash = stable_hash(
        {
            "arrays": array_hash(g.astype("<f8"), m.astype("<i8")),
            "singularity_tolerance": float(singularity_tolerance),
            "negative_tolerance": float(negative_tolerance),
            "energy_tolerance": float(energy_tolerance),
            "singularity_policy": str(singularity_policy),
            "singularity_limit_epsilon_nm": float(singularity_limit_epsilon_nm),
            "singularity_limit_rtol": float(singularity_limit_rtol),
            "store_spectra": bool(store_spectra),
        }
    )
    boolean_metrics = {"solver_regularized", "solver_valid", "physics_valid"}
    metric_arrays: dict[str, np.ndarray] = {
        key: np.full(n, np.nan, dtype=np.float64)
        for key in METRIC_COLUMNS
        if key not in boolean_metrics
    }
    for key in boolean_metrics:
        metric_arrays[key] = np.zeros(n, dtype=bool)
    completed = np.zeros(n, dtype=bool)
    solver_error = np.full(n, "", dtype="U256")
    spectra: Optional[dict[str, np.ndarray]] = None
    if store_spectra:
        spectra = {
            "qsca": np.full((n, width), np.nan, dtype=np.float64),
            "qabs": np.full((n, width), np.nan, dtype=np.float64),
            "qext": np.full((n, width), np.nan, dtype=np.float64),
        }
    if resume and checkpoint_path is not None and Path(checkpoint_path).is_file():
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            if str(np.asarray(archive["input_hash"]).item()) != input_hash:
                raise RuntimeError(f"Checkpoint input hash mismatch: {checkpoint_path}")
            completed = np.asarray(archive["completed"], dtype=bool)
            solver_error = np.asarray(archive["solver_error"]).astype("U256")
            for key in metric_arrays:
                metric_arrays[key] = np.asarray(archive[f"metric_{key}"]).copy()
            if store_spectra and spectra is not None:
                for key in spectra:
                    spectra[key] = np.asarray(archive[key], dtype=np.float64)
        print(f"{print_label}: resumed {int(completed.sum())}/{n} completed rows")

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        payload: dict[str, Any] = {
            "input_hash": np.asarray(input_hash),
            "completed": completed,
            "solver_error": solver_error,
        }
        for key, value in metric_arrays.items():
            payload[f"metric_{key}"] = value
        if store_spectra and spectra is not None:
            payload.update(spectra)
        save_npz_atomic(Path(checkpoint_path), **payload)

    if singularity_policy not in {"reject", "symmetric_limit"}:
        raise ValueError("singularity_policy must be 'reject' or 'symmetric_limit'.")
    pending = np.flatnonzero(~completed)
    batch_counter = 0
    for offset in range(0, len(pending), int(batch_size)):
        rows = pending[offset : offset + int(batch_size)]
        geometry_np = g[rows]
        material_np = m[rows]
        minimum_sine_np = numpy_min_abs_sin(geometry_np, physics["wavelengths"])
        near_singular = minimum_sine_np <= float(singularity_tolerance)
        metric_arrays["minimum_abs_sin"][rows] = minimum_sine_np
        if np.any(near_singular):
            singular_rows = rows[near_singular]
            if singularity_policy == "reject":
                solver_error[singular_rows] = "solver_singularity_guard"
                completed[singular_rows] = True
            else:
                gt = torch.as_tensor(g[singular_rows], dtype=torch.float64, device=device)
                mt = torch.as_tensor(m[singular_rows], dtype=torch.long, device=device)
                qsca, qabs, qext, limit_error, errors = symmetric_outer_radius_limit_forward(
                    gt, mt, physics, epsilon_nm=float(singularity_limit_epsilon_nm)
                )
                solver_ok = torch.as_tensor(
                    [error == "" for error in errors], dtype=torch.bool, device=device
                ) & torch.isfinite(limit_error) & (limit_error <= float(singularity_limit_rtol))
                valid, residual, minimum = physical_validity(
                    qsca, qabs, qext,
                    negative_tolerance=float(negative_tolerance),
                    energy_tolerance=float(energy_tolerance),
                )
                valid = valid & solver_ok
                frame = metrics_frame_from_spectra(
                    torch.clamp(qsca, min=0.0),
                    torch.clamp(qabs, min=0.0),
                    torch.clamp(qext, min=0.0),
                    gt,
                    residual=residual,
                    minimum_efficiency=minimum,
                    minimum_abs_sin=torch.as_tensor(
                        minimum_sine_np[near_singular], dtype=torch.float64, device=device
                    ),
                    solver_valid=solver_ok,
                    physics_valid=valid,
                )
                frame["solver_regularized"] = True
                frame["singularity_limit_error"] = limit_error.detach().cpu().numpy()
                for key in metric_arrays:
                    metric_arrays[key][singular_rows] = frame[key].to_numpy()
                solver_error[singular_rows] = np.asarray(
                    [error if error else ("" if bool(ok) else "singularity_limit_not_converged")
                     for error, ok in zip(errors, solver_ok.detach().cpu().numpy())],
                    dtype="U256",
                )
                if store_spectra and spectra is not None:
                    spectra["qsca"][singular_rows] = torch.clamp(qsca, min=0.0).detach().cpu().numpy()
                    spectra["qabs"][singular_rows] = torch.clamp(qabs, min=0.0).detach().cpu().numpy()
                    spectra["qext"][singular_rows] = torch.clamp(qext, min=0.0).detach().cpu().numpy()
                completed[singular_rows] = True
        good_local = np.flatnonzero(~near_singular)
        if len(good_local):
            good_rows = rows[good_local]
            gt = torch.as_tensor(g[good_rows], dtype=torch.float64, device=device)
            mt = torch.as_tensor(m[good_rows], dtype=torch.long, device=device)
            qsca, qabs, qext, errors = recursive_forward(gt, mt, physics)
            solver_ok = torch.as_tensor([error == "" for error in errors], dtype=torch.bool, device=device)
            valid, residual, minimum = physical_validity(
                qsca,
                qabs,
                qext,
                negative_tolerance=float(negative_tolerance),
                energy_tolerance=float(energy_tolerance),
            )
            valid = valid & solver_ok
            clean_qsca = torch.clamp(qsca, min=0.0)
            clean_qabs = torch.clamp(qabs, min=0.0)
            clean_qext = torch.clamp(qext, min=0.0)
            frame = metrics_frame_from_spectra(
                clean_qsca,
                clean_qabs,
                clean_qext,
                gt,
                residual=residual,
                minimum_efficiency=minimum,
                minimum_abs_sin=torch.as_tensor(minimum_sine_np[good_local], dtype=torch.float64, device=device),
                solver_valid=solver_ok,
                physics_valid=valid,
            )
            frame["solver_regularized"] = False
            frame["singularity_limit_error"] = 0.0
            for key in metric_arrays:
                metric_arrays[key][good_rows] = frame[key].to_numpy()
            solver_error[good_rows] = np.asarray(errors, dtype="U256")
            if store_spectra and spectra is not None:
                spectra["qsca"][good_rows] = clean_qsca.detach().cpu().numpy()
                spectra["qabs"][good_rows] = clean_qabs.detach().cpu().numpy()
                spectra["qext"][good_rows] = clean_qext.detach().cpu().numpy()
            completed[good_rows] = True
        batch_counter += 1
        print(f"{print_label}: {int(completed.sum())}/{n}")
        if batch_counter % max(1, int(checkpoint_every_batches)) == 0:
            save_checkpoint()
    save_checkpoint()
    frame = pd.DataFrame(metric_arrays)
    frame.insert(0, "solver_error", solver_error)
    frame.insert(0, "row_index", np.arange(n, dtype=np.int64))
    return frame, spectra


def aggregate_perturbation_metrics(
    state_metrics: pd.DataFrame,
    parent_index: np.ndarray,
    *,
    prefix: str = "robust_",
) -> pd.DataFrame:
    frame = state_metrics.copy()
    frame["parent_index"] = np.asarray(parent_index, dtype=np.int64)
    rows: list[Dict[str, Any]] = []
    for parent, group in frame.groupby("parent_index", sort=True):
        valid = group["physics_valid"].astype(bool) & group["solver_valid"].astype(bool)
        record: Dict[str, Any] = {
            "parent_index": int(parent),
            f"{prefix}all_states_valid": bool(valid.all()),
            f"{prefix}valid_state_count": int(valid.sum()),
            f"{prefix}state_count": int(len(group)),
        }
        if valid.any():
            selected = group.loc[valid]
            record.update(
                {
                    f"{prefix}tail_mean_qsca": float(selected["tail_mean_qsca"].min()),
                    f"{prefix}qsca_p10": float(selected["qsca_p10"].min()),
                    f"{prefix}mean_qsca": float(selected["mean_qsca"].min()),
                    f"{prefix}min_qsca": float(selected["min_qsca"].min()),
                    f"{prefix}qsca_cv": float(selected["qsca_cv"].max()),
                    f"{prefix}band_absorption_fraction": float(selected["band_absorption_fraction"].max()),
                    f"{prefix}absorption_fraction_p95": float(selected["absorption_fraction_p95"].max()),
                    f"{prefix}max_qabs": float(selected["max_qabs"].max()),
                    f"{prefix}mean_qabs": float(selected["mean_qabs"].max()),
                    f"{prefix}single_scattering_albedo": float(selected["single_scattering_albedo"].min()),
                    f"{prefix}energy_residual": float(selected["energy_residual"].max()),
                    f"{prefix}minimum_abs_sin": float(selected["minimum_abs_sin"].min()),
                }
            )
        else:
            for name in (
                "tail_mean_qsca", "qsca_p10", "mean_qsca", "min_qsca", "qsca_cv",
                "band_absorption_fraction", "absorption_fraction_p95", "max_qabs",
                "mean_qabs", "single_scattering_albedo", "energy_residual", "minimum_abs_sin",
            ):
                record[f"{prefix}{name}"] = np.nan
        rows.append(record)
    return pd.DataFrame(rows).sort_values("parent_index").reset_index(drop=True)


def write_protocol_lock(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    base = dict(payload)
    base.pop("protocol_sha256", None)
    base["protocol_sha256"] = stable_hash(base)
    save_json_atomic(path, base)
    return base


def verify_protocol_lock(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    recorded = str(payload.get("protocol_sha256", ""))
    check = dict(payload)
    check.pop("protocol_sha256", None)
    if recorded != stable_hash(check):
        raise RuntimeError("Stage-08 protocol lock self-hash mismatch.")
    return payload


def sequence_strings(sequences: np.ndarray, names: Sequence[str]) -> np.ndarray:
    material_names = np.asarray(names).astype(str)
    return np.asarray(["|".join(material_names[row].tolist()) for row in sequences], dtype="U128")


def boundary_clearance_nm(geometry: np.ndarray) -> np.ndarray:
    g = np.asarray(geometry, dtype=np.float64)
    margins = np.column_stack(
        (
            g - ORIGINAL_LOWER[None, :],
            ORIGINAL_UPPER[None, :] - g,
            ORIGINAL_MAX_OUTER - g.sum(axis=1),
        )
    )
    return margins.min(axis=1)


def validate_common_self_test() -> None:
    unit = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    g = robust_geometry_from_unit(unit)
    assert np.all(robust_geometry_mask(g))
    perturb = full_perturbations()
    expanded, _, _, _ = expand_perturbations(g, np.zeros((len(g), 4), dtype=np.int64), perturb)
    assert np.all(original_geometry_mask(expanded))
    tensor = torch.as_tensor(g, dtype=torch.float64)
    reconstructed = robust_geometry_from_raw(robust_raw_from_geometry(tensor)).detach().cpu().numpy()
    assert np.allclose(g, reconstructed, atol=1.0e-5)
    qsca = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    qabs = torch.zeros_like(qsca)
    qext = qsca.clone()
    metrics = spectral_metrics_torch(qsca, qabs, qext, torch.tensor([250.0], dtype=torch.float64))
    assert float(metrics["band_absorption_fraction"][0]) == 0.0
    assert float(metrics["mean_qsca"][0]) == 2.5


__all__ = [
    "ABSORPTION_TIERS", "DEFAULT_DATASET", "DEFAULT_PGFF_DIR", "GEOMETRY_COLUMNS",
    "MATERIAL_COLUMNS", "PREPARED_DATA", "PROJECT_ROOT", "RESULT_ROOT", "ROBUST_LOWER",
    "CONTINUOUS_MAX_OUTER",
    "ROBUST_MAX_OUTER", "ROBUST_UPPER", "STAGE02_DIR", "aggregate_perturbation_metrics",
    "array_hash", "atomic_write_csv", "axial_perturbations", "boundary_clearance_nm",
    "evaluate_design_pool", "expand_perturbations", "full_perturbations", "load_json",
    "symmetric_outer_radius_limit_forward",
    "load_physics_assets", "load_prepared_registry", "numpy_min_abs_sin", "original_geometry_mask",
    "prepare_output_dir", "project_robust_geometry_numpy", "recursive_forward", "resolve_device",
    "robust_geometry_from_raw", "robust_geometry_from_unit", "robust_geometry_mask",
    "robust_raw_from_geometry", "save_json_atomic", "save_npz_atomic", "sequence_strings",
    "set_seed", "sha256_file", "sobol_geometries", "spectral_metrics_torch", "stable_hash",
    "tier_feasible", "normalized_absorption_violation_numpy", "absorption_violation_torch",
    "torch_min_abs_sin", "validate_common_self_test", "verify_frozen_assets",
    "verify_protocol_lock", "write_protocol_lock", "artifact",
]
