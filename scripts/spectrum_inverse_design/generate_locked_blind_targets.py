# -*- coding: utf-8 -*-
"""Stage 02B: generate and freeze ten strict public blind Qsca targets.

This benchmark-construction stage is deliberately independent of the inverse
network and PGFF. It uses only the frozen forward database and PyMieDiff.
The completed Stage 02 model is hashed before target generation to document
that inverse-network training was frozen before the blind benchmark existed.

Public output
-------------
``results/reproduced/spectrum_inverse_design/blind_targets_public/blind_target_spectra.npz``
contains only Qsca-derived public information. Geometry, materials, Qabs and
Qext are written exclusively to a separate directory whose name contains
``private``; Stage 03 refuses to read any such path.

Formal target protocol
----------------------
* 4096 unique Sobol, off-grid, boundary-interior physical candidates.
* Exact PyMieDiff, double precision, energy-conservation validation.
* Balanced coverage of all 400 legal material sequences.
* Deterministic 25% calibration / 75% selection split.
* Exact nearest-neighbour comparison against all 30,000 database spectra.
* Same-material geometry-nearest-neighbour percentile >= 50%.
* Spectral difficulty bands: 3 easy (P50-P70), 4 medium (P70-P90),
  3 hard (P90-P97.5), calibrated on the disjoint calibration subset.
* Target peak amplitude within database P10-P90 and spectral dynamic range
  at least database P25.
* At least 8 distinct material sequences; no sequence used more than twice.
* Within each difficulty band, deterministic farthest-point spectral selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from core_shell_optical_design.inverse_design.inverse_design_data import (
    DEFAULT_DATASET,
    DEFAULT_RESULT_ROOT,
    artifact,
    load_forward_dataset,
    resolve_device,
    save_json_atomic,
    save_npz_atomic,
    set_seed,
    sha256_file,
)
from core_shell_optical_design.inverse_design.forward_model_bridge import pymiediff_forward
from core_shell_optical_design.inverse_design.geometry_parameterization import (
    LOWER_BOUNDS,
    MAX_OUTER_RADIUS,
    UPPER_BOUNDS,
    enumerate_legal_material_sequences,
    geometry_bounds_mask,
    material_sequence_mask,
)

STAGE = "generate_locked_blind_targets"
VERSION = "tandem-mie-inverse-2026-07-24-target-benchmark-v2-singularity-guard"
DEFAULT_STAGE02_DIR = DEFAULT_RESULT_ROOT / "inverse_model_training"
DEFAULT_PUBLIC_DIR = DEFAULT_RESULT_ROOT / "blind_targets_public"
DEFAULT_PRIVATE_DIR = DEFAULT_RESULT_ROOT / "blind_references_private"
DEFAULT_DIAGNOSTICS_DIR = DEFAULT_RESULT_ROOT / "target_selection_diagnostics"
DEFAULT_FIGURE_DIR = DEFAULT_RESULT_ROOT / "target_selection_visual_diagnostics"
DEFAULT_ARCHIVE_ROOT = DEFAULT_RESULT_ROOT / "blind_benchmark_archive"
DEFAULT_WORK_DIR = DEFAULT_RESULT_ROOT / "blind_target_generation_work"


@dataclass(frozen=True)
class TargetProtocol:
    target_count: int = 10
    pool_size: int = 4096
    seed: int = 20260725
    calibration_fraction: float = 0.25
    offgrid_quantization_nm: float = 0.25
    boundary_margin_nm: float = 2.0
    geometry_percentile_min: float = 0.50
    easy_percentile_low: float = 0.50
    easy_percentile_high: float = 0.70
    medium_percentile_low: float = 0.70
    medium_percentile_high: float = 0.90
    hard_percentile_low: float = 0.90
    hard_percentile_high: float = 0.975
    easy_count: int = 3
    medium_count: int = 4
    hard_count: int = 3
    minimum_unique_material_sequences: int = 8
    maximum_targets_per_material_sequence: int = 2
    peak_amplitude_database_quantiles: Tuple[float, float] = (0.10, 0.90)
    dynamic_range_database_quantile_min: float = 0.25
    pymiediff_backend: str = "pena"
    pymiediff_precision: str = "double"
    solver_singularity_tolerance: float = 1.0e-6
    negative_efficiency_tolerance: float = 1.0e-7
    energy_residual_tolerance: float = 1.0e-8


def parse_args() -> argparse.Namespace:
    protocol = TargetProtocol()
    parser = argparse.ArgumentParser(
        description="Generate a frozen, database-external ten-target Qsca blind benchmark"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--stage02-dir", type=Path, default=DEFAULT_STAGE02_DIR)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--private-dir", type=Path, default=DEFAULT_PRIVATE_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DEFAULT_DIAGNOSTICS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--pool-size", type=int, default=protocol.pool_size)
    parser.add_argument("--seed", type=int, default=protocol.seed)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--distance-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--pymiediff-batch-size", type=int, default=32)
    parser.add_argument("--spectral-query-batch-size", type=int, default=64)
    parser.add_argument(
        "--solver-singularity-tolerance",
        type=float,
        default=protocol.solver_singularity_tolerance,
        help=(
            "Exclude candidates with min |sin(2*pi*Rout/lambda)| at or below this value. "
            "The Pena recurrence is singular at exact x=m*pi points."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=8,
        help="Save a resumable private PyMieDiff pool checkpoint every N batches.",
    )
    parser.add_argument(
        "--restart-pool-evaluation",
        action="store_true",
        help="Discard an existing private pool checkpoint and recompute from batch 1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--save-pool-spectra",
        action="store_true",
        help="Also save all pool Qsca/Qabs/Qext; useful for audit but larger.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.pool_size < 512:
        raise ValueError("Formal target construction requires --pool-size >= 512.")
    if args.pymiediff_batch_size <= 0 or args.spectral_query_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.checkpoint_every_batches <= 0:
        raise ValueError("--checkpoint-every-batches must be positive.")
    if not np.isfinite(args.solver_singularity_tolerance) or args.solver_singularity_tolerance <= 0.0:
        raise ValueError("--solver-singularity-tolerance must be finite and positive.")
    if args.public_dir.resolve() == args.private_dir.resolve():
        raise ValueError("Public and private output directories must be different.")
    if "private" in {part.lower() for part in args.public_dir.parts}:
        raise ValueError("The public target path must not contain a private directory.")
    if "private" not in {part.lower() for part in args.private_dir.parts} and not any(
        "private" in part.lower() for part in args.private_dir.parts
    ):
        raise ValueError("The hidden-reference output path must visibly contain 'private'.")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spectrum_hash(wavelengths: np.ndarray, qsca: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(wavelengths, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(qsca, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def verify_stage02_frozen(stage02_dir: Path) -> Dict[str, Any]:
    summary_path = Path(stage02_dir) / "training_summary.json"
    best_model_path = Path(stage02_dir) / "best_model.pt"
    if not summary_path.is_file() or not best_model_path.is_file():
        raise FileNotFoundError(
            "Stage 02 must be completed and frozen before blind target generation.\n"
            f"Missing: {summary_path if not summary_path.is_file() else best_model_path}"
        )
    summary = load_json(summary_path)
    if summary.get("status") != "passed" or not bool(summary.get("formal_run", False)):
        raise RuntimeError("Stage 02 summary is not a passed formal run.")
    policy = summary.get("data_policy", {})
    if bool(policy.get("blind_targets_read", True)):
        raise RuntimeError("Stage 02 reports that blind targets were read; strict blindness failed.")
    if bool(policy.get("test_split_evaluated", True)):
        raise RuntimeError("Stage 02 reports test-split evaluation; expected untouched test split.")
    return {
        "summary": artifact(summary_path),
        "best_model": artifact(best_model_path),
        "best_epoch": int(summary["best_epoch"]),
        "best_validation_mean_nmae": float(summary["best_validation_mean_nmae"]),
    }


def archive_or_prepare(paths: Sequence[Path], archive_root: Path, overwrite: bool) -> Path | None:
    occupied = [Path(p) for p in paths if Path(p).exists() and any(Path(p).iterdir())]
    if occupied and not overwrite:
        raise FileExistsError(
            "A frozen target benchmark already exists. Use --overwrite only when intentionally "
            "creating a new benchmark. Existing outputs:\n" + "\n".join(str(p) for p in occupied)
        )
    archive_dir: Path | None = None
    if occupied:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path(archive_root) / stamp
        archive_dir.mkdir(parents=True, exist_ok=False)
        for path in occupied:
            destination = archive_dir / path.name
            shutil.move(str(path), str(destination))
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
    return archive_dir


def solver_singularity_diagnostics(
    geometry: np.ndarray,
    wavelengths_nm: np.ndarray,
    tolerance: float,
) -> Dict[str, np.ndarray]:
    """Diagnose Peña-recursion singularities at external x = m*pi.

    PyMieDiff's Peña logarithmic-derivative recurrence is numerically singular
    when the external size parameter x=2*pi*Rout/lambda is exactly an integer
    multiple of pi.  Such points can produce very large, non-physical negative
    absorption even in double precision and therefore must be excluded before
    high-fidelity evaluation rather than clipped afterwards.
    """
    values = np.asarray(geometry, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("geometry must have shape [N,4].")
    if len(wavelengths) == 0 or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_nm must be positive and non-empty.")
    outer_radius = values.sum(axis=1)
    phase = 2.0 * np.pi * outer_radius[:, None] / wavelengths[None, :]
    absolute_sine = np.abs(np.sin(phase))
    critical_index = np.argmin(absolute_sine, axis=1)
    row = np.arange(len(values), dtype=np.int64)
    minimum = absolute_sine[row, critical_index]
    return {
        "is_safe": minimum > float(tolerance),
        "outer_radius_nm": outer_radius,
        "min_abs_sin": minimum,
        "critical_wavelength_index": critical_index,
        "critical_wavelength_nm": wavelengths[critical_index],
        "critical_phase_over_pi": phase[row, critical_index] / np.pi,
    }


def sample_sobol_geometries(
    n_samples: int,
    seed: int,
    quantization_nm: float,
    margin_nm: float,
    wavelengths_nm: np.ndarray,
    singularity_tolerance: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if quantization_nm <= 0.0:
        raise ValueError("quantization_nm must be positive.")
    low = LOWER_BOUNDS + float(margin_nm)
    high = UPPER_BOUNDS - float(margin_nm)
    outer_max = MAX_OUTER_RADIUS - float(margin_nm)
    if np.any(high <= low) or outer_max <= float(low.sum()):
        raise ValueError("Boundary margin leaves no feasible geometry volume.")

    engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=int(seed))
    rows: list[np.ndarray] = []
    seen: set[Tuple[int, int, int, int]] = set()
    draw_count = max(8192, int(n_samples) * 4)
    attempts = 0
    rejected_singular = 0
    rejected_integer_grid = 0
    rejected_geometry = 0
    duplicate_count = 0
    examined = 0
    while len(rows) < int(n_samples):
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Could not obtain enough unique feasible solver-safe Sobol geometries.")
        u = engine.draw(draw_count).cpu().numpy().astype(np.float64)
        geometry = low[None, :] + u * (high - low)[None, :]
        geometry = np.round(geometry / quantization_nm) * quantization_nm
        examined += len(geometry)
        legal = (
            np.all(geometry >= low[None, :] - 1.0e-10, axis=1)
            & np.all(geometry <= high[None, :] + 1.0e-10, axis=1)
            & (geometry.sum(axis=1) <= outer_max + 1.0e-10)
        )
        fractional = np.any(np.abs(geometry - np.round(geometry)) > 1.0e-8, axis=1)
        rejected_geometry += int(np.count_nonzero(~legal))
        rejected_integer_grid += int(np.count_nonzero(legal & ~fractional))
        preliminary = geometry[legal & fractional]
        diagnostics = solver_singularity_diagnostics(
            preliminary,
            wavelengths_nm,
            singularity_tolerance,
        )
        rejected_singular += int(np.count_nonzero(~diagnostics["is_safe"]))
        for row in preliminary[diagnostics["is_safe"]]:
            key = tuple(np.round(row / quantization_nm).astype(np.int64).tolist())
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            rows.append(row.copy())
            if len(rows) == int(n_samples):
                break
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (int(n_samples), 4):
        raise AssertionError("Sobol geometry output has the wrong shape.")
    if not np.all(geometry_bounds_mask(result)):
        raise AssertionError("Sobol sampler produced an illegal geometry.")
    final_diagnostics = solver_singularity_diagnostics(
        result,
        wavelengths_nm,
        singularity_tolerance,
    )
    if not np.all(final_diagnostics["is_safe"]):
        raise AssertionError("Sobol sampler retained a solver-singular geometry.")
    sampling = {
        "examined_sobol_points": int(examined),
        "attempts": int(attempts),
        "rejected_geometry": int(rejected_geometry),
        "rejected_integer_grid": int(rejected_integer_grid),
        "rejected_solver_singular": int(rejected_singular),
        "rejected_duplicate": int(duplicate_count),
        "retained": int(len(result)),
        "solver_singularity_tolerance": float(singularity_tolerance),
        "minimum_retained_abs_sin": float(np.min(final_diagnostics["min_abs_sin"])),
        "median_retained_abs_sin": float(np.median(final_diagnostics["min_abs_sin"])),
    }
    return result, sampling


def balanced_material_assignment(
    legal_sequences: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    repeats = int(math.ceil(n_samples / len(legal_sequences)))
    tiled = np.tile(np.asarray(legal_sequences, dtype=np.int64), (repeats, 1))[:n_samples]
    order = rng.permutation(n_samples)
    return tiled[order].copy()


def _save_pool_checkpoint(
    path: Path,
    *,
    completed_count: int,
    input_hash: str,
    qsca: np.ndarray,
    qabs: np.ndarray,
    qext: np.ndarray,
) -> None:
    save_npz_atomic(
        path,
        completed_count=np.asarray(int(completed_count), dtype=np.int64),
        input_hash=np.asarray(str(input_hash)),
        q_sca=np.asarray(qsca, dtype=np.float64),
        q_abs=np.asarray(qabs, dtype=np.float64),
        q_ext=np.asarray(qext, dtype=np.float64),
    )


def evaluate_pymiediff_pool(
    dataset: Mapping[str, Any],
    geometry: np.ndarray,
    materials: np.ndarray,
    device: torch.device,
    batch_size: int,
    protocol: TargetProtocol,
    *,
    checkpoint_path: Path,
    checkpoint_every_batches: int,
    restart_checkpoint: bool,
    failure_report_path: Path,
) -> Dict[str, np.ndarray]:
    n = len(geometry)
    w = len(dataset["wavelengths"])
    input_hash = array_hash(
        np.asarray(geometry, dtype="<f8"),
        np.asarray(materials, dtype="<i8"),
        np.asarray(dataset["wavelengths"], dtype="<f8"),
    )
    checkpoint_path = Path(checkpoint_path)
    if restart_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    completed = 0
    qsca = np.full((n, w), np.nan, dtype=np.float64)
    qabs = np.full_like(qsca, np.nan)
    qext = np.full_like(qsca, np.nan)
    if checkpoint_path.is_file():
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            saved_hash = str(np.asarray(archive["input_hash"]).item())
            if saved_hash != input_hash:
                raise RuntimeError(
                    "Existing private pool checkpoint belongs to different geometry/material inputs. "
                    "Use --restart-pool-evaluation to discard it."
                )
            completed = int(np.asarray(archive["completed_count"]).item())
            qsca_saved = np.asarray(archive["q_sca"], dtype=np.float64)
            qabs_saved = np.asarray(archive["q_abs"], dtype=np.float64)
            qext_saved = np.asarray(archive["q_ext"], dtype=np.float64)
            if qsca_saved.shape != (n, w) or qabs_saved.shape != (n, w) or qext_saved.shape != (n, w):
                raise RuntimeError("Existing pool checkpoint has incompatible array shapes.")
            qsca[:] = qsca_saved
            qabs[:] = qabs_saved
            qext[:] = qext_saved
        if completed < 0 or completed > n:
            raise RuntimeError("Existing pool checkpoint has an invalid completed_count.")
        if completed:
            print(f"Resuming private PyMieDiff pool checkpoint at {completed}/{n}.", flush=True)

    with torch.no_grad():
        batch_counter = 0
        for begin in range(completed, n, int(batch_size)):
            end = min(begin + int(batch_size), n)
            result = pymiediff_forward(
                torch.as_tensor(geometry[begin:end], dtype=torch.float64, device=device),
                torch.as_tensor(materials[begin:end], dtype=torch.long, device=device),
                dataset,
                backend=protocol.pymiediff_backend,
                precision=protocol.pymiediff_precision,
            )
            qsca[begin:end] = result["q_sca"].detach().cpu().numpy()
            qabs[begin:end] = result["q_abs"].detach().cpu().numpy()
            qext[begin:end] = result["q_ext"].detach().cpu().numpy()
            batch_counter += 1
            print(f"PyMieDiff target pool: {end}/{n}", flush=True)
            if batch_counter % int(checkpoint_every_batches) == 0 or end == n:
                _save_pool_checkpoint(
                    checkpoint_path,
                    completed_count=end,
                    input_hash=input_hash,
                    qsca=qsca,
                    qabs=qabs,
                    qext=qext,
                )

    failure: Dict[str, Any] = {
        "input_hash": input_hash,
        "negative_tolerance": float(protocol.negative_efficiency_tolerance),
        "energy_residual_tolerance": float(protocol.energy_residual_tolerance),
    }
    for name, values in (("Qsca", qsca), ("Qabs", qabs), ("Qext", qext)):
        if not np.isfinite(values).all():
            location = np.argwhere(~np.isfinite(values))[0]
            failure.update(
                {
                    "failure": "non_finite_efficiency",
                    "quantity": name,
                    "candidate_index": int(location[0]),
                    "wavelength_index": int(location[1]),
                }
            )
            save_json_atomic(failure_report_path, failure)
            raise RuntimeError(f"{name} contains NaN or Inf; see {failure_report_path}.")
        minimum_location = np.unravel_index(int(np.argmin(values)), values.shape)
        minimum = float(values[minimum_location])
        if minimum < -float(protocol.negative_efficiency_tolerance):
            candidate_index, wavelength_index = map(int, minimum_location)
            singular = solver_singularity_diagnostics(
                geometry[candidate_index],
                dataset["wavelengths"],
                protocol.solver_singularity_tolerance,
            )
            failure.update(
                {
                    "failure": "materially_negative_efficiency",
                    "quantity": name,
                    "minimum": minimum,
                    "candidate_index": candidate_index,
                    "wavelength_index": wavelength_index,
                    "wavelength_nm": float(dataset["wavelengths"][wavelength_index]),
                    "geometry": np.asarray(geometry[candidate_index], dtype=np.float64).tolist(),
                    "material_indices": np.asarray(materials[candidate_index], dtype=np.int64).tolist(),
                    "outer_radius_nm": float(np.sum(geometry[candidate_index])),
                    "min_abs_sin": float(singular["min_abs_sin"][0]),
                    "critical_wavelength_nm": float(singular["critical_wavelength_nm"][0]),
                }
            )
            save_json_atomic(failure_report_path, failure)
            raise RuntimeError(
                f"{name} contains materially negative values: {minimum:.3e}; "
                f"diagnostics saved to {failure_report_path}."
            )

    raw_relative_residual = np.max(np.abs(qext - qsca - qabs), axis=1) / np.maximum(
        1.0, np.max(np.abs(qext), axis=1)
    )
    if float(raw_relative_residual.max()) > float(protocol.energy_residual_tolerance):
        candidate_index = int(np.argmax(raw_relative_residual))
        failure.update(
            {
                "failure": "energy_closure",
                "candidate_index": candidate_index,
                "maximum_relative_residual": float(raw_relative_residual[candidate_index]),
                "geometry": np.asarray(geometry[candidate_index], dtype=np.float64).tolist(),
                "material_indices": np.asarray(materials[candidate_index], dtype=np.int64).tolist(),
            }
        )
        save_json_atomic(failure_report_path, failure)
        raise RuntimeError(
            "PyMieDiff relative energy residual exceeds tolerance: "
            f"{raw_relative_residual.max():.3e}; see {failure_report_path}."
        )

    correction_counts = {
        "qsca": int(np.count_nonzero((qsca < 0.0) & (qsca >= -protocol.negative_efficiency_tolerance))),
        "qabs": int(np.count_nonzero((qabs < 0.0) & (qabs >= -protocol.negative_efficiency_tolerance))),
        "qext": int(np.count_nonzero((qext < 0.0) & (qext >= -protocol.negative_efficiency_tolerance))),
    }
    qsca = np.maximum(qsca, 0.0)
    qabs = np.maximum(qabs, 0.0)
    qext = qsca + qabs
    relative_residual = np.max(np.abs(qext - qsca - qabs), axis=1) / np.maximum(
        1.0, np.max(np.abs(qext), axis=1)
    )
    return {
        "q_sca": qsca,
        "q_abs": qabs,
        "q_ext": qext,
        "relative_energy_residual": relative_residual,
        "raw_relative_energy_residual": raw_relative_residual,
        "roundoff_correction_counts": correction_counts,
        "checkpoint_input_hash": np.asarray(input_hash),
    }


def pack_material_key(row: np.ndarray) -> Tuple[int, int, int, int]:
    return tuple(np.asarray(row, dtype=np.int64).tolist())


def nearest_same_material_geometry(
    database_geometry: np.ndarray,
    database_materials: np.ndarray,
    query_geometry: np.ndarray,
    query_materials: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    groups: Dict[Tuple[int, int, int, int], np.ndarray] = {}
    for key in {pack_material_key(row) for row in database_materials}:
        groups[key] = np.flatnonzero(
            np.all(database_materials == np.asarray(key, dtype=np.int64)[None, :], axis=1)
        )
    scale = UPPER_BOUNDS - LOWER_BOUNDS
    nearest_index = np.empty(len(query_geometry), dtype=np.int64)
    nearest_distance = np.empty(len(query_geometry), dtype=np.float64)
    for i, (geometry, material) in enumerate(zip(query_geometry, query_materials)):
        key = pack_material_key(material)
        indices = groups.get(key)
        if indices is None or len(indices) == 0:
            raise RuntimeError(f"No database geometry exists for legal material sequence {key}.")
        difference = (database_geometry[indices] - geometry[None, :]) / scale[None, :]
        distance = np.sqrt(np.sum(difference * difference, axis=1))
        local = int(np.argmin(distance))
        nearest_index[i] = int(indices[local])
        nearest_distance[i] = float(distance[local])
    return nearest_index, nearest_distance


def exact_nearest_spectral_distance(
    query_qsca: np.ndarray,
    database_qsca: np.ndarray,
    device: torch.device,
    query_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query_qsca, dtype=np.float64)
    database = np.asarray(database_qsca, dtype=np.float64)
    db = torch.as_tensor(database, dtype=torch.float64, device=device)
    db_mean_square = db.square().mean(dim=1)
    db_t = db.transpose(0, 1).contiguous()
    nearest_index = np.empty(len(query), dtype=np.int64)
    nearest_rmse = np.empty(len(query), dtype=np.float64)
    nearest_nrmse = np.empty(len(query), dtype=np.float64)
    n_wavelengths = query.shape[1]
    for batch_id, begin in enumerate(range(0, len(query), int(query_batch_size)), start=1):
        end = min(begin + int(query_batch_size), len(query))
        q = torch.as_tensor(query[begin:end], dtype=torch.float64, device=device)
        q_mean_square = q.square().mean(dim=1, keepdim=True)
        cross_mean = (q @ db_t) / float(n_wavelengths)
        mse = (q_mean_square + db_mean_square[None, :] - 2.0 * cross_mean).clamp_min(0.0)
        rmse = torch.sqrt(mse)
        values, indices = torch.min(rmse, dim=1)
        peak = q.amax(dim=1).clamp_min(1.0e-12)
        nearest_index[begin:end] = indices.cpu().numpy().astype(np.int64)
        nearest_rmse[begin:end] = values.cpu().numpy()
        nearest_nrmse[begin:end] = (values / peak).cpu().numpy()
        print(f"Exact spectral NN batch {batch_id}: {end}/{len(query)}", flush=True)
    return nearest_index, nearest_rmse, nearest_nrmse


def empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=np.float64))
    if len(sorted_reference) == 0:
        raise ValueError("Percentile reference is empty.")
    return np.searchsorted(sorted_reference, values, side="right") / float(len(sorted_reference))


def difficulty_labels(percentile: np.ndarray, protocol: TargetProtocol) -> np.ndarray:
    labels = np.full(len(percentile), "excluded", dtype="U16")
    bands = {
        "easy": (protocol.easy_percentile_low, protocol.easy_percentile_high),
        "medium": (protocol.medium_percentile_low, protocol.medium_percentile_high),
        "hard": (protocol.hard_percentile_low, protocol.hard_percentile_high),
    }
    for name, (low, high) in bands.items():
        labels[(percentile >= low) & (percentile < high)] = name
    return labels


def candidate_features(qsca: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.max(qsca, axis=1, keepdims=True), 1.0e-12)
    normalized = qsca / scale
    basic = np.column_stack(
        (
            np.mean(qsca, axis=1),
            np.std(qsca, axis=1),
            np.ptp(qsca, axis=1),
            wavelengths[np.argmax(qsca, axis=1)],
        )
    )
    basic = (basic - basic.mean(axis=0, keepdims=True)) / np.maximum(
        basic.std(axis=0, keepdims=True), 1.0e-12
    )
    return np.column_stack((normalized, basic))


def farthest_point_order(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if len(x) == 0:
        return np.empty(0, dtype=np.int64)
    center = x.mean(axis=0)
    first = int(np.argmax(np.sum((x - center[None, :]) ** 2, axis=1)))
    selected = [first]
    minimum_distance = np.sum((x - x[first][None, :]) ** 2, axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < len(x):
        nxt = int(np.argmax(minimum_distance))
        if not np.isfinite(minimum_distance[nxt]):
            break
        selected.append(nxt)
        distance = np.sum((x - x[nxt][None, :]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected, dtype=np.int64)] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def select_diverse_targets(
    eligible: pd.DataFrame,
    qsca: np.ndarray,
    wavelengths: np.ndarray,
    protocol: TargetProtocol,
) -> np.ndarray:
    desired = {
        "easy": protocol.easy_count,
        "medium": protocol.medium_count,
        "hard": protocol.hard_count,
    }
    selected: list[int] = []
    material_counts: Counter[str] = Counter()
    for level in ("easy", "medium", "hard"):
        rows = eligible.index[eligible["difficulty"] == level].to_numpy(dtype=np.int64)
        if len(rows) < desired[level]:
            raise RuntimeError(
                f"Only {len(rows)} eligible {level} candidates; need {desired[level]}. "
                "Increase --pool-size or inspect diagnostics."
            )
        order = farthest_point_order(candidate_features(qsca[rows], wavelengths))
        chosen = 0
        for local in order:
            index = int(rows[int(local)])
            key = str(eligible.loc[index, "material_key"])
            if material_counts[key] >= protocol.maximum_targets_per_material_sequence:
                continue
            selected.append(index)
            material_counts[key] += 1
            chosen += 1
            if chosen == desired[level]:
                break
        if chosen != desired[level]:
            raise RuntimeError(f"Material cap prevented selecting enough {level} targets.")

    while len({str(eligible.loc[i, "material_key"]) for i in selected}) < protocol.minimum_unique_material_sequences:
        counts = Counter(str(eligible.loc[i, "material_key"]) for i in selected)
        used = {str(eligible.loc[i, "material_key"]) for i in selected}
        changed = False
        for position, old_index in enumerate(list(selected)):
            old_key = str(eligible.loc[old_index, "material_key"])
            if counts[old_key] <= 1:
                continue
            level = str(eligible.loc[old_index, "difficulty"])
            candidates = eligible.index[
                (eligible["difficulty"] == level)
                & (~eligible.index.isin(selected))
                & (~eligible["material_key"].isin(used))
            ].to_numpy(dtype=np.int64)
            if len(candidates) == 0:
                continue
            selected_norm = qsca[np.asarray(selected)] / np.maximum(
                np.max(qsca[np.asarray(selected)], axis=1, keepdims=True), 1.0e-12
            )
            candidate_norm = qsca[candidates] / np.maximum(
                np.max(qsca[candidates], axis=1, keepdims=True), 1.0e-12
            )
            distance = np.min(
                np.sqrt(
                    np.mean(
                        (candidate_norm[:, None, :] - selected_norm[None, :, :]) ** 2,
                        axis=2,
                    )
                ),
                axis=1,
            )
            selected[position] = int(candidates[int(np.argmax(distance))])
            changed = True
            break
        if not changed:
            raise RuntimeError("Could not satisfy the minimum unique-material requirement.")
    return np.asarray(selected, dtype=np.int64)


def peak_metadata(qsca: np.ndarray, wavelengths: np.ndarray) -> Dict[str, Any]:
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise ImportError("scipy is required for formal peak metadata.") from exc
    target = np.asarray(qsca, dtype=np.float64)
    dynamic_range = float(np.ptp(target))
    prominence = max(0.05 * dynamic_range, 1.0e-6)
    peaks, properties = find_peaks(target, prominence=prominence)
    global_index = int(np.argmax(target))
    return {
        "peak_criterion_applicable": bool(len(peaks) == 1 and int(peaks[0]) == global_index),
        "significant_peak_count": int(len(peaks)),
        "target_peak_index": global_index,
        "target_peak_nm": float(wavelengths[global_index]),
        "target_peak_qsca": float(target[global_index]),
        "peak_prominence_threshold": float(prominence),
        "significant_peak_indices": peaks.astype(np.int64),
        "significant_peak_prominences": np.asarray(
            properties.get("prominences", np.empty(0)), dtype=np.float64
        ),
    }


def plot_targets(
    figure_dir: Path,
    wavelengths: np.ndarray,
    task_ids: np.ndarray,
    difficulty: np.ndarray,
    qsca: np.ndarray,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.4,
            "font.size": 10,
        }
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    for task_id, level, spectrum in zip(task_ids, difficulty, qsca):
        ax.plot(wavelengths, spectrum, linewidth=1.6, label=f"{task_id} ({level})")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Scattering efficiency, Qsca")
    ax.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
    ax.legend(ncol=2, fontsize=7.5, frameon=True)
    fig.tight_layout()
    fig.savefig(figure_dir / "blind_targets_overview.png", dpi=600, bbox_inches="tight")
    fig.savefig(figure_dir / "blind_targets_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    for task_id, level, spectrum in zip(task_ids, difficulty, qsca):
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        ax.plot(wavelengths, spectrum, linewidth=2.0)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Scattering efficiency, Qsca")
        ax.set_title(f"{task_id} — {level}")
        ax.set_xlim(float(wavelengths[0]), float(wavelengths[-1]))
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task_id}.png", dpi=600, bbox_inches="tight")
        fig.savefig(figure_dir / f"{task_id}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    start = time.perf_counter()
    protocol = TargetProtocol(
        pool_size=int(args.pool_size),
        seed=int(args.seed),
        solver_singularity_tolerance=float(args.solver_singularity_tolerance),
    )
    if protocol.easy_count + protocol.medium_count + protocol.hard_count != protocol.target_count:
        raise RuntimeError("Difficulty composition does not sum to ten targets.")

    set_seed(protocol.seed)
    rng = np.random.default_rng(protocol.seed)
    stage02 = verify_stage02_frozen(args.stage02_dir)
    archive_dir = archive_or_prepare(
        (args.public_dir, args.private_dir, args.diagnostics_dir, args.figure_dir),
        args.archive_root,
        args.overwrite,
    )

    dataset = load_forward_dataset(args.dataset)
    material_frame = enumerate_legal_material_sequences(dataset["material_names"])
    legal_sequences = material_frame[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64)
    if legal_sequences.shape != (400, 4):
        raise RuntimeError(f"Expected 400 legal material sequences, got {legal_sequences.shape}.")
    if not np.all(material_sequence_mask(legal_sequences, dataset["material_names"])):
        raise RuntimeError("Material registry contains an illegal sequence.")

    print("Stage 02B strict blind target generation")
    print(f"Target pool: {protocol.pool_size}; seed: {protocol.seed}")
    print(f"PyMieDiff device: {resolve_device(args.device)}")
    print(f"Spectral-distance device: {resolve_device(args.distance_device)}")
    print("Inverse network and PGFF are not loaded by this stage.")

    geometry, sampling_diagnostics = sample_sobol_geometries(
        protocol.pool_size,
        protocol.seed,
        protocol.offgrid_quantization_nm,
        protocol.boundary_margin_nm,
        dataset["wavelengths"],
        protocol.solver_singularity_tolerance,
    )
    materials = balanced_material_assignment(legal_sequences, protocol.pool_size, rng)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pool_input_path = work_dir / "blind_target_pool_inputs.npz"
    save_npz_atomic(
        pool_input_path,
        geometry=geometry,
        material_indices=materials,
        wavelengths_nm=np.asarray(dataset["wavelengths"], dtype=np.float64),
        sampling_diagnostics_json=np.asarray(
            json.dumps(sampling_diagnostics, sort_keys=True, separators=(",", ":"))
        ),
    )
    print(
        "Solver-singularity guard: "
        f"rejected={sampling_diagnostics['rejected_solver_singular']}, "
        f"minimum retained |sin(x)|={sampling_diagnostics['minimum_retained_abs_sin']:.3e}"
    )
    pymie_device = resolve_device(args.device)
    result = evaluate_pymiediff_pool(
        dataset,
        geometry,
        materials,
        pymie_device,
        args.pymiediff_batch_size,
        protocol,
        checkpoint_path=work_dir / "blind_target_candidate_pool.npz",
        checkpoint_every_batches=args.checkpoint_every_batches,
        restart_checkpoint=args.restart_pool_evaluation,
        failure_report_path=work_dir / "pymiediff_validation_failure_private.json",
    )

    geometry_nn_index, geometry_nn_distance = nearest_same_material_geometry(
        np.asarray(dataset["geometry"], dtype=np.float64),
        np.asarray(dataset["material_indices"], dtype=np.int64),
        geometry,
        materials,
    )
    spectral_nn_index, spectral_nn_rmse, spectral_nn_nrmse = exact_nearest_spectral_distance(
        result["q_sca"],
        np.asarray(dataset["Q_sca"], dtype=np.float64),
        resolve_device(args.distance_device),
        args.spectral_query_batch_size,
    )

    permutation = rng.permutation(protocol.pool_size)
    calibration_count = max(256, int(round(protocol.pool_size * protocol.calibration_fraction)))
    calibration_index = permutation[:calibration_count]
    selection_index = permutation[calibration_count:]
    subset = np.full(protocol.pool_size, "selection", dtype="U16")
    subset[calibration_index] = "calibration"

    geometry_percentile = empirical_percentile(
        geometry_nn_distance, geometry_nn_distance[calibration_index]
    )
    spectral_percentile = empirical_percentile(
        spectral_nn_nrmse, spectral_nn_nrmse[calibration_index]
    )
    difficulty = difficulty_labels(spectral_percentile, protocol)

    database_peak = np.max(dataset["Q_sca"], axis=1)
    database_range = np.ptp(dataset["Q_sca"], axis=1)
    peak_low, peak_high = np.quantile(
        database_peak, protocol.peak_amplitude_database_quantiles
    )
    range_min = float(np.quantile(database_range, protocol.dynamic_range_database_quantile_min))
    pool_peak = np.max(result["q_sca"], axis=1)
    pool_range = np.ptp(result["q_sca"], axis=1)

    material_names = np.asarray(dataset["material_names"]).astype(str)
    material_key = np.asarray(["-".join(map(str, row)) for row in materials], dtype="U32")
    material_label = np.asarray(
        ["|".join(material_names[row].tolist()) for row in materials], dtype="U128"
    )
    exact_geometry_duplicate = geometry_nn_distance <= 1.0e-12
    eligible = (
        (subset == "selection")
        & (~exact_geometry_duplicate)
        & (geometry_percentile >= protocol.geometry_percentile_min)
        & (pool_peak >= float(peak_low))
        & (pool_peak <= float(peak_high))
        & (pool_range >= range_min)
        & np.isin(difficulty, ["easy", "medium", "hard"])
    )

    frame = pd.DataFrame(
        {
            "pool_index": np.arange(protocol.pool_size, dtype=np.int64),
            "subset": subset,
            "difficulty": difficulty,
            "eligible": eligible,
            "material_key": material_key,
            "materials": material_label,
            "m0": materials[:, 0],
            "m1": materials[:, 1],
            "m2": materials[:, 2],
            "m3": materials[:, 3],
            "r_core": geometry[:, 0],
            "t1": geometry[:, 1],
            "t2": geometry[:, 2],
            "t3": geometry[:, 3],
            "outer_radius": geometry.sum(axis=1),
            "geometry_nn_dataset_index": geometry_nn_index,
            "geometry_nn_distance_standardized": geometry_nn_distance,
            "geometry_nn_percentile": geometry_percentile,
            "spectral_nn_dataset_index": spectral_nn_index,
            "spectral_nn_rmse": spectral_nn_rmse,
            "spectral_nn_nrmse": spectral_nn_nrmse,
            "spectral_nn_percentile": spectral_percentile,
            "peak_qsca": pool_peak,
            "dynamic_range_qsca": pool_range,
            "relative_energy_residual": result["relative_energy_residual"],
        }
    )
    eligible_frame = frame.loc[eligible].copy()
    if eligible_frame.empty:
        raise RuntimeError("No candidate passed the formal target filters.")
    selected_indices = select_diverse_targets(
        eligible_frame, result["q_sca"], dataset["wavelengths"], protocol
    )
    selected = frame.loc[selected_indices].copy()
    order_map = {"easy": 0, "medium": 1, "hard": 2}
    selected["difficulty_order"] = selected["difficulty"].map(order_map).astype(int)
    selected = selected.sort_values(
        ["difficulty_order", "spectral_nn_percentile", "material_key"],
        kind="mergesort",
    ).drop(columns="difficulty_order").reset_index(drop=True)
    selected_pool_index = selected["pool_index"].to_numpy(dtype=np.int64)

    expected_counts = {"easy": 3, "medium": 4, "hard": 3}
    actual_counts = selected["difficulty"].value_counts().to_dict()
    if any(int(actual_counts.get(key, 0)) != value for key, value in expected_counts.items()):
        raise RuntimeError(f"Selected difficulty composition is wrong: {actual_counts}")
    unique_materials = int(selected["material_key"].nunique())
    maximum_material_use = int(selected["material_key"].value_counts().max())
    if unique_materials < protocol.minimum_unique_material_sequences:
        raise RuntimeError("Selected targets do not satisfy material-sequence diversity.")
    if maximum_material_use > protocol.maximum_targets_per_material_sequence:
        raise RuntimeError("A material sequence exceeds the maximum target-use cap.")

    task_ids = np.asarray(
        [f"BLIND_{i:02d}" for i in range(1, protocol.target_count + 1)], dtype="U32"
    )
    target_qsca = result["q_sca"][selected_pool_index]
    target_qabs = result["q_abs"][selected_pool_index]
    target_qext = result["q_ext"][selected_pool_index]
    reference_geometry = geometry[selected_pool_index]
    reference_materials = materials[selected_pool_index]
    hashes = np.asarray(
        [spectrum_hash(dataset["wavelengths"], row) for row in target_qsca], dtype="U64"
    )
    metadata = [peak_metadata(row, dataset["wavelengths"]) for row in target_qsca]
    benchmark_id = stable_hash(
        {
            "version": VERSION,
            "protocol": asdict(protocol),
            "dataset_sha256": sha256_file(args.dataset),
            "stage02_best_model_sha256": stage02["best_model"]["sha256"],
            "target_hashes": hashes.tolist(),
        }
    )[:24]

    public_path = Path(args.public_dir) / "blind_target_spectra.npz"
    save_npz_atomic(
        public_path,
        task_id=task_ids,
        wavelengths_nm=np.asarray(dataset["wavelengths"], dtype=np.float64),
        target_qsca=target_qsca,
        difficulty=selected["difficulty"].astype(str).to_numpy(dtype="U16"),
        target_hash=hashes,
        benchmark_id=np.asarray(benchmark_id),
        peak_criterion_applicable=np.asarray(
            [value["peak_criterion_applicable"] for value in metadata], dtype=bool
        ),
        target_peak_nm=np.asarray([value["target_peak_nm"] for value in metadata]),
        significant_peak_count=np.asarray(
            [value["significant_peak_count"] for value in metadata], dtype=np.int64
        ),
        spectral_nn_percentile=selected["spectral_nn_percentile"].to_numpy(dtype=np.float64),
        geometry_nn_percentile=selected["geometry_nn_percentile"].to_numpy(dtype=np.float64),
    )

    public_summary = pd.DataFrame(
        {
            "task_id": task_ids,
            "difficulty": selected["difficulty"].astype(str).to_numpy(),
            "target_hash": hashes,
            "benchmark_id": benchmark_id,
            "target_peak_qsca": [value["target_peak_qsca"] for value in metadata],
            "target_peak_nm": [value["target_peak_nm"] for value in metadata],
            "significant_peak_count": [value["significant_peak_count"] for value in metadata],
            "peak_criterion_applicable": [
                value["peak_criterion_applicable"] for value in metadata
            ],
            "dynamic_range_qsca": np.ptp(target_qsca, axis=1),
            "spectral_nn_percentile": selected["spectral_nn_percentile"].to_numpy(),
            "geometry_nn_percentile": selected["geometry_nn_percentile"].to_numpy(),
        }
    )
    atomic_write_csv(Path(args.public_dir) / "blind_target_public_summary.csv", public_summary)

    private_path = Path(args.private_dir) / "hidden_reference_structures.npz"
    save_npz_atomic(
        private_path,
        task_id=task_ids,
        benchmark_id=np.asarray(benchmark_id),
        reference_geometry=reference_geometry,
        reference_radii=np.cumsum(reference_geometry, axis=1),
        reference_material_indices=reference_materials,
        reference_material_names=np.asarray(
            [[material_names[int(v)] for v in row] for row in reference_materials], dtype="U64"
        ),
        reference_qsca=target_qsca,
        reference_qabs=target_qabs,
        reference_qext=target_qext,
        target_hash=hashes,
        generation_seed=np.asarray(protocol.seed, dtype=np.int64),
        selected_pool_index=selected_pool_index,
        geometry_nn_dataset_index=selected["geometry_nn_dataset_index"].to_numpy(dtype=np.int64),
        spectral_nn_dataset_index=selected["spectral_nn_dataset_index"].to_numpy(dtype=np.int64),
        spectral_nn_nrmse=selected["spectral_nn_nrmse"].to_numpy(dtype=np.float64),
    )
    selected_private = selected.copy()
    selected_private.insert(0, "task_id", task_ids)
    selected_private.insert(1, "target_hash", hashes)
    selected_private.insert(2, "benchmark_id", benchmark_id)
    atomic_write_csv(Path(args.private_dir) / "hidden_reference_summary.csv", selected_private)
    with open(Path(args.private_dir) / "private_reference_access_policy.txt", "w", encoding="utf-8") as handle:
        handle.write(
            "This directory contains the hidden generating structures. Do not open or read it "
            "during Stage 03 or PyMieDiff refinement. Unblind only after all inverse results are frozen.\n"
        )

    atomic_write_csv(Path(args.diagnostics_dir) / "target_pool_diagnostics.csv", frame)
    if args.save_pool_spectra:
        save_npz_atomic(
            Path(args.diagnostics_dir) / "target_pool_spectra.npz",
            wavelengths_nm=np.asarray(dataset["wavelengths"], dtype=np.float64),
            geometry=geometry,
            material_indices=materials,
            q_sca=result["q_sca"],
            q_abs=result["q_abs"],
            q_ext=result["q_ext"],
        )
    if not args.no_plots:
        plot_targets(
            args.figure_dir,
            dataset["wavelengths"],
            task_ids,
            selected["difficulty"].astype(str).to_numpy(),
            target_qsca,
        )

    manifest_path = Path(args.public_dir) / "blind_benchmark_manifest.json"
    manifest = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "benchmark_id": benchmark_id,
        "wall_time_seconds": float(time.perf_counter() - start),
        "protocol": asdict(protocol),
        "scientific_design": {
            "benchmark_generated_after_inverse_model_frozen": True,
            "inverse_network_loaded": False,
            "pgff_loaded": False,
            "legacy_inverse_results_read": False,
            "public_inverse_input": "Qsca only",
            "calibration_and_selection_subsets_disjoint": True,
            "database_external_definition": (
                "off-integer-grid physical geometry, no exact same-material geometry duplicate, "
                "solver-singularity-safe PyMieDiff geometry, and quantified exact nearest "
                "spectral/geometry distances to the full database"
            ),
            "pymiediff_solver_singularity_guard": True,
            "singularity_definition": "min |sin(2*pi*Rout/lambda)| > frozen tolerance",
            "difficulty_composition": expected_counts,
            "minimum_unique_material_sequences": protocol.minimum_unique_material_sequences,
            "maximum_targets_per_material_sequence": protocol.maximum_targets_per_material_sequence,
        },
        "frozen_stage02": stage02,
        "inputs": {"dataset": artifact(args.dataset)},
        "devices": {
            "pymiediff": str(pymie_device),
            "spectral_distance": str(resolve_device(args.distance_device)),
        },
        "counts": {
            "database_samples": int(len(dataset["geometry"])),
            "candidate_pool": int(protocol.pool_size),
            "calibration": int(calibration_count),
            "selection": int(len(selection_index)),
            "eligible": int(eligible.sum()),
            "selected": int(len(selected)),
            "unique_selected_material_sequences": unique_materials,
            "maximum_selected_material_sequence_use": maximum_material_use,
            "solver_singular_sobol_candidates_rejected": int(
                sampling_diagnostics["rejected_solver_singular"]
            ),
        },
        "sampling_diagnostics": sampling_diagnostics,
        "pymiediff_validation": {
            "maximum_raw_relative_energy_residual": float(
                np.max(result["raw_relative_energy_residual"])
            ),
            "roundoff_correction_counts": result["roundoff_correction_counts"],
        },
        "database_thresholds": {
            "peak_qsca_p10": float(peak_low),
            "peak_qsca_p90": float(peak_high),
            "dynamic_range_qsca_p25": float(range_min),
        },
        "outputs": {
            "public_targets": artifact(public_path),
            "public_summary": artifact(Path(args.public_dir) / "blind_target_public_summary.csv"),
            "private_reference": artifact(private_path),
            "pool_diagnostics": artifact(Path(args.diagnostics_dir) / "target_pool_diagnostics.csv"),
            "private_pool_inputs": artifact(pool_input_path),
            "private_pool_checkpoint": artifact(
                work_dir / "blind_target_candidate_pool.npz"
            ),
        },
        "archived_previous_benchmark": str(archive_dir.resolve()) if archive_dir else None,
    }
    save_json_atomic(manifest_path, manifest)
    save_json_atomic(
        Path(args.public_dir) / "blind_benchmark_lock.json",
        {
            "benchmark_id": benchmark_id,
            "public_target_sha256": sha256_file(public_path),
            "manifest_sha256": sha256_file(manifest_path),
            "frozen": True,
            "instruction": "Do not regenerate after beginning Stage 03 formal inference.",
        },
    )

    print("Stage 02B strict blind target generation PASSED")
    print(f"Benchmark ID: {benchmark_id}")
    print(f"Public targets: {public_path}")
    print(f"Private reference: {private_path}")
    print(f"Eligible candidates: {int(eligible.sum())}/{protocol.pool_size}")
    print(f"Selected difficulty: {actual_counts}")
    print(f"Selected unique material sequences: {unique_materials}/10")
    print(
        "Solver-singular candidates rejected before PyMieDiff: "
        f"{sampling_diagnostics['rejected_solver_singular']}"
    )
    print("The inverse network and PGFF were not used to construct or select targets.")
    print("You may now run Stage 03 using only blind_target_spectra.npz.")


if __name__ == "__main__":
    main()
