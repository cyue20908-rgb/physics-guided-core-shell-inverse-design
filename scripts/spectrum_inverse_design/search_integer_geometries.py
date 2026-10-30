# -*- coding: utf-8 -*-
"""Stage 05: strict-blind integer-nanometer PyMieDiff validation.

This stage converts the selected continuous Stage-04 candidates into a local
1-nm manufacturing grid, evaluates every unique legal integer design directly
with double-precision PyMieDiff, and freezes the best blind result for each
public target.

Strict-blind policy
-------------------
* consumes public target Qsca embedded in the Stage-04 selected-candidate NPZ;
* keeps each discrete material sequence fixed;
* does not load PGFF, the inverse network, hidden reference structures,
  target Qabs/Qext, or train/validation/test spectra;
* reads only wavelength and material optical tables from the forward asset;
* rejects Peña-recursion singularities rather than nudging integer geometry;
* ranks final designs only by high-fidelity PyMieDiff metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import hashlib
import json
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch

from core_shell_optical_design.inverse_design.inverse_design_data import (
    DEFAULT_DATASET,
    DEFAULT_RESULT_ROOT,
    artifact,
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
    geometry_bounds_mask,
    material_sequence_mask,
)
from core_shell_optical_design.inverse_design.inverse_design_objectives import per_sample_metrics

STAGE = "search_integer_geometries"
VERSION = "tandem-mie-inverse-2026-07-27-stage05-v1"

DEFAULT_STAGE04_DIR = DEFAULT_RESULT_ROOT / "continuous_geometry_refinement"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "integer_geometry_search"


@dataclass(frozen=True)
class IntegerProtocol:
    integer_radius_nm: int = 2
    batch_size: int = 64
    final_top_k: int = 10
    pymiediff_backend: str = "pena"
    pymiediff_precision: str = "double"
    solver_singularity_tolerance: float = 1.0e-6
    negative_efficiency_tolerance: float = 1.0e-7
    energy_residual_tolerance: float = 1.0e-8
    success_nmae: float = 0.05
    success_nrmse: float = 0.05
    success_r2: float = 0.99
    high_accuracy_nmae: float = 0.02
    high_accuracy_nrmse: float = 0.02
    peak_error_tolerance_nm: float = 6.0
    dominant_peak_ratio: float = 1.10
    seed: int = 20260727


def parse_args() -> argparse.Namespace:
    p = IntegerProtocol()
    parser = argparse.ArgumentParser(
        description="Strict-blind integer-nanometer PyMieDiff validation"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--stage04-dir", type=Path, default=DEFAULT_STAGE04_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--integer-radius", type=int, default=p.integer_radius_nm)
    parser.add_argument("--batch-size", type=int, default=p.batch_size)
    parser.add_argument("--final-top-k", type=int, default=p.final_top_k)
    parser.add_argument(
        "--solver-singularity-tolerance",
        type=float,
        default=p.solver_singularity_tolerance,
    )
    parser.add_argument(
        "--negative-efficiency-tolerance",
        type=float,
        default=p.negative_efficiency_tolerance,
    )
    parser.add_argument(
        "--energy-residual-tolerance",
        type=float,
        default=p.energy_residual_tolerance,
    )
    parser.add_argument("--success-nmae", type=float, default=p.success_nmae)
    parser.add_argument("--success-nrmse", type=float, default=p.success_nrmse)
    parser.add_argument("--success-r2", type=float, default=p.success_r2)
    parser.add_argument(
        "--high-accuracy-nmae", type=float, default=p.high_accuracy_nmae
    )
    parser.add_argument(
        "--high-accuracy-nrmse", type=float, default=p.high_accuracy_nrmse
    )
    parser.add_argument(
        "--peak-error-tolerance-nm",
        type=float,
        default=p.peak_error_tolerance_nm,
    )
    parser.add_argument(
        "--dominant-peak-ratio", type=float, default=p.dominant_peak_ratio
    )
    parser.add_argument("--seed", type=int, default=p.seed)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.integer_radius < 0 or args.integer_radius > 10:
        raise ValueError("--integer-radius must be between 0 and 10 nm.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not 1 <= args.final_top_k <= 100:
        raise ValueError("--final-top-k must be between 1 and 100.")
    for name in (
        "solver_singularity_tolerance",
        "negative_efficiency_tolerance",
        "energy_residual_tolerance",
        "success_nmae",
        "success_nrmse",
        "high_accuracy_nmae",
        "high_accuracy_nrmse",
        "peak_error_tolerance_nm",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if not np.isfinite(args.success_r2) or args.success_r2 > 1.0:
        raise ValueError("--success-r2 must be finite and <= 1.")
    if not np.isfinite(args.dominant_peak_ratio) or args.dominant_peak_ratio <= 1.0:
        raise ValueError("--dominant-peak-ratio must exceed 1.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value


def stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def prepare_output_dir(path: Path, *, resume: bool, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists() and not resume:
        existing = [item for item in path.iterdir()]
        if existing:
            raise FileExistsError(
                f"Output directory already contains results: {path}\n"
                "Use --resume to continue or --overwrite to restart."
            )
    path.mkdir(parents=True, exist_ok=True)
    (path / "task_checkpoints").mkdir(exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)


def load_physics_assets(path: Path) -> Dict[str, Any]:
    """Read only optical tables; do not load database spectra or split indices."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Forward asset not found: {path}")
    required = ("wavelengths", "material_names", "material_n", "material_k")
    with np.load(path, allow_pickle=True) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Forward asset is missing fields: {missing}")
        wavelengths = np.asarray(archive["wavelengths"], dtype=np.float64)
        names = np.asarray(archive["material_names"], dtype=str)
        material_n = np.asarray(archive["material_n"], dtype=np.float64)
        material_k = np.asarray(archive["material_k"], dtype=np.float64)
    if wavelengths.ndim != 1 or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("Wavelengths must be one-dimensional and strictly increasing.")
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


def verify_stage04(stage04_dir: Path, dataset_path: Path) -> Dict[str, Any]:
    stage04_dir = Path(stage04_dir)
    if "private" in {part.strip().lower() for part in stage04_dir.parts}:
        raise RuntimeError("Stage 05 refuses a Stage-04 path containing 'private'.")
    summary_path = stage04_dir / "continuous_refinement_run_summary.json"
    selected_path = stage04_dir / "selected_integer_search_candidates.npz"
    selected_csv = stage04_dir / "selected_integer_search_candidates.csv"
    if not selected_path.is_file() or not selected_csv.is_file():
        raise FileNotFoundError(
            "Stage 04 selected-candidate artifacts are incomplete. Expected:\n"
            f"{selected_path}\n{selected_csv}"
        )
    summary = load_json(summary_path)
    if summary.get("status") != "passed":
        raise RuntimeError("Stage 04 summary is not a passed run.")

    policy = summary.get("blind_policy", {})
    forbidden_true = (
        "reference_geometry_read",
        "reference_materials_read",
        "target_qabs_read",
        "target_qext_read",
        "pgff_loaded",
        "inverse_network_loaded",
        "forward_spectra_read",
        "test_split_read",
    )
    violated = [name for name in forbidden_true if bool(policy.get(name, False))]
    if violated:
        raise RuntimeError(f"Stage 04 blind policy is invalid: {violated}")

    outputs = summary.get("outputs", {})
    recorded = outputs.get("selected_stage05_npz", {})
    if recorded and str(recorded.get("sha256", "")) != sha256_file(selected_path):
        raise RuntimeError("Stage-04 selected NPZ hash differs from continuous_refinement_run_summary.json.")
    inputs = summary.get("inputs", {})
    recorded_dataset = inputs.get("forward_physics_asset", inputs.get("dataset", {}))
    if recorded_dataset and str(recorded_dataset.get("sha256", "")) not in {"", sha256_file(dataset_path)}:
        raise RuntimeError("Forward dataset differs from the dataset used by Stage 04.")
    return {
        "summary": summary,
        "summary_path": summary_path,
        "selected_path": selected_path,
        "selected_csv": selected_csv,
    }


def load_stage04_candidates(
    path: Path,
    wavelengths_expected: np.ndarray,
    material_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    required = (
        "task_id",
        "wavelengths_nm",
        "target_qsca",
        "target_hash",
        "candidate_task_index",
        "candidate_continuous_rank",
        "seed_candidate_uid",
        "candidate_sequence_id",
        "candidate_restart_index",
        "candidate_material_indices",
        "candidate_continuous_geometry",
        "candidate_mie_qsca",
        "candidate_mie_nmae",
        "candidate_mie_nrmse",
        "candidate_mie_r2",
        "candidate_mie_peak_error_nm",
        "candidate_min_abs_sin",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Stage 04 selected NPZ is missing fields: {missing}")
        data = {key: np.asarray(archive[key]) for key in required}

    task_id = data["task_id"].astype(str)
    wavelengths = data["wavelengths_nm"].astype(np.float64)
    target_qsca = data["target_qsca"].astype(np.float64)
    target_hash = data["target_hash"].astype(str)
    task_index = data["candidate_task_index"].astype(np.int64)
    continuous_rank = data["candidate_continuous_rank"].astype(np.int64)
    seed_uid = data["seed_candidate_uid"].astype(str)
    sequence_id = data["candidate_sequence_id"].astype(np.int64)
    restart_index = data["candidate_restart_index"].astype(np.int64)
    materials = data["candidate_material_indices"].astype(np.int64)
    geometry = data["candidate_continuous_geometry"].astype(np.float64)

    if not np.array_equal(wavelengths, np.asarray(wavelengths_expected, dtype=np.float64)):
        raise RuntimeError("Stage 04 and forward-physics wavelength grids differ.")
    if len(task_id) != 10:
        raise RuntimeError(f"Expected 10 blind targets, received {len(task_id)}.")
    if target_qsca.shape != (len(task_id), len(wavelengths)):
        raise ValueError("Target Qsca shape is incompatible with task IDs/wavelengths.")
    n = len(task_index)
    if n != 100:
        raise RuntimeError(f"Expected 100 selected continuous candidates, received {n}.")
    if materials.shape != (n, 4) or geometry.shape != (n, 4):
        raise ValueError("Stage-04 material/geometry shape mismatch.")
    if not np.isfinite(target_qsca).all() or float(target_qsca.min()) < -1.0e-10:
        raise ValueError("Public target Qsca contains invalid values.")
    if not np.isfinite(geometry).all() or not np.all(geometry_bounds_mask(geometry)):
        raise ValueError("Stage-04 continuous geometry is not legal.")
    if not np.all(material_sequence_mask(materials, material_names)):
        raise ValueError("Stage-04 material sequence is illegal.")
    for task in range(len(task_id)):
        rows = np.flatnonzero(task_index == task)
        if len(rows) != 10:
            raise RuntimeError(f"{task_id[task]} has {len(rows)} candidates, expected 10.")
        if sorted(continuous_rank[rows].tolist()) != list(range(1, 11)):
            raise RuntimeError(f"{task_id[task]} continuous ranks are not exactly 1..10.")

    return {
        "task_id": task_id,
        "wavelengths_nm": wavelengths,
        "target_qsca": target_qsca,
        "target_hash": target_hash,
        "candidate_task_index": task_index,
        "candidate_continuous_rank": continuous_rank,
        "seed_candidate_uid": seed_uid,
        "candidate_sequence_id": sequence_id,
        "candidate_restart_index": restart_index,
        "candidate_material_indices": materials,
        "candidate_continuous_geometry": geometry,
        "candidate_mie_qsca": data["candidate_mie_qsca"].astype(np.float64),
        "candidate_mie_nmae": data["candidate_mie_nmae"].astype(np.float64),
        "candidate_mie_nrmse": data["candidate_mie_nrmse"].astype(np.float64),
        "candidate_mie_r2": data["candidate_mie_r2"].astype(np.float64),
        "candidate_mie_peak_error_nm": data["candidate_mie_peak_error_nm"].astype(np.float64),
        "candidate_min_abs_sin": data["candidate_min_abs_sin"].astype(np.float64),
    }


def numpy_min_abs_sin(geometry: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    values = np.asarray(geometry, dtype=np.float64)
    outer = values.sum(axis=1)
    phase = 2.0 * np.pi * outer[:, None] / np.asarray(wavelengths_nm, dtype=np.float64)[None, :]
    return np.abs(np.sin(phase)).min(axis=1)


def task_input_hash(
    target_hash: str,
    materials: np.ndarray,
    geometry: np.ndarray,
    continuous_rank: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(target_hash).encode("utf-8"))
    digest.update(np.asarray(materials, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(geometry, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(continuous_rank, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def make_design_uid(task_id: str, materials: Sequence[int], geometry: Sequence[int]) -> str:
    text = (
        f"{task_id}|"
        + "-".join(str(int(x)) for x in materials)
        + "|"
        + "-".join(str(int(x)) for x in geometry)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def generate_integer_pool(
    *,
    task_id: str,
    materials: np.ndarray,
    continuous_geometry: np.ndarray,
    continuous_rank: np.ndarray,
    sequence_id: np.ndarray,
    seed_uid: np.ndarray,
    radius: int,
    wavelengths_nm: np.ndarray,
    singularity_tolerance: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    offsets = np.asarray(list(product(range(-radius, radius + 1), repeat=4)), dtype=np.int64)
    legal_rows: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    rejected_rows: list[Dict[str, Any]] = []
    lower = np.asarray(LOWER_BOUNDS, dtype=np.float64)
    upper = np.asarray(UPPER_BOUNDS, dtype=np.float64)

    for index in range(len(materials)):
        base = np.rint(continuous_geometry[index]).astype(np.int64)
        candidates = base[None, :] + offsets
        bounds_ok = ((candidates >= lower[None, :]) & (candidates <= upper[None, :])).all(axis=1)
        radius_ok = candidates.sum(axis=1) <= int(MAX_OUTER_RADIUS)
        candidates = candidates[bounds_ok & radius_ok]
        if len(candidates) == 0:
            raise RuntimeError(f"{task_id}: no legal integer neighbors for continuous rank {continuous_rank[index]}.")

        singularity = numpy_min_abs_sin(candidates.astype(np.float64), wavelengths_nm)
        safe = singularity > float(singularity_tolerance)
        for geometry, min_sin in zip(candidates[~safe], singularity[~safe]):
            rejected_rows.append(
                {
                    "task_id": task_id,
                    "continuous_rank": int(continuous_rank[index]),
                    "sequence_id": int(sequence_id[index]),
                    "m0": int(materials[index, 0]),
                    "m1": int(materials[index, 1]),
                    "m2": int(materials[index, 2]),
                    "m3": int(materials[index, 3]),
                    "r_core": int(geometry[0]),
                    "t1": int(geometry[1]),
                    "t2": int(geometry[2]),
                    "t3": int(geometry[3]),
                    "outer_radius": int(geometry.sum()),
                    "minimum_abs_sin": float(min_sin),
                    "rejection_reason": "pena_solver_singularity_guard",
                }
            )

        for geometry, min_sin in zip(candidates[safe], singularity[safe]):
            key = tuple(materials[index].astype(int).tolist() + geometry.astype(int).tolist())
            distance = float(np.linalg.norm(geometry.astype(np.float64) - continuous_geometry[index]))
            if key not in legal_rows:
                legal_rows[key] = {
                    "task_id": task_id,
                    "design_uid": make_design_uid(task_id, materials[index], geometry),
                    "sequence_id": int(sequence_id[index]),
                    "m0": int(materials[index, 0]),
                    "m1": int(materials[index, 1]),
                    "m2": int(materials[index, 2]),
                    "m3": int(materials[index, 3]),
                    "r_core": int(geometry[0]),
                    "t1": int(geometry[1]),
                    "t2": int(geometry[2]),
                    "t3": int(geometry[3]),
                    "outer_radius": int(geometry.sum()),
                    "minimum_abs_sin": float(min_sin),
                    "best_source_continuous_rank": int(continuous_rank[index]),
                    "best_source_seed_uid": str(seed_uid[index]),
                    "distance_from_source_nm": distance,
                    "source_count": 1,
                }
            else:
                row = legal_rows[key]
                row["source_count"] = int(row["source_count"]) + 1
                current = (distance, int(continuous_rank[index]))
                previous = (
                    float(row["distance_from_source_nm"]),
                    int(row["best_source_continuous_rank"]),
                )
                if current < previous:
                    row["best_source_continuous_rank"] = int(continuous_rank[index])
                    row["best_source_seed_uid"] = str(seed_uid[index])
                    row["distance_from_source_nm"] = distance

    pool = pd.DataFrame(list(legal_rows.values()))
    if pool.empty:
        raise RuntimeError(f"{task_id}: integer candidate pool is empty.")
    pool = pool.sort_values(
        ["best_source_continuous_rank", "distance_from_source_nm", "sequence_id", "r_core", "t1", "t2", "t3"],
        kind="mergesort",
    ).reset_index(drop=True)
    pool["candidate_index"] = np.arange(len(pool), dtype=np.int64)
    rejected = pd.DataFrame(rejected_rows)
    return pool, rejected


def physical_validity(
    qsca: torch.Tensor,
    qabs: torch.Tensor,
    qext: torch.Tensor,
    *,
    negative_tolerance: float,
    energy_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = (
        torch.isfinite(qsca).all(dim=1)
        & torch.isfinite(qabs).all(dim=1)
        & torch.isfinite(qext).all(dim=1)
    )
    minimum = torch.minimum(torch.minimum(qsca.amin(dim=1), qabs.amin(dim=1)), qext.amin(dim=1))
    nonnegative = minimum >= -float(negative_tolerance)
    residual = (qext - qsca - qabs).abs().amax(dim=1) / torch.maximum(
        torch.ones(qext.shape[0], dtype=qext.dtype, device=qext.device),
        qext.abs().amax(dim=1),
    )
    energy_ok = residual <= float(energy_tolerance)
    return finite & nonnegative & energy_ok, residual, minimum


def recursive_forward(
    geometry: torch.Tensor,
    materials: torch.Tensor,
    physics: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Evaluate a batch and recursively isolate hard solver failures."""
    count = geometry.shape[0]
    try:
        result = pymiediff_forward(
            geometry,
            materials,
            physics,
            backend="pena",
            precision="double",
        )
        return result["q_sca"], result["q_abs"], result["q_ext"], [""] * count
    except Exception as exc:  # solver isolation path
        if count == 1:
            width = len(physics["wavelengths"])
            nan = torch.full((1, width), torch.nan, dtype=torch.float64, device=geometry.device)
            return nan.clone(), nan.clone(), nan.clone(), [f"{type(exc).__name__}: {exc}"]
        midpoint = count // 2
        left = recursive_forward(geometry[:midpoint], materials[:midpoint], physics)
        right = recursive_forward(geometry[midpoint:], materials[midpoint:], physics)
        return (
            torch.cat((left[0], right[0]), dim=0),
            torch.cat((left[1], right[1]), dim=0),
            torch.cat((left[2], right[2]), dim=0),
            left[3] + right[3],
        )


def evaluate_integer_pool(
    *,
    pool: pd.DataFrame,
    target_qsca: np.ndarray,
    wavelengths_nm: np.ndarray,
    physics: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    n = len(pool)
    width = len(wavelengths_nm)
    qsca_all = np.full((n, width), np.nan, dtype=np.float64)
    qabs_all = np.full((n, width), np.nan, dtype=np.float64)
    qext_all = np.full((n, width), np.nan, dtype=np.float64)
    metrics_rows: list[Dict[str, Any]] = []
    rejected_rows: list[Dict[str, Any]] = []

    target = torch.as_tensor(target_qsca, dtype=torch.float64, device=device)
    wavelengths = torch.as_tensor(wavelengths_nm, dtype=torch.float64, device=device)

    for start in range(0, n, int(args.batch_size)):
        stop = min(start + int(args.batch_size), n)
        batch = pool.iloc[start:stop]
        geometry_np = batch[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.float64)
        materials_np = batch[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64)
        geometry = torch.as_tensor(geometry_np, dtype=torch.float64, device=device)
        materials = torch.as_tensor(materials_np, dtype=torch.long, device=device)
        qsca, qabs, qext, solver_errors = recursive_forward(geometry, materials, physics)
        valid, residual, minimum = physical_validity(
            qsca,
            qabs,
            qext,
            negative_tolerance=float(args.negative_efficiency_tolerance),
            energy_tolerance=float(args.energy_residual_tolerance),
        )
        clean_qsca = qsca.clamp_min(0.0)
        target_batch = target[None, :].expand(stop - start, -1)
        safe_qsca = torch.where(valid[:, None], clean_qsca, target_batch)
        metrics = per_sample_metrics(safe_qsca, target_batch, wavelengths)

        qsca_np = clean_qsca.detach().cpu().numpy()
        qabs_np = qabs.clamp_min(0.0).detach().cpu().numpy()
        qext_np = (qsca.clamp_min(0.0) + qabs.clamp_min(0.0)).detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy().astype(bool)
        residual_np = residual.detach().cpu().numpy()
        minimum_np = minimum.detach().cpu().numpy()
        metric_np = {key: value.detach().cpu().numpy() for key, value in metrics.items()}

        for local, global_index in enumerate(range(start, stop)):
            base = pool.iloc[global_index].to_dict()
            error = solver_errors[local]
            if not valid_np[local] or error:
                if error:
                    reason = "solver_exception"
                elif not np.isfinite(qsca_np[local]).all():
                    reason = "non_finite_efficiency"
                elif minimum_np[local] < -float(args.negative_efficiency_tolerance):
                    reason = "materially_negative_efficiency"
                elif residual_np[local] > float(args.energy_residual_tolerance):
                    reason = "energy_closure_failure"
                else:
                    reason = "unknown_physical_failure"
                rejected_rows.append(
                    {
                        **base,
                        "rejection_reason": reason,
                        "solver_error": error,
                        "minimum_efficiency": float(minimum_np[local]),
                        "energy_residual": float(residual_np[local]),
                    }
                )
                continue

            qsca_all[global_index] = qsca_np[local]
            qabs_all[global_index] = qabs_np[local]
            qext_all[global_index] = qext_np[local]
            metrics_rows.append(
                {
                    **base,
                    "mie_nmae": float(metric_np["nmae"][local]),
                    "mie_nrmse": float(metric_np["nrmse"][local]),
                    "mie_r2": float(metric_np["r2"][local]),
                    "mie_peak_error_nm": float(metric_np["peak_error_nm"][local]),
                    "energy_residual": float(residual_np[local]),
                    "minimum_efficiency": float(minimum_np[local]),
                }
            )
        print(f"  PyMieDiff integer evaluation: {stop}/{n}", flush=True)

    metrics_frame = pd.DataFrame(metrics_rows)
    rejected = pd.DataFrame(rejected_rows)
    if metrics_frame.empty:
        raise RuntimeError("No physically valid integer candidates survived PyMieDiff validation.")
    valid_indices = metrics_frame["candidate_index"].to_numpy(dtype=np.int64)
    return (
        metrics_frame,
        qsca_all[valid_indices],
        qabs_all[valid_indices],
        qext_all[valid_indices],
        rejected,
    )


def local_peak_indices(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 3:
        return np.asarray([int(np.argmax(y))], dtype=np.int64)
    interior = np.flatnonzero(
        (y[1:-1] >= y[:-2])
        & (y[1:-1] >= y[2:])
        & ((y[1:-1] > y[:-2]) | (y[1:-1] > y[2:]))
    ) + 1
    if len(interior) == 0:
        return np.asarray([int(np.argmax(y))], dtype=np.int64)
    return interior.astype(np.int64)


def dominant_peak_is_unique(target: np.ndarray, ratio: float) -> bool:
    peaks = local_peak_indices(target)
    heights = np.asarray(target, dtype=np.float64)[peaks]
    order = np.argsort(heights)[::-1]
    if len(order) == 1:
        return True
    largest = float(heights[order[0]])
    second = float(heights[order[1]])
    if second <= 0.0:
        return True
    return largest / second >= float(ratio)


def completed_checkpoint_path(output_dir: Path, task_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(task_id))
    return Path(output_dir) / "task_checkpoints" / f"{safe}_integer_completed.npz"


def save_task_checkpoint(
    path: Path,
    *,
    input_hash: str,
    config_hash: str,
    frame: pd.DataFrame,
    qsca: np.ndarray,
    qabs: np.ndarray,
    qext: np.ndarray,
    rejected_pool: pd.DataFrame,
    rejected_physics: pd.DataFrame,
) -> None:
    save_npz_atomic(
        path,
        input_hash=np.asarray(input_hash),
        config_hash=np.asarray(config_hash),
        frame_json=np.asarray(frame.to_json(orient="records")),
        qsca=np.asarray(qsca, dtype=np.float64),
        qabs=np.asarray(qabs, dtype=np.float64),
        qext=np.asarray(qext, dtype=np.float64),
        rejected_pool_json=np.asarray(rejected_pool.to_json(orient="records")),
        rejected_physics_json=np.asarray(rejected_physics.to_json(orient="records")),
    )


def load_task_checkpoint(path: Path, input_hash: str, config_hash: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["input_hash"].item()) != input_hash:
            raise RuntimeError(f"Task checkpoint input hash mismatch: {path}")
        if str(archive["config_hash"].item()) != config_hash:
            raise RuntimeError(f"Task checkpoint configuration hash mismatch: {path}")
        frame = pd.DataFrame(json.loads(str(archive["frame_json"].item())))
        rejected_pool = pd.DataFrame(json.loads(str(archive["rejected_pool_json"].item())))
        rejected_physics = pd.DataFrame(json.loads(str(archive["rejected_physics_json"].item())))
        return {
            "frame": frame,
            "qsca": np.asarray(archive["qsca"], dtype=np.float64),
            "qabs": np.asarray(archive["qabs"], dtype=np.float64),
            "qext": np.asarray(archive["qext"], dtype=np.float64),
            "rejected_pool": rejected_pool,
            "rejected_physics": rejected_physics,
        }


def evaluate_task(
    *,
    task_index: int,
    task_id: str,
    target_hash: str,
    target_qsca: np.ndarray,
    source: Mapping[str, np.ndarray],
    physics: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    config_hash: str,
) -> Dict[str, Any]:
    rows = np.flatnonzero(source["candidate_task_index"] == task_index)
    materials = source["candidate_material_indices"][rows]
    geometry = source["candidate_continuous_geometry"][rows]
    continuous_rank = source["candidate_continuous_rank"][rows]
    sequence_id = source["candidate_sequence_id"][rows]
    seed_uid = source["seed_candidate_uid"][rows]
    input_hash = task_input_hash(target_hash, materials, geometry, continuous_rank)
    checkpoint = completed_checkpoint_path(args.output_dir, task_id)
    if args.resume and checkpoint.is_file():
        print(f"{task_id}: loading completed integer checkpoint", flush=True)
        return load_task_checkpoint(checkpoint, input_hash, config_hash)

    pool, rejected_pool = generate_integer_pool(
        task_id=task_id,
        materials=materials,
        continuous_geometry=geometry,
        continuous_rank=continuous_rank,
        sequence_id=sequence_id,
        seed_uid=seed_uid,
        radius=int(args.integer_radius),
        wavelengths_nm=physics["wavelengths"],
        singularity_tolerance=float(args.solver_singularity_tolerance),
    )
    print(
        f"{task_id}: unique legal integer pool={len(pool)}, "
        f"singularity-rejected={len(rejected_pool)}",
        flush=True,
    )
    frame, qsca, qabs, qext, rejected_physics = evaluate_integer_pool(
        pool=pool,
        target_qsca=target_qsca,
        wavelengths_nm=physics["wavelengths"],
        physics=physics,
        device=device,
        args=args,
    )
    frame = frame.sort_values(
        ["mie_nmae", "mie_nrmse", "mie_peak_error_nm", "best_source_continuous_rank", "sequence_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    # Reorder spectra to match the sorted frame.
    old_candidate_indices = frame["candidate_index"].to_numpy(dtype=np.int64)
    unsorted_valid = np.sort(old_candidate_indices)
    position = {int(value): index for index, value in enumerate(unsorted_valid.tolist())}
    order = np.asarray([position[int(value)] for value in old_candidate_indices], dtype=np.int64)
    qsca, qabs, qext = qsca[order], qabs[order], qext[order]
    frame["integer_rank"] = np.arange(1, len(frame) + 1, dtype=np.int64)

    save_task_checkpoint(
        checkpoint,
        input_hash=input_hash,
        config_hash=config_hash,
        frame=frame,
        qsca=qsca,
        qabs=qabs,
        qext=qext,
        rejected_pool=rejected_pool,
        rejected_physics=rejected_physics,
    )
    print(
        f"{task_id}: integer validation completed; best Mie NMAE={frame.iloc[0]['mie_nmae']:.6f}",
        flush=True,
    )
    return {
        "frame": frame,
        "qsca": qsca,
        "qabs": qabs,
        "qext": qext,
        "rejected_pool": rejected_pool,
        "rejected_physics": rejected_physics,
    }


def add_success_flags(frame: pd.DataFrame, target_qsca: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    result = frame.copy()
    unique_peak = dominant_peak_is_unique(target_qsca, float(args.dominant_peak_ratio))
    result["target_unique_dominant_peak"] = bool(unique_peak)
    peak_ok = (~result["target_unique_dominant_peak"]) | (
        result["mie_peak_error_nm"] <= float(args.peak_error_tolerance_nm)
    )
    result["standard_success"] = (
        (result["mie_nmae"] <= float(args.success_nmae))
        & (result["mie_nrmse"] <= float(args.success_nrmse))
        & (result["mie_r2"] >= float(args.success_r2))
        & peak_ok
    )
    result["high_accuracy"] = (
        (result["mie_nmae"] <= float(args.high_accuracy_nmae))
        & (result["mie_nrmse"] <= float(args.high_accuracy_nrmse))
    )
    return result


def save_selected_npz(
    path: Path,
    *,
    tasks: Mapping[str, np.ndarray],
    selected: pd.DataFrame,
    spectra_by_task: Mapping[int, Dict[str, Any]],
) -> None:
    qsca: list[np.ndarray] = []
    qabs: list[np.ndarray] = []
    qext: list[np.ndarray] = []
    for row in selected.itertuples(index=False):
        task = spectra_by_task[int(row.task_index)]
        local = int(row.integer_rank) - 1
        qsca.append(task["qsca"][local])
        qabs.append(task["qabs"][local])
        qext.append(task["qext"][local])
    save_npz_atomic(
        path,
        task_id=np.asarray(tasks["task_id"]),
        wavelengths_nm=np.asarray(tasks["wavelengths_nm"], dtype=np.float64),
        target_qsca=np.asarray(tasks["target_qsca"], dtype=np.float64),
        target_hash=np.asarray(tasks["target_hash"]),
        candidate_task_index=selected["task_index"].to_numpy(dtype=np.int64),
        candidate_integer_rank=selected["integer_rank"].to_numpy(dtype=np.int64),
        candidate_design_uid=selected["design_uid"].to_numpy(dtype="U64"),
        candidate_sequence_id=selected["sequence_id"].to_numpy(dtype=np.int64),
        candidate_material_indices=selected[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64),
        candidate_integer_geometry=selected[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.int64),
        candidate_mie_qsca=np.asarray(qsca, dtype=np.float64),
        candidate_mie_qabs=np.asarray(qabs, dtype=np.float64),
        candidate_mie_qext=np.asarray(qext, dtype=np.float64),
        candidate_mie_nmae=selected["mie_nmae"].to_numpy(dtype=np.float64),
        candidate_mie_nrmse=selected["mie_nrmse"].to_numpy(dtype=np.float64),
        candidate_mie_r2=selected["mie_r2"].to_numpy(dtype=np.float64),
        candidate_mie_peak_error_nm=selected["mie_peak_error_nm"].to_numpy(dtype=np.float64),
        candidate_standard_success=selected["standard_success"].to_numpy(dtype=bool),
        candidate_high_accuracy=selected["high_accuracy"].to_numpy(dtype=bool),
    )


def plot_results(
    output_dir: Path,
    source: Mapping[str, np.ndarray],
    final: pd.DataFrame,
    results_by_task: Mapping[int, Dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = Path(output_dir) / "figures"
    wavelengths = source["wavelengths_nm"]
    for task_index, task_id in enumerate(source["task_id"].astype(str)):
        task_rows = final.loc[final["task_index"] == task_index].head(3)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(wavelengths, source["target_qsca"][task_index], linewidth=2.2, label="Target Qsca")
        task = results_by_task[task_index]
        for row in task_rows.itertuples(index=False):
            spectrum = task["qsca"][int(row.integer_rank) - 1]
            ax.plot(
                wavelengths,
                spectrum,
                linewidth=1.5,
                label=f"Rank {int(row.integer_rank)}: {row.materials}",
            )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Qsca")
        ax.set_title(f"{task_id}: integer-nanometer PyMieDiff validation")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task_id}_integer_validation.png", dpi=220)
        fig.savefig(figure_dir / f"{task_id}_integer_validation.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    prepare_output_dir(args.output_dir, resume=args.resume, overwrite=args.overwrite)
    started = time.time()

    physics = load_physics_assets(args.dataset)
    stage04 = verify_stage04(args.stage04_dir, args.dataset)
    source = load_stage04_candidates(
        stage04["selected_path"], physics["wavelengths"], physics["material_names"]
    )
    protocol = {
        **asdict(IntegerProtocol()),
        "integer_radius_nm": int(args.integer_radius),
        "batch_size": int(args.batch_size),
        "final_top_k": int(args.final_top_k),
        "solver_singularity_tolerance": float(args.solver_singularity_tolerance),
        "negative_efficiency_tolerance": float(args.negative_efficiency_tolerance),
        "energy_residual_tolerance": float(args.energy_residual_tolerance),
        "success_nmae": float(args.success_nmae),
        "success_nrmse": float(args.success_nrmse),
        "success_r2": float(args.success_r2),
        "high_accuracy_nmae": float(args.high_accuracy_nmae),
        "high_accuracy_nrmse": float(args.high_accuracy_nrmse),
        "peak_error_tolerance_nm": float(args.peak_error_tolerance_nm),
        "dominant_peak_ratio": float(args.dominant_peak_ratio),
        "seed": int(args.seed),
    }
    config_hash = stable_hash(protocol)

    print("Stage 05 strict-blind integer-nanometer validation")
    print(f"Device: {device}; integer radius: ±{args.integer_radius} nm")
    print(f"Blind targets: {len(source['task_id'])}; Stage-04 seeds: {len(source['candidate_task_index'])}")
    print("PyMieDiff: Peña backend, double precision")

    results_by_task: Dict[int, Dict[str, Any]] = {}
    frames: list[pd.DataFrame] = []
    rejected_pool_frames: list[pd.DataFrame] = []
    rejected_physics_frames: list[pd.DataFrame] = []

    for task_index, task_id in enumerate(source["task_id"].astype(str)):
        result = evaluate_task(
            task_index=task_index,
            task_id=task_id,
            target_hash=str(source["target_hash"][task_index]),
            target_qsca=source["target_qsca"][task_index],
            source=source,
            physics=physics,
            device=device,
            args=args,
            config_hash=config_hash,
        )
        frame = result["frame"].copy()
        frame.insert(0, "task_index", int(task_index))
        frame = add_success_flags(frame, source["target_qsca"][task_index], args)
        frame["materials"] = [
            "|".join(physics["material_names"][np.asarray(row, dtype=np.int64)].tolist())
            for row in frame[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64)
        ]
        frames.append(frame)
        if not result["rejected_pool"].empty:
            rejected = result["rejected_pool"].copy()
            rejected.insert(0, "task_index", int(task_index))
            rejected_pool_frames.append(rejected)
        if not result["rejected_physics"].empty:
            rejected = result["rejected_physics"].copy()
            rejected.insert(0, "task_index", int(task_index))
            rejected_physics_frames.append(rejected)
        results_by_task[task_index] = result

    all_frame = pd.concat(frames, ignore_index=True)
    all_frame = all_frame.sort_values(["task_index", "integer_rank"], kind="mergesort").reset_index(drop=True)
    selected = all_frame.loc[all_frame["integer_rank"] <= int(args.final_top_k)].copy().reset_index(drop=True)
    frozen = all_frame.loc[all_frame["integer_rank"] == 1].copy().reset_index(drop=True)

    if len(frozen) != 10:
        raise RuntimeError(f"Expected one frozen result per target, received {len(frozen)}.")
    if not np.all(geometry_bounds_mask(frozen[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.float64))):
        raise RuntimeError("A frozen integer geometry is illegal.")
    if not np.allclose(
        frozen[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.float64),
        np.rint(frozen[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.float64)),
    ):
        raise RuntimeError("A frozen geometry is not on the 1-nm integer grid.")

    all_csv = args.output_dir / "all_valid_integer_candidates.csv"
    selected_csv = args.output_dir / "selected_integer_designs.csv"
    frozen_csv = args.output_dir / "frozen_blind_design_results.csv"
    summary_csv = args.output_dir / "integer_design_target_summary.csv"
    rejected_singularity_csv = args.output_dir / "rejected_numerical_singularities.csv"
    rejected_physics_csv = args.output_dir / "rejected_physical_candidates.csv"
    selected_npz = args.output_dir / "selected_integer_designs.npz"
    frozen_npz = args.output_dir / "frozen_blind_design_results.npz"

    atomic_write_csv(all_csv, all_frame)
    atomic_write_csv(selected_csv, selected)
    atomic_write_csv(frozen_csv, frozen)
    rejected_pool = pd.concat(rejected_pool_frames, ignore_index=True) if rejected_pool_frames else pd.DataFrame()
    rejected_physics = pd.concat(rejected_physics_frames, ignore_index=True) if rejected_physics_frames else pd.DataFrame()
    atomic_write_csv(rejected_singularity_csv, rejected_pool)
    atomic_write_csv(rejected_physics_csv, rejected_physics)

    target_summary = frozen[[
        "task_index", "task_id", "materials", "r_core", "t1", "t2", "t3", "outer_radius",
        "mie_nmae", "mie_nrmse", "mie_r2", "mie_peak_error_nm",
        "target_unique_dominant_peak", "standard_success", "high_accuracy",
        "best_source_continuous_rank", "distance_from_source_nm",
    ]].copy()
    target_summary["valid_integer_candidate_count"] = [
        int((all_frame["task_index"] == task).sum()) for task in target_summary["task_index"]
    ]
    atomic_write_csv(summary_csv, target_summary)

    save_selected_npz(selected_npz, tasks=source, selected=selected, spectra_by_task=results_by_task)
    save_selected_npz(frozen_npz, tasks=source, selected=frozen, spectra_by_task=results_by_task)

    if not args.no_plots:
        plot_results(args.output_dir, source, selected, results_by_task)

    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "protocol": protocol,
        "blind_policy": {
            "public_target_qsca_read": True,
            "reference_geometry_read": False,
            "reference_materials_read": False,
            "target_qabs_read": False,
            "target_qext_read": False,
            "pgff_loaded": False,
            "inverse_network_loaded": False,
            "forward_spectra_read": False,
            "train_validation_test_split_read": False,
            "materials_fixed_during_integer_search": True,
            "final_ranking_uses_pymiediff_only": True,
        },
        "inputs": {
            "forward_physics_asset": artifact(args.dataset),
            "stage04_summary": artifact(stage04["summary_path"]),
            "stage04_selected_candidates": artifact(stage04["selected_path"]),
        },
        "counts": {
            "targets": int(len(source["task_id"])),
            "stage04_continuous_seeds": int(len(source["candidate_task_index"])),
            "valid_unique_integer_candidates": int(len(all_frame)),
            "rejected_solver_singularities": int(len(rejected_pool)),
            "rejected_physical_evaluations": int(len(rejected_physics)),
            "selected_final_candidates": int(len(selected)),
            "frozen_results": int(len(frozen)),
            "standard_success_targets": int(frozen["standard_success"].sum()),
            "high_accuracy_targets": int(frozen["high_accuracy"].sum()),
        },
        "aggregate_metrics": {
            "median_frozen_nmae": float(frozen["mie_nmae"].median()),
            "mean_frozen_nmae": float(frozen["mie_nmae"].mean()),
            "maximum_frozen_nmae": float(frozen["mie_nmae"].max()),
            "median_frozen_nrmse": float(frozen["mie_nrmse"].median()),
            "minimum_frozen_r2": float(frozen["mie_r2"].min()),
            "maximum_energy_residual": float(all_frame["energy_residual"].max()),
            "minimum_solver_distance": float(all_frame["minimum_abs_sin"].min()),
        },
        "outputs": {
            "all_valid_integer_candidates": artifact(all_csv),
            "selected_final_integer_candidates_csv": artifact(selected_csv),
            "frozen_blind_integer_results_csv": artifact(frozen_csv),
            "integer_target_summary": artifact(summary_csv),
            "rejected_integer_singularities": artifact(rejected_singularity_csv),
            "rejected_integer_physics": artifact(rejected_physics_csv),
            "selected_final_integer_candidates_npz": artifact(selected_npz),
            "frozen_blind_integer_results_npz": artifact(frozen_npz),
        },
    }
    summary_path = args.output_dir / "integer_search_run_summary.json"
    save_json_atomic(summary_path, summary)

    print("Stage 05 strict-blind integer-nanometer validation PASSED")
    print(f"Valid unique integer candidates: {len(all_frame)}")
    print(f"Frozen blind results: {len(frozen)}")
    print(f"Standard success: {int(frozen['standard_success'].sum())}/10")
    print(f"High accuracy: {int(frozen['high_accuracy'].sum())}/10")
    print(f"Median frozen Mie NMAE: {float(frozen['mie_nmae'].median()):.6f}")
    print("No hidden reference, target Qabs/Qext, PGFF, inverse network, or split spectra were used.")


if __name__ == "__main__":
    main()
