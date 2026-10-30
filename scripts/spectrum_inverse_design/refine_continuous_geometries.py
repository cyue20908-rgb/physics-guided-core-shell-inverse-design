# -*- coding: utf-8 -*-
"""Stage 04: strict-blind differentiable PyMieDiff geometry refinement.

For every Stage-03 material seed, this stage keeps the four material indices
fixed and optimizes only the continuous geometry ``[r_core, t1, t2, t3]``.
The objective consumes public target Qsca only.  PyMieDiff supplies the exact
Mie forward calculation and the geometry gradient; PGFF, the inverse network,
hidden reference structures, target Qabs/Qext, and the forward test split are
not loaded.

Scientific safeguards
---------------------
* exact Stage-03 input/provenance validation;
* deterministic multiple restarts per material sequence;
* hard differentiable geometry parameterization;
* double-precision PyMieDiff (Peña backend by default);
* explicit Peña-recursion singularity guard and differentiable barrier;
* non-negativity and energy-closure checks on every accepted best candidate;
* resumable per-target checkpoints;
* final ranking by high-fidelity target-peak-normalized Qsca NMAE.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import hashlib
import json
import math
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
    GEOMETRY_CHECK_ATOL_NM,
    LOWER_BOUNDS,
    MATERIAL_COLUMNS,
    MAX_OUTER_RADIUS,
    N_LAYERS,
    UPPER_BOUNDS,
    geometry_bounds_mask,
    geometry_from_raw,
    material_sequence_mask,
    raw_from_geometry,
)
from core_shell_optical_design.inverse_design.inverse_design_objectives import per_sample_metrics

STAGE = "continuous_geometry_refinement"
VERSION = "tandem-mie-inverse-2026-07-26-stage04-v1"

DEFAULT_STAGE03_DIR = DEFAULT_RESULT_ROOT / "material_sequence_screening"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "continuous_geometry_refinement"


@dataclass(frozen=True)
class RefinementProtocol:
    restarts_per_seed: int = 4
    steps: int = 250
    learning_rate: float = 3.0e-2
    minimum_learning_rate: float = 1.0e-3
    smooth_l1_beta: float = 5.0e-3
    gradient_clip: float = 5.0
    raw_restart_std: float = 0.30
    history_every: int = 5
    print_every: int = 25
    stage05_top_k: int = 10
    pymiediff_backend: str = "pena"
    pymiediff_precision: str = "double"
    solver_singularity_tolerance: float = 1.0e-6
    optimization_singularity_margin: float = 1.0e-3
    singularity_nudge_threshold: float = 2.0e-4
    singularity_barrier_weight: float = 5.0e-2
    negative_efficiency_tolerance: float = 1.0e-7
    energy_residual_tolerance: float = 1.0e-8
    seed: int = 20260726


def parse_args() -> argparse.Namespace:
    protocol = RefinementProtocol()
    parser = argparse.ArgumentParser(
        description="Strict-blind PyMieDiff continuous geometry refinement"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--stage03-dir", type=Path, default=DEFAULT_STAGE03_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--restarts", type=int, default=protocol.restarts_per_seed)
    parser.add_argument("--steps", type=int, default=protocol.steps)
    parser.add_argument("--learning-rate", type=float, default=protocol.learning_rate)
    parser.add_argument(
        "--minimum-learning-rate", type=float, default=protocol.minimum_learning_rate
    )
    parser.add_argument("--smooth-l1-beta", type=float, default=protocol.smooth_l1_beta)
    parser.add_argument("--gradient-clip", type=float, default=protocol.gradient_clip)
    parser.add_argument("--raw-restart-std", type=float, default=protocol.raw_restart_std)
    parser.add_argument("--history-every", type=int, default=protocol.history_every)
    parser.add_argument("--print-every", type=int, default=protocol.print_every)
    parser.add_argument("--stage05-top-k", type=int, default=protocol.stage05_top_k)
    parser.add_argument(
        "--solver-singularity-tolerance",
        type=float,
        default=protocol.solver_singularity_tolerance,
    )
    parser.add_argument(
        "--optimization-singularity-margin",
        type=float,
        default=protocol.optimization_singularity_margin,
    )
    parser.add_argument(
        "--singularity-nudge-threshold",
        type=float,
        default=protocol.singularity_nudge_threshold,
    )
    parser.add_argument(
        "--singularity-barrier-weight",
        type=float,
        default=protocol.singularity_barrier_weight,
    )
    parser.add_argument(
        "--negative-efficiency-tolerance",
        type=float,
        default=protocol.negative_efficiency_tolerance,
    )
    parser.add_argument(
        "--energy-residual-tolerance",
        type=float,
        default=protocol.energy_residual_tolerance,
    )
    parser.add_argument("--seed", type=int, default=protocol.seed)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.restarts < 1:
        raise ValueError("--restarts must be at least 1.")
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    if args.learning_rate <= 0.0 or args.minimum_learning_rate <= 0.0:
        raise ValueError("Learning rates must be positive.")
    if args.minimum_learning_rate > args.learning_rate:
        raise ValueError("--minimum-learning-rate cannot exceed --learning-rate.")
    if args.smooth_l1_beta <= 0.0:
        raise ValueError("--smooth-l1-beta must be positive.")
    if args.gradient_clip <= 0.0:
        raise ValueError("--gradient-clip must be positive.")
    if args.raw_restart_std < 0.0:
        raise ValueError("--raw-restart-std cannot be negative.")
    if args.history_every < 1 or args.print_every < 1:
        raise ValueError("History/print intervals must be positive.")
    if not 1 <= args.stage05_top_k <= 20:
        raise ValueError("--stage05-top-k must be between 1 and 20.")
    for name in (
        "solver_singularity_tolerance",
        "optimization_singularity_margin",
        "singularity_nudge_threshold",
        "negative_efficiency_tolerance",
        "energy_residual_tolerance",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.solver_singularity_tolerance >= args.singularity_nudge_threshold:
        raise ValueError(
            "--singularity-nudge-threshold must exceed the final solver tolerance."
        )
    if args.singularity_nudge_threshold >= args.optimization_singularity_margin:
        raise ValueError(
            "--optimization-singularity-margin must exceed the nudge threshold."
        )
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


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def stable_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_output_dir(path: Path, *, resume: bool, overwrite: bool) -> None:
    path = Path(path)
    if overwrite and path.exists():
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        summary = path / "continuous_refinement_run_summary.json"
        if summary.is_file():
            existing = load_json(summary)
            if existing.get("status") == "passed":
                raise FileExistsError(
                    f"Stage 04 is already completed: {summary}\n"
                    "Use --overwrite only when intentionally replacing the frozen result."
                )
        if not resume:
            raise FileExistsError(
                f"Stage 04 output is non-empty: {path}\n"
                "Use --resume for a partial run, or --overwrite to restart deliberately."
            )
    path.mkdir(parents=True, exist_ok=True)
    (path / "task_checkpoints").mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)


def load_physics_assets(path: Path) -> Dict[str, Any]:
    """Read only wavelengths and material optical tables from the forward NPZ."""
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
    if wavelengths.ndim != 1 or len(wavelengths) == 0 or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("wavelengths must be one-dimensional and strictly increasing.")
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


def verify_stage03(stage03_dir: Path, dataset_path: Path) -> Dict[str, Any]:
    stage03_dir = Path(stage03_dir)
    if "private" in {part.strip().lower() for part in stage03_dir.parts}:
        raise RuntimeError("Stage 04 refuses a Stage-03 path containing 'private'.")
    summary_path = stage03_dir / "material_sequence_screening_run_summary.json"
    seeds_path = stage03_dir / "selected_refinement_seeds.npz"
    csv_path = stage03_dir / "selected_refinement_seeds.csv"
    summary = load_json(summary_path)
    if summary.get("status") != "passed":
        raise RuntimeError("Stage 03 summary is not a passed run.")
    policy = summary.get("blind_policy", {})
    forbidden_true = (
        "reference_geometry_read",
        "reference_materials_read",
        "target_qabs_read",
        "target_qext_read",
        "forward_test_split_evaluated",
        "legacy_inverse_candidate_results_read",
        "stage03_uses_pymiediff",
    )
    violated = [name for name in forbidden_true if bool(policy.get(name, True))]
    if violated:
        raise RuntimeError(f"Stage 03 blind policy is invalid: {violated}")
    outputs = summary.get("outputs", {})
    recorded = outputs.get("selected_stage04_seeds_npz", {})
    if recorded and str(recorded.get("sha256", "")) != sha256_file(seeds_path):
        raise RuntimeError("selected_refinement_seeds.npz hash differs from Stage 03 summary.")
    inputs = summary.get("inputs", {})
    recorded_dataset = inputs.get("dataset", {})
    if recorded_dataset and str(recorded_dataset.get("sha256", "")) != sha256_file(dataset_path):
        raise RuntimeError("Forward dataset hash differs from the dataset used by Stage 03.")
    if not seeds_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError("Stage 03 selected seed artifacts are incomplete.")
    return {
        "summary": summary,
        "summary_path": summary_path,
        "seeds_path": seeds_path,
        "csv_path": csv_path,
    }


def load_stage03_seeds(
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
        "candidate_screen_rank",
        "candidate_sequence_id",
        "candidate_uid",
        "candidate_material_indices",
        "candidate_geometry",
        "candidate_pgff_qsca",
        "candidate_pgff_nmae",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"Stage 03 seed NPZ is missing fields: {missing}")
        data = {key: np.asarray(archive[key]) for key in required}
    task_id = data["task_id"].astype(str)
    wavelengths = data["wavelengths_nm"].astype(np.float64)
    target_qsca = data["target_qsca"].astype(np.float64)
    target_hash = data["target_hash"].astype(str)
    task_index = data["candidate_task_index"].astype(np.int64)
    screen_rank = data["candidate_screen_rank"].astype(np.int64)
    sequence_id = data["candidate_sequence_id"].astype(np.int64)
    uid = data["candidate_uid"].astype(str)
    materials = data["candidate_material_indices"].astype(np.int64)
    geometry = data["candidate_geometry"].astype(np.float64)
    pgff_qsca = data["candidate_pgff_qsca"].astype(np.float64)
    pgff_nmae = data["candidate_pgff_nmae"].astype(np.float64)
    if not np.array_equal(wavelengths, np.asarray(wavelengths_expected, dtype=np.float64)):
        raise RuntimeError("Stage 03 and forward-physics wavelength grids differ.")
    if target_qsca.shape != (len(task_id), len(wavelengths)):
        raise ValueError("target_qsca shape is incompatible with task IDs/wavelengths.")
    n = len(task_index)
    expected_shapes = {
        "screen_rank": (n,),
        "sequence_id": (n,),
        "candidate_uid": (n,),
        "materials": (n, 4),
        "geometry": (n, 4),
        "pgff_qsca": (n, len(wavelengths)),
        "pgff_nmae": (n,),
    }
    actual = {
        "screen_rank": screen_rank.shape,
        "sequence_id": sequence_id.shape,
        "candidate_uid": uid.shape,
        "materials": materials.shape,
        "geometry": geometry.shape,
        "pgff_qsca": pgff_qsca.shape,
        "pgff_nmae": pgff_nmae.shape,
    }
    bad = {key: (actual[key], shape) for key, shape in expected_shapes.items() if actual[key] != shape}
    if bad:
        raise ValueError(f"Stage 03 seed array shape mismatch: {bad}")
    if len(task_id) != 10 or n != 200:
        raise RuntimeError(f"Expected 10 targets and 200 seeds, got {len(task_id)} and {n}.")
    if not np.isfinite(target_qsca).all() or float(target_qsca.min()) < -1.0e-10:
        raise ValueError("Public target Qsca contains invalid values.")
    if not np.isfinite(geometry).all() or not np.all(geometry_bounds_mask(geometry)):
        raise ValueError("Stage 03 seed geometry is not strictly legal.")
    if not np.all(material_sequence_mask(materials, material_names)):
        raise ValueError("Stage 03 seed material sequence is illegal.")
    if len(np.unique(uid)) != n:
        raise RuntimeError("Stage 03 candidate UIDs are not unique.")
    for i in range(len(task_id)):
        rows = np.flatnonzero(task_index == i)
        if len(rows) != 20:
            raise RuntimeError(f"Target {task_id[i]} has {len(rows)} seeds, expected 20.")
        if sorted(screen_rank[rows].tolist()) != list(range(1, 21)):
            raise RuntimeError(f"Target {task_id[i]} screen ranks are not exactly 1..20.")
        if len(np.unique(sequence_id[rows])) != 20:
            raise RuntimeError(f"Target {task_id[i]} contains repeated material sequences.")
    return {
        "task_id": task_id,
        "wavelengths_nm": wavelengths,
        "target_qsca": target_qsca,
        "target_hash": target_hash,
        "candidate_task_index": task_index,
        "candidate_screen_rank": screen_rank,
        "candidate_sequence_id": sequence_id,
        "candidate_uid": uid,
        "candidate_material_indices": materials,
        "candidate_geometry": geometry,
        "candidate_pgff_qsca": pgff_qsca,
        "candidate_pgff_nmae": pgff_nmae,
    }


def torch_singularity_diagnostics(
    geometry: torch.Tensor,
    wavelengths_nm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outer = geometry.sum(dim=1)
    phase = 2.0 * math.pi * outer[:, None] / wavelengths_nm[None, :]
    absolute_sine = torch.abs(torch.sin(phase))
    minimum, index = absolute_sine.min(dim=1)
    return minimum, index


def numpy_min_abs_sin(geometry: np.ndarray, wavelengths_nm: np.ndarray) -> np.ndarray:
    values = np.asarray(geometry, dtype=np.float64)
    outer = values.sum(axis=1)
    phase = 2.0 * np.pi * outer[:, None] / np.asarray(wavelengths_nm, dtype=np.float64)[None, :]
    return np.abs(np.sin(phase)).min(axis=1)


def generate_restart_raw(
    seed_geometry: np.ndarray,
    restarts: int,
    raw_std: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    seed_tensor = torch.as_tensor(seed_geometry, dtype=torch.float64, device=device)
    base_raw = raw_from_geometry(seed_tensor)
    n_seed = len(seed_geometry)
    raw_rows = []
    seed_local_index = []
    restart_index = []
    for seed_i in range(n_seed):
        for restart_i in range(restarts):
            raw = base_raw[seed_i].clone()
            if restart_i > 0 and raw_std > 0.0:
                noise = torch.as_tensor(
                    rng.normal(0.0, raw_std, size=4), dtype=torch.float64, device=device
                )
                raw = raw + noise
            raw_rows.append(raw)
            seed_local_index.append(seed_i)
            restart_index.append(restart_i)
    return (
        torch.stack(raw_rows, dim=0),
        np.asarray(seed_local_index, dtype=np.int64),
        np.asarray(restart_index, dtype=np.int64),
    )


def _candidate_geometry_shifts(
    geometry: np.ndarray,
    max_outer: float,
) -> Iterable[np.ndarray]:
    shifts = (0.01, -0.01, 0.025, -0.025, 0.05, -0.05, 0.10, -0.10, 0.25, -0.25, 0.50, -0.50)
    coordinate_order = (3, 2, 1, 0)
    for shift in shifts:
        for coordinate in coordinate_order:
            trial = geometry.copy()
            trial[coordinate] += shift
            if trial[coordinate] < LOWER_BOUNDS[coordinate] or trial[coordinate] > UPPER_BOUNDS[coordinate]:
                continue
            if trial.sum() > max_outer:
                continue
            if np.all(geometry_bounds_mask(trial[None, :])):
                yield trial


def nudge_raw_away_from_singularities(
    raw: torch.Tensor,
    wavelengths_nm: np.ndarray,
    threshold: float,
) -> int:
    """Deterministically move unsafe outer radii before calling PyMieDiff."""
    with torch.no_grad():
        geometry = geometry_from_raw(raw).detach().cpu().numpy().astype(np.float64)
        minimum = numpy_min_abs_sin(geometry, wavelengths_nm)
        unsafe = np.flatnonzero(minimum <= float(threshold))
        if len(unsafe) == 0:
            return 0
        max_outer = MAX_OUTER_RADIUS - 1.0e-3
        for index in unsafe:
            best_geometry = geometry[index].copy()
            best_value = float(minimum[index])
            for trial in _candidate_geometry_shifts(best_geometry, max_outer):
                value = float(numpy_min_abs_sin(trial[None, :], wavelengths_nm)[0])
                if value > best_value:
                    best_value = value
                    best_geometry = trial
                if value > max(float(threshold) * 5.0, 1.0e-3):
                    break
            if best_value <= float(threshold):
                raise RuntimeError(
                    "Could not nudge a candidate away from a PyMieDiff solver singularity."
                )
            replacement = raw_from_geometry(
                torch.as_tensor(best_geometry, dtype=raw.dtype, device=raw.device)
            )
            raw[index].copy_(replacement)
        return int(len(unsafe))


def compute_physical_validity(
    qsca: torch.Tensor,
    qabs: torch.Tensor,
    qext: torch.Tensor,
    min_abs_sin: torch.Tensor,
    *,
    negative_tolerance: float,
    energy_tolerance: float,
    singularity_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    finite = (
        torch.isfinite(qsca).all(dim=1)
        & torch.isfinite(qabs).all(dim=1)
        & torch.isfinite(qext).all(dim=1)
    )
    nonnegative = (
        (qsca.amin(dim=1) >= -float(negative_tolerance))
        & (qabs.amin(dim=1) >= -float(negative_tolerance))
        & (qext.amin(dim=1) >= -float(negative_tolerance))
    )
    residual = (qext - qsca - qabs).abs().amax(dim=1) / torch.maximum(
        torch.ones(qext.shape[0], dtype=qext.dtype, device=qext.device),
        qext.abs().amax(dim=1),
    )
    energy_ok = residual <= float(energy_tolerance)
    singularity_ok = min_abs_sin > float(singularity_tolerance)
    return finite & nonnegative & energy_ok & singularity_ok, residual


def per_sample_smooth_loss(
    prediction_qsca: torch.Tensor,
    target_qsca: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    scale = target_qsca.amax(dim=1, keepdim=True).clamp_min(1.0e-6)
    return F.smooth_l1_loss(
        prediction_qsca / scale,
        target_qsca / scale,
        beta=float(beta),
        reduction="none",
    ).mean(dim=1)


def clean_efficiency(values: torch.Tensor, tolerance: float) -> torch.Tensor:
    if bool((values < -float(tolerance)).any()):
        return values
    return values.clamp_min(0.0)


def evaluate_current(
    raw: torch.Tensor,
    materials: torch.Tensor,
    target_repeated: torch.Tensor,
    wavelengths: torch.Tensor,
    physics: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    geometry = geometry_from_raw(raw)
    result = pymiediff_forward(
        geometry,
        materials,
        physics,
        backend="pena",
        precision="double",
    )
    qsca = result["q_sca"]
    qabs = result["q_abs"]
    qext = result["q_ext"]
    min_abs_sin, critical_index = torch_singularity_diagnostics(geometry, wavelengths)
    valid, energy_residual = compute_physical_validity(
        qsca,
        qabs,
        qext,
        min_abs_sin,
        negative_tolerance=args.negative_efficiency_tolerance,
        energy_tolerance=args.energy_residual_tolerance,
        singularity_tolerance=args.solver_singularity_tolerance,
    )
    qsca_clean = clean_efficiency(qsca, args.negative_efficiency_tolerance)
    safe_qsca = torch.where(valid[:, None], qsca_clean, target_repeated.detach())
    metrics = per_sample_metrics(safe_qsca, target_repeated, wavelengths)
    for key in metrics:
        metrics[key] = torch.where(
            valid,
            metrics[key],
            torch.full_like(metrics[key], torch.inf),
        )
    return {
        "geometry": geometry,
        "q_sca": qsca,
        "q_abs": qabs,
        "q_ext": qext,
        "q_sca_clean": qsca_clean,
        "valid": valid,
        "energy_residual": energy_residual,
        "min_abs_sin": min_abs_sin,
        "critical_index": critical_index,
        **metrics,
    }


def update_best(
    state: Dict[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
    step: int,
) -> None:
    improved = current["valid"] & (current["nmae"] < state["nmae"])
    if not bool(improved.any()):
        return
    state["nmae"][improved] = current["nmae"][improved].detach()
    state["nrmse"][improved] = current["nrmse"][improved].detach()
    state["r2"][improved] = current["r2"][improved].detach()
    state["peak_error_nm"][improved] = current["peak_error_nm"][improved].detach()
    state["energy_residual"][improved] = current["energy_residual"][improved].detach()
    state["min_abs_sin"][improved] = current["min_abs_sin"][improved].detach()
    state["geometry"][improved] = current["geometry"][improved].detach()
    state["q_sca"][improved] = current["q_sca_clean"][improved].detach()
    state["q_abs"][improved] = current["q_abs"][improved].detach().clamp_min(0.0)
    state["q_ext"][improved] = (
        state["q_sca"][improved] + state["q_abs"][improved]
    )
    state["step"][improved] = int(step)


def initialize_best(batch: int, wavelengths: int, device: torch.device) -> Dict[str, torch.Tensor]:
    inf = torch.full((batch,), torch.inf, dtype=torch.float64, device=device)
    return {
        "nmae": inf.clone(),
        "nrmse": inf.clone(),
        "r2": torch.full((batch,), -torch.inf, dtype=torch.float64, device=device),
        "peak_error_nm": inf.clone(),
        "energy_residual": inf.clone(),
        "min_abs_sin": torch.zeros(batch, dtype=torch.float64, device=device),
        "geometry": torch.zeros((batch, 4), dtype=torch.float64, device=device),
        "q_sca": torch.zeros((batch, wavelengths), dtype=torch.float64, device=device),
        "q_abs": torch.zeros((batch, wavelengths), dtype=torch.float64, device=device),
        "q_ext": torch.zeros((batch, wavelengths), dtype=torch.float64, device=device),
        "step": torch.full((batch,), -1, dtype=torch.int64, device=device),
    }


def checkpoint_path(output_dir: Path, task_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(task_id))
    return Path(output_dir) / "task_checkpoints" / f"{safe}_completed.npz"


def active_checkpoint_path(output_dir: Path, task_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(task_id))
    return Path(output_dir) / "task_checkpoints" / f"{safe}_active.pt"


def save_active_checkpoint(
    path: Path,
    *,
    input_hash: str,
    config_hash: str,
    step: int,
    raw: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best: Mapping[str, torch.Tensor],
    initial: Mapping[str, torch.Tensor],
    history: Sequence[Mapping[str, Any]],
    seed_local_index: np.ndarray,
    restart_index: np.ndarray,
    nudge_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "stage": STAGE,
        "version": VERSION,
        "input_hash": input_hash,
        "config_hash": config_hash,
        "step": int(step),
        "raw": raw.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best": {key: value.detach().cpu() for key, value in best.items()},
        "initial": {key: value.detach().cpu() for key, value in initial.items()},
        "history": list(history),
        "seed_local_index": np.asarray(seed_local_index, dtype=np.int64),
        "restart_index": np.asarray(restart_index, dtype=np.int64),
        "nudge_count": int(nudge_count),
    }
    torch.save(payload, temp)
    os.replace(temp, path)


def load_active_checkpoint(
    path: Path,
    device: torch.device,
    input_hash: str,
    config_hash: str,
) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("input_hash") != input_hash or payload.get("config_hash") != config_hash:
        raise RuntimeError(f"Active checkpoint does not match current inputs/config: {path}")
    return payload


def task_input_hash(
    target_hash: str,
    seed_uid: np.ndarray,
    materials: np.ndarray,
    geometry: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(target_hash).encode("utf-8"))
    for uid in seed_uid.astype(str):
        digest.update(uid.encode("utf-8"))
    digest.update(np.asarray(materials, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(geometry, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def optimize_target(
    *,
    task_index: int,
    task_id: str,
    target_hash: str,
    target_qsca: np.ndarray,
    seed_uid: np.ndarray,
    seed_screen_rank: np.ndarray,
    seed_sequence_id: np.ndarray,
    seed_materials: np.ndarray,
    seed_geometry: np.ndarray,
    seed_pgff_nmae: np.ndarray,
    physics: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    config_hash: str,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(args.seed) + int(task_index) * 100003)
    raw_init, seed_local_index, restart_index = generate_restart_raw(
        seed_geometry,
        int(args.restarts),
        float(args.raw_restart_std),
        rng,
        device,
    )
    materials_np = seed_materials[seed_local_index]
    materials = torch.as_tensor(materials_np, dtype=torch.long, device=device)
    wavelengths_np = np.asarray(physics["wavelengths"], dtype=np.float64)
    wavelengths = torch.as_tensor(wavelengths_np, dtype=torch.float64, device=device)
    target = torch.as_tensor(target_qsca, dtype=torch.float64, device=device)
    target_repeated = target[None, :].expand(len(raw_init), -1)
    input_hash = task_input_hash(target_hash, seed_uid, seed_materials, seed_geometry)

    raw = torch.nn.Parameter(raw_init.clone())
    optimizer = torch.optim.Adam([raw], lr=float(args.learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.steps)),
        eta_min=float(args.minimum_learning_rate),
    )
    best = initialize_best(len(raw), len(wavelengths_np), device)
    initial: Dict[str, torch.Tensor]
    history: list[Dict[str, Any]] = []
    nudge_count = 0
    start_step = 0
    active_path = active_checkpoint_path(args.output_dir, task_id)
    if args.resume and active_path.is_file():
        payload = load_active_checkpoint(active_path, device, input_hash, config_hash)
        raw.data.copy_(payload["raw"].to(device=device, dtype=torch.float64))
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        best = {key: value.to(device) for key, value in payload["best"].items()}
        initial = {key: value.to(device) for key, value in payload["initial"].items()}
        history = list(payload["history"])
        seed_local_index = np.asarray(payload["seed_local_index"], dtype=np.int64)
        restart_index = np.asarray(payload["restart_index"], dtype=np.int64)
        nudge_count = int(payload.get("nudge_count", 0))
        start_step = int(payload["step"]) + 1
        print(f"{task_id}: resuming active checkpoint at step {start_step}/{args.steps}")
    else:
        nudge_count += nudge_raw_away_from_singularities(
            raw, wavelengths_np, float(args.singularity_nudge_threshold)
        )
        with torch.no_grad():
            current = evaluate_current(
                raw, materials, target_repeated, wavelengths, physics, args
            )
        update_best(best, current, step=0)
        initial = {
            "nmae": current["nmae"].detach().clone(),
            "nrmse": current["nrmse"].detach().clone(),
            "r2": current["r2"].detach().clone(),
            "peak_error_nm": current["peak_error_nm"].detach().clone(),
            "energy_residual": current["energy_residual"].detach().clone(),
            "min_abs_sin": current["min_abs_sin"].detach().clone(),
            "geometry": current["geometry"].detach().clone(),
            "q_sca": current["q_sca_clean"].detach().clone(),
        }
        if not bool(current["valid"].any()):
            raise RuntimeError(f"{task_id}: no physically valid initial PyMieDiff restart.")
        history.append(
            {
                "task_index": int(task_index),
                "task_id": str(task_id),
                "step": 0,
                "learning_rate": float(args.learning_rate),
                "best_nmae": float(current["nmae"][current["valid"]].min().item()),
                "median_nmae": float(current["nmae"][current["valid"]].median().item()),
                "valid_count": int(current["valid"].sum().item()),
                "restart_count": int(len(raw)),
                "minimum_abs_sin": float(current["min_abs_sin"].min().item()),
            }
        )

    for step in range(start_step, int(args.steps)):
        nudge_count += nudge_raw_away_from_singularities(
            raw, wavelengths_np, float(args.singularity_nudge_threshold)
        )
        optimizer.zero_grad(set_to_none=True)
        current = evaluate_current(raw, materials, target_repeated, wavelengths, physics, args)
        update_best(best, current, step=step)

        safe_prediction = torch.where(
            current["valid"][:, None],
            current["q_sca_clean"],
            target_repeated.detach(),
        )
        spectral_loss = per_sample_smooth_loss(
            safe_prediction, target_repeated, float(args.smooth_l1_beta)
        )
        normalized_barrier = torch.relu(
            float(args.optimization_singularity_margin) - current["min_abs_sin"]
        ) / float(args.optimization_singularity_margin)
        barrier = normalized_barrier.square()
        invalid_penalty = (~current["valid"]).to(torch.float64)
        per_candidate_loss = (
            spectral_loss
            + float(args.singularity_barrier_weight) * barrier
            + invalid_penalty
        )
        loss = per_candidate_loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{task_id}: non-finite optimization loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([raw], float(args.gradient_clip))
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise RuntimeError(f"{task_id}: non-finite geometry gradient at step {step}.")
        optimizer.step()
        scheduler.step()

        record = (step + 1) % int(args.history_every) == 0 or step + 1 == int(args.steps)
        announce = (step + 1) % int(args.print_every) == 0 or step == start_step or step + 1 == int(args.steps)
        if record or announce:
            valid = current["valid"]
            best_now = float(best["nmae"].min().item())
            median_now = float(current["nmae"][valid].median().item()) if bool(valid.any()) else float("inf")
            row = {
                "task_index": int(task_index),
                "task_id": str(task_id),
                "step": int(step + 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "loss": float(loss.detach().item()),
                "best_nmae": best_now,
                "median_nmae": median_now,
                "valid_count": int(valid.sum().item()),
                "restart_count": int(len(raw)),
                "minimum_abs_sin": float(current["min_abs_sin"].min().item()),
                "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
            }
            if record:
                history.append(row)
            if announce:
                print(
                    f"{task_id} step {step + 1:4d}/{args.steps}: "
                    f"best={best_now:.6f}, median={median_now:.6f}, "
                    f"valid={int(valid.sum().item())}/{len(raw)}, "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
            if record:
                save_active_checkpoint(
                    active_path,
                    input_hash=input_hash,
                    config_hash=config_hash,
                    step=step,
                    raw=raw,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best=best,
                    initial=initial,
                    history=history,
                    seed_local_index=seed_local_index,
                    restart_index=restart_index,
                    nudge_count=nudge_count,
                )

    nudge_count += nudge_raw_away_from_singularities(
        raw, wavelengths_np, float(args.singularity_nudge_threshold)
    )
    with torch.no_grad():
        final = evaluate_current(raw, materials, target_repeated, wavelengths, physics, args)
    update_best(best, final, step=int(args.steps))
    if not bool(torch.isfinite(best["nmae"]).all()):
        bad = torch.nonzero(~torch.isfinite(best["nmae"])).flatten().cpu().tolist()
        raise RuntimeError(f"{task_id}: restarts without valid PyMieDiff result: {bad}")

    output: Dict[str, np.ndarray] = {
        "task_index": np.full(len(raw), int(task_index), dtype=np.int64),
        "task_id": np.full(len(raw), str(task_id), dtype="U64"),
        "target_hash": np.full(len(raw), str(target_hash), dtype="U64"),
        "seed_local_index": seed_local_index,
        "restart_index": restart_index,
        "seed_candidate_uid": seed_uid[seed_local_index].astype("U64"),
        "seed_screen_rank": seed_screen_rank[seed_local_index].astype(np.int64),
        "seed_sequence_id": seed_sequence_id[seed_local_index].astype(np.int64),
        "material_indices": materials_np.astype(np.int64),
        "seed_pgff_nmae": seed_pgff_nmae[seed_local_index].astype(np.float64),
        "seed_geometry": seed_geometry[seed_local_index].astype(np.float64),
        "initial_geometry": initial["geometry"].detach().cpu().numpy().astype(np.float64),
        "initial_mie_nmae": initial["nmae"].detach().cpu().numpy().astype(np.float64),
        "initial_mie_nrmse": initial["nrmse"].detach().cpu().numpy().astype(np.float64),
        "initial_mie_r2": initial["r2"].detach().cpu().numpy().astype(np.float64),
        "initial_mie_peak_error_nm": initial["peak_error_nm"].detach().cpu().numpy().astype(np.float64),
        "initial_min_abs_sin": initial["min_abs_sin"].detach().cpu().numpy().astype(np.float64),
        "best_geometry": best["geometry"].detach().cpu().numpy().astype(np.float64),
        "best_mie_qsca": best["q_sca"].detach().cpu().numpy().astype(np.float64),
        "best_mie_qabs": best["q_abs"].detach().cpu().numpy().astype(np.float64),
        "best_mie_qext": best["q_ext"].detach().cpu().numpy().astype(np.float64),
        "best_mie_nmae": best["nmae"].detach().cpu().numpy().astype(np.float64),
        "best_mie_nrmse": best["nrmse"].detach().cpu().numpy().astype(np.float64),
        "best_mie_r2": best["r2"].detach().cpu().numpy().astype(np.float64),
        "best_mie_peak_error_nm": best["peak_error_nm"].detach().cpu().numpy().astype(np.float64),
        "best_energy_residual": best["energy_residual"].detach().cpu().numpy().astype(np.float64),
        "best_min_abs_sin": best["min_abs_sin"].detach().cpu().numpy().astype(np.float64),
        "best_step": best["step"].detach().cpu().numpy().astype(np.int64),
        "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        "input_hash": np.asarray(input_hash),
        "config_hash": np.asarray(config_hash),
        "nudge_count": np.asarray(int(nudge_count), dtype=np.int64),
    }
    completed = checkpoint_path(args.output_dir, task_id)
    save_npz_atomic(completed, **output)
    if active_path.exists():
        active_path.unlink()
    print(
        f"{task_id}: PyMieDiff continuous refinement completed; "
        f"best NMAE={float(output['best_mie_nmae'].min()):.6f}, "
        f"solver nudges={nudge_count}",
        flush=True,
    )
    return output


def load_completed_task(path: Path, input_hash: str, config_hash: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if str(data["input_hash"].item()) != input_hash:
        raise RuntimeError(f"Completed target checkpoint has a different input hash: {path}")
    if str(data["config_hash"].item()) != config_hash:
        raise RuntimeError(f"Completed target checkpoint has a different config hash: {path}")
    return data


def restart_frame(
    outputs: Sequence[Mapping[str, np.ndarray]],
    material_names: np.ndarray,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for output in outputs:
        n = len(output["restart_index"])
        for i in range(n):
            material = output["material_indices"][i].astype(np.int64)
            seed_geometry = output["seed_geometry"][i]
            initial_geometry = output["initial_geometry"][i]
            best_geometry = output["best_geometry"][i]
            initial_nmae = float(output["initial_mie_nmae"][i])
            best_nmae = float(output["best_mie_nmae"][i])
            rows.append(
                {
                    "task_index": int(output["task_index"][i]),
                    "task_id": str(output["task_id"][i]),
                    "target_hash": str(output["target_hash"][i]),
                    "seed_candidate_uid": str(output["seed_candidate_uid"][i]),
                    "seed_screen_rank": int(output["seed_screen_rank"][i]),
                    "sequence_id": int(output["seed_sequence_id"][i]),
                    "restart_index": int(output["restart_index"][i]),
                    "m0": int(material[0]),
                    "m1": int(material[1]),
                    "m2": int(material[2]),
                    "m3": int(material[3]),
                    "materials": "|".join(material_names[material].tolist()),
                    "seed_pgff_nmae": float(output["seed_pgff_nmae"][i]),
                    "seed_r_core": float(seed_geometry[0]),
                    "seed_t1": float(seed_geometry[1]),
                    "seed_t2": float(seed_geometry[2]),
                    "seed_t3": float(seed_geometry[3]),
                    "initial_r_core": float(initial_geometry[0]),
                    "initial_t1": float(initial_geometry[1]),
                    "initial_t2": float(initial_geometry[2]),
                    "initial_t3": float(initial_geometry[3]),
                    "initial_outer_radius": float(initial_geometry.sum()),
                    "initial_mie_nmae": initial_nmae,
                    "initial_mie_nrmse": float(output["initial_mie_nrmse"][i]),
                    "initial_mie_r2": float(output["initial_mie_r2"][i]),
                    "initial_mie_peak_error_nm": float(output["initial_mie_peak_error_nm"][i]),
                    "initial_min_abs_sin": float(output["initial_min_abs_sin"][i]),
                    "best_r_core": float(best_geometry[0]),
                    "best_t1": float(best_geometry[1]),
                    "best_t2": float(best_geometry[2]),
                    "best_t3": float(best_geometry[3]),
                    "best_outer_radius": float(best_geometry.sum()),
                    "best_mie_nmae": best_nmae,
                    "best_mie_nrmse": float(output["best_mie_nrmse"][i]),
                    "best_mie_r2": float(output["best_mie_r2"][i]),
                    "best_mie_peak_error_nm": float(output["best_mie_peak_error_nm"][i]),
                    "best_energy_residual": float(output["best_energy_residual"][i]),
                    "best_min_abs_sin": float(output["best_min_abs_sin"][i]),
                    "best_step": int(output["best_step"][i]),
                    "absolute_nmae_improvement": initial_nmae - best_nmae,
                    "relative_nmae_improvement": (
                        (initial_nmae - best_nmae) / max(initial_nmae, 1.0e-12)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["task_index", "seed_screen_rank", "restart_index"], kind="mergesort"
    ).reset_index(drop=True)
    return frame


def rank_best_per_seed(frame: pd.DataFrame, stage05_top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranked_restarts = frame.sort_values(
        [
            "task_index",
            "seed_candidate_uid",
            "best_mie_nmae",
            "best_mie_nrmse",
            "best_mie_peak_error_nm",
            "restart_index",
        ],
        ascending=[True, True, True, True, True, True],
        kind="mergesort",
    )
    best_seed = ranked_restarts.groupby(
        ["task_index", "seed_candidate_uid"], sort=False, as_index=False
    ).head(1).copy()
    best_seed = best_seed.sort_values(
        ["task_index", "best_mie_nmae", "best_mie_nrmse", "best_mie_peak_error_nm", "sequence_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    best_seed["continuous_rank"] = best_seed.groupby("task_index").cumcount() + 1
    selected = best_seed.loc[best_seed["continuous_rank"] <= int(stage05_top_k)].copy()
    selected = selected.reset_index(drop=True)
    return best_seed, selected


def history_frame(outputs: Sequence[Mapping[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for output in outputs:
        text = str(np.asarray(output["history_json"]).item())
        rows.extend(json.loads(text))
    return pd.DataFrame(rows).sort_values(["task_index", "step"]).reset_index(drop=True)


def locate_restart_row(output: Mapping[str, np.ndarray], seed_uid: str, restart_index: int) -> int:
    matches = np.flatnonzero(
        (output["seed_candidate_uid"].astype(str) == str(seed_uid))
        & (output["restart_index"].astype(np.int64) == int(restart_index))
    )
    if len(matches) != 1:
        raise RuntimeError("Could not uniquely locate selected restart spectrum.")
    return int(matches[0])


def save_selected_npz(
    path: Path,
    selected: pd.DataFrame,
    outputs_by_task: Mapping[int, Mapping[str, np.ndarray]],
    seeds: Mapping[str, np.ndarray],
) -> None:
    qsca = []
    qabs = []
    qext = []
    for row in selected.itertuples(index=False):
        output = outputs_by_task[int(row.task_index)]
        index = locate_restart_row(output, str(row.seed_candidate_uid), int(row.restart_index))
        qsca.append(output["best_mie_qsca"][index])
        qabs.append(output["best_mie_qabs"][index])
        qext.append(output["best_mie_qext"][index])
    save_npz_atomic(
        path,
        task_id=np.asarray(seeds["task_id"]),
        wavelengths_nm=np.asarray(seeds["wavelengths_nm"], dtype=np.float64),
        target_qsca=np.asarray(seeds["target_qsca"], dtype=np.float64),
        target_hash=np.asarray(seeds["target_hash"]),
        candidate_task_index=selected["task_index"].to_numpy(dtype=np.int64),
        candidate_continuous_rank=selected["continuous_rank"].to_numpy(dtype=np.int64),
        seed_candidate_uid=selected["seed_candidate_uid"].to_numpy(dtype="U64"),
        candidate_sequence_id=selected["sequence_id"].to_numpy(dtype=np.int64),
        candidate_restart_index=selected["restart_index"].to_numpy(dtype=np.int64),
        candidate_material_indices=selected[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64),
        candidate_continuous_geometry=selected[["best_r_core", "best_t1", "best_t2", "best_t3"]].to_numpy(dtype=np.float64),
        candidate_mie_qsca=np.asarray(qsca, dtype=np.float64),
        candidate_mie_qabs=np.asarray(qabs, dtype=np.float64),
        candidate_mie_qext=np.asarray(qext, dtype=np.float64),
        candidate_mie_nmae=selected["best_mie_nmae"].to_numpy(dtype=np.float64),
        candidate_mie_nrmse=selected["best_mie_nrmse"].to_numpy(dtype=np.float64),
        candidate_mie_r2=selected["best_mie_r2"].to_numpy(dtype=np.float64),
        candidate_mie_peak_error_nm=selected["best_mie_peak_error_nm"].to_numpy(dtype=np.float64),
        candidate_min_abs_sin=selected["best_min_abs_sin"].to_numpy(dtype=np.float64),
    )


def plot_results(
    output_dir: Path,
    seeds: Mapping[str, np.ndarray],
    selected: pd.DataFrame,
    history: pd.DataFrame,
    outputs_by_task: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    wavelengths = seeds["wavelengths_nm"]
    figure_dir = Path(output_dir) / "figures"
    for task_index, task_id in enumerate(seeds["task_id"].astype(str)):
        target = seeds["target_qsca"][task_index]
        top = selected.loc[selected["task_index"] == task_index].head(3)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(wavelengths, target, linewidth=2.2, label="Target Qsca")
        for row in top.itertuples(index=False):
            output = outputs_by_task[task_index]
            index = locate_restart_row(output, str(row.seed_candidate_uid), int(row.restart_index))
            ax.plot(
                wavelengths,
                output["best_mie_qsca"][index],
                linewidth=1.5,
                label=f"Rank {int(row.continuous_rank)}: {row.materials}",
            )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Qsca")
        ax.set_title(f"{task_id}: continuous PyMieDiff refinement")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task_id}_spectra.png", dpi=220)
        fig.savefig(figure_dir / f"{task_id}_spectra.pdf")
        plt.close(fig)

        task_history = history.loc[history["task_index"] == task_index]
        fig, ax = plt.subplots(figsize=(6.8, 4.5))
        ax.plot(task_history["step"], task_history["best_nmae"], label="Best NMAE")
        ax.plot(task_history["step"], task_history["median_nmae"], label="Median current NMAE")
        ax.set_xlabel("Optimization step")
        ax.set_ylabel("Target-peak-normalized error")
        ax.set_title(f"{task_id}: PyMieDiff optimization convergence")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task_id}_convergence.png", dpi=220)
        fig.savefig(figure_dir / f"{task_id}_convergence.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    start_time = time.perf_counter()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    prepare_output_dir(args.output_dir, resume=bool(args.resume), overwrite=bool(args.overwrite))

    physics = load_physics_assets(args.dataset)
    stage03 = verify_stage03(args.stage03_dir, args.dataset)
    seeds = load_stage03_seeds(
        stage03["seeds_path"], physics["wavelengths"], physics["material_names"]
    )
    protocol = {
        "restarts_per_seed": int(args.restarts),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "minimum_learning_rate": float(args.minimum_learning_rate),
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "gradient_clip": float(args.gradient_clip),
        "raw_restart_std": float(args.raw_restart_std),
        "history_every": int(args.history_every),
        "print_every": int(args.print_every),
        "stage05_top_k": int(args.stage05_top_k),
        "pymiediff_backend": "pena",
        "pymiediff_precision": "double",
        "solver_singularity_tolerance": float(args.solver_singularity_tolerance),
        "optimization_singularity_margin": float(args.optimization_singularity_margin),
        "singularity_nudge_threshold": float(args.singularity_nudge_threshold),
        "singularity_barrier_weight": float(args.singularity_barrier_weight),
        "negative_efficiency_tolerance": float(args.negative_efficiency_tolerance),
        "energy_residual_tolerance": float(args.energy_residual_tolerance),
        "seed": int(args.seed),
    }
    config_hash = stable_hash(
        {
            "version": VERSION,
            "protocol": protocol,
            "dataset_sha256": sha256_file(args.dataset),
            "stage03_seeds_sha256": sha256_file(stage03["seeds_path"]),
        }
    )

    print("Stage 04 strict-blind PyMieDiff continuous refinement")
    print(f"Device: {device}; PyMieDiff precision: double")
    print(
        f"Blind targets: {len(seeds['task_id'])}; Stage-03 seeds: {len(seeds['candidate_uid'])}; "
        f"restarts per seed: {args.restarts}"
    )
    print(
        f"Continuous optimizations: {len(seeds['candidate_uid']) * int(args.restarts)}; "
        f"steps: {args.steps}"
    )
    print("PGFF, inverse network, private references, Qabs/Qext targets, and test split are not loaded.")

    outputs: list[Dict[str, np.ndarray]] = []
    outputs_by_task: Dict[int, Dict[str, np.ndarray]] = {}
    for task_index, task_id in enumerate(seeds["task_id"].astype(str)):
        rows = np.flatnonzero(seeds["candidate_task_index"] == task_index)
        order = np.argsort(seeds["candidate_screen_rank"][rows], kind="mergesort")
        rows = rows[order]
        input_hash = task_input_hash(
            str(seeds["target_hash"][task_index]),
            seeds["candidate_uid"][rows],
            seeds["candidate_material_indices"][rows],
            seeds["candidate_geometry"][rows],
        )
        completed_path = checkpoint_path(args.output_dir, task_id)
        if args.resume and completed_path.is_file():
            output = load_completed_task(completed_path, input_hash, config_hash)
            print(f"{task_id}: loaded completed target checkpoint.")
        else:
            output = optimize_target(
                task_index=task_index,
                task_id=task_id,
                target_hash=str(seeds["target_hash"][task_index]),
                target_qsca=seeds["target_qsca"][task_index],
                seed_uid=seeds["candidate_uid"][rows],
                seed_screen_rank=seeds["candidate_screen_rank"][rows],
                seed_sequence_id=seeds["candidate_sequence_id"][rows],
                seed_materials=seeds["candidate_material_indices"][rows],
                seed_geometry=seeds["candidate_geometry"][rows],
                seed_pgff_nmae=seeds["candidate_pgff_nmae"][rows],
                physics=physics,
                device=device,
                args=args,
                config_hash=config_hash,
            )
        outputs.append(output)
        outputs_by_task[task_index] = output

    all_restarts = restart_frame(outputs, physics["material_names"])
    best_per_seed, selected_stage05 = rank_best_per_seed(
        all_restarts, int(args.stage05_top_k)
    )
    history = history_frame(outputs)
    if len(all_restarts) != 200 * int(args.restarts):
        raise RuntimeError("Unexpected Stage 04 restart count.")
    if len(best_per_seed) != 200:
        raise RuntimeError("Expected one best restart for each of 200 material seeds.")
    if len(selected_stage05) != 10 * int(args.stage05_top_k):
        raise RuntimeError("Unexpected Stage 05 seed count.")
    if not np.all(geometry_bounds_mask(
        best_per_seed[["best_r_core", "best_t1", "best_t2", "best_t3"]].to_numpy(dtype=np.float64)
    )):
        raise RuntimeError("Final Stage 04 best geometry contains an illegal row.")
    if float(best_per_seed["best_min_abs_sin"].min()) <= float(args.solver_singularity_tolerance):
        raise RuntimeError("Final Stage 04 result violates the PyMieDiff singularity guard.")
    if float(best_per_seed["best_energy_residual"].max()) > float(args.energy_residual_tolerance):
        raise RuntimeError("Final Stage 04 result violates energy closure tolerance.")

    all_csv = args.output_dir / "continuous_refinement_all_restarts.csv"
    best_csv = args.output_dir / "continuous_refinement_best_per_material.csv"
    selected_csv = args.output_dir / "selected_integer_search_candidates.csv"
    history_csv = args.output_dir / "continuous_refinement_history.csv"
    atomic_write_csv(all_csv, all_restarts)
    atomic_write_csv(best_csv, best_per_seed)
    atomic_write_csv(selected_csv, selected_stage05)
    atomic_write_csv(history_csv, history)

    all_qsca = np.concatenate([output["best_mie_qsca"] for output in outputs], axis=0)
    all_qabs = np.concatenate([output["best_mie_qabs"] for output in outputs], axis=0)
    all_qext = np.concatenate([output["best_mie_qext"] for output in outputs], axis=0)
    all_npz = args.output_dir / "continuous_refinement_all_restarts.npz"
    save_npz_atomic(
        all_npz,
        task_id=np.asarray(seeds["task_id"]),
        wavelengths_nm=np.asarray(seeds["wavelengths_nm"], dtype=np.float64),
        target_qsca=np.asarray(seeds["target_qsca"], dtype=np.float64),
        target_hash=np.asarray(seeds["target_hash"]),
        candidate_task_index=all_restarts["task_index"].to_numpy(dtype=np.int64),
        seed_candidate_uid=all_restarts["seed_candidate_uid"].to_numpy(dtype="U64"),
        candidate_restart_index=all_restarts["restart_index"].to_numpy(dtype=np.int64),
        candidate_material_indices=all_restarts[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64),
        candidate_best_geometry=all_restarts[["best_r_core", "best_t1", "best_t2", "best_t3"]].to_numpy(dtype=np.float64),
        candidate_best_mie_qsca=all_qsca,
        candidate_best_mie_qabs=all_qabs,
        candidate_best_mie_qext=all_qext,
        candidate_best_mie_nmae=all_restarts["best_mie_nmae"].to_numpy(dtype=np.float64),
    )
    selected_npz = args.output_dir / "selected_integer_search_candidates.npz"
    save_selected_npz(selected_npz, selected_stage05, outputs_by_task, seeds)

    target_rows = []
    for task_index, task_id in enumerate(seeds["task_id"].astype(str)):
        current = best_per_seed.loc[best_per_seed["task_index"] == task_index]
        best = current.iloc[0]
        target_rows.append(
            {
                "task_index": int(task_index),
                "task_id": str(task_id),
                "best_materials": str(best["materials"]),
                "best_sequence_id": int(best["sequence_id"]),
                "best_r_core": float(best["best_r_core"]),
                "best_t1": float(best["best_t1"]),
                "best_t2": float(best["best_t2"]),
                "best_t3": float(best["best_t3"]),
                "best_outer_radius": float(best["best_outer_radius"]),
                "best_mie_nmae": float(best["best_mie_nmae"]),
                "best_mie_nrmse": float(best["best_mie_nrmse"]),
                "best_mie_r2": float(best["best_mie_r2"]),
                "best_mie_peak_error_nm": float(best["best_mie_peak_error_nm"]),
                "best_min_abs_sin": float(best["best_min_abs_sin"]),
                "best_energy_residual": float(best["best_energy_residual"]),
                "nmae_le_0p05_count": int((current["best_mie_nmae"] <= 0.05).sum()),
                "nmae_le_0p02_count": int((current["best_mie_nmae"] <= 0.02).sum()),
            }
        )
    target_summary = pd.DataFrame(target_rows)
    target_summary_path = args.output_dir / "continuous_refinement_target_summary.csv"
    atomic_write_csv(target_summary_path, target_summary)

    if not args.no_plots:
        plot_results(args.output_dir, seeds, selected_stage05, history, outputs_by_task)

    best_values = target_summary["best_mie_nmae"].to_numpy(dtype=np.float64)
    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": float(time.perf_counter() - start_time),
        "device": str(device),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "blind_policy": {
            "inverse_input": "public target Qsca only",
            "reference_geometry_read": False,
            "reference_materials_read": False,
            "target_qabs_read": False,
            "target_qext_read": False,
            "pgff_loaded": False,
            "inverse_network_loaded": False,
            "forward_training_qsca_loaded": False,
            "forward_test_split_loaded_or_evaluated": False,
            "legacy_inverse_results_read": False,
        },
        "protocol": protocol,
        "dimensions": {
            "blind_targets": int(len(seeds["task_id"])),
            "stage03_material_seeds": 200,
            "restarts_per_seed": int(args.restarts),
            "continuous_optimization_trajectories": int(len(all_restarts)),
            "best_per_material": int(len(best_per_seed)),
            "selected_for_stage05": int(len(selected_stage05)),
            "wavelengths": int(len(seeds["wavelengths_nm"])),
        },
        "high_fidelity_statistics": {
            "mean_target_best_nmae": float(best_values.mean()),
            "median_target_best_nmae": float(np.median(best_values)),
            "maximum_target_best_nmae": float(best_values.max()),
            "minimum_target_best_nmae": float(best_values.min()),
            "targets_nmae_le_0p05": int(np.sum(best_values <= 0.05)),
            "targets_nmae_le_0p02": int(np.sum(best_values <= 0.02)),
            "minimum_final_abs_sin": float(best_per_seed["best_min_abs_sin"].min()),
            "maximum_final_energy_residual": float(best_per_seed["best_energy_residual"].max()),
            "total_solver_nudges": int(sum(int(np.asarray(o["nudge_count"]).item()) for o in outputs)),
        },
        "provenance": {
            "config_hash": config_hash,
            "stage03_status": stage03["summary"].get("status"),
            "stage03_median_best_pgff_nmae": stage03["summary"].get("screening_statistics", {}).get("median_best_pgff_nmae"),
        },
        "inputs": {
            "forward_physics_asset": artifact(args.dataset),
            "stage03_summary": artifact(stage03["summary_path"]),
            "stage03_selected_seeds": artifact(stage03["seeds_path"]),
            "stage03_selected_seeds_csv": artifact(stage03["csv_path"]),
        },
        "outputs": {
            "all_restarts_csv": artifact(all_csv),
            "best_per_material_csv": artifact(best_csv),
            "selected_stage05_csv": artifact(selected_csv),
            "history_csv": artifact(history_csv),
            "target_summary_csv": artifact(target_summary_path),
            "all_restarts_npz": artifact(all_npz),
            "selected_stage05_npz": artifact(selected_npz),
        },
    }
    summary_path = args.output_dir / "continuous_refinement_run_summary.json"
    save_json_atomic(summary_path, summary)

    print("Stage 04 strict-blind PyMieDiff continuous refinement PASSED")
    print(f"Best continuous candidates per material: {len(best_per_seed)}")
    print(f"Selected Stage 05 candidates: {len(selected_stage05)}")
    print(f"Median target-best Mie NMAE: {np.median(best_values):.6f}")
    print(
        "No private reference, target Qabs/Qext, PGFF, inverse network, forward spectra, "
        "or test split was used."
    )


if __name__ == "__main__":
    main()
