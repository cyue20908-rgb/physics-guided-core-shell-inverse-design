# -*- coding: utf-8 -*-
"""Stage 08C: constrained differentiable PyMieDiff continuous refinement.

For each of the 40 material sequences selected by Stage 08B, five geometry
seeds and four deterministic restarts are optimized. Materials remain exact and
fixed. The objective maximizes a robust low-tail Qsca score while enforcing the
frozen low-absorption tier. Each optimization step evaluates the nominal state
and one deterministic continuous ±1 nm perturbation; final trajectory ranking
uses the nominal state plus all eight axial ±1 nm states.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from core_shell_optical_design.robust_design.robust_design_core import (
    DEFAULT_DATASET,
    RESULT_ROOT,
    absorption_violation_torch,
    aggregate_perturbation_metrics,
    artifact,
    atomic_write_csv,
    axial_perturbations,
    evaluate_design_pool,
    expand_perturbations,
    load_json,
    load_physics_assets,
    normalized_absorption_violation_numpy,
    prepare_output_dir,
    recursive_forward,
    resolve_device,
    robust_geometry_from_raw,
    robust_raw_from_geometry,
    save_json_atomic,
    save_npz_atomic,
    sequence_strings,
    set_seed,
    sha256_file,
    spectral_metrics_torch,
    tier_feasible,
    torch_min_abs_sin,
    verify_protocol_lock,
)
from core_shell_optical_design.robust_design.robust_design_core import physical_validity  # explicit internal physics audit

STAGE = "refine_robust_geometries"
VERSION = "goal-free-robust-design-2026-07-25-stage08C-v1"
DEFAULT_STAGE08A_DIR = RESULT_ROOT / "absorption_constraint_calibration"
DEFAULT_STAGE08B_DIR = RESULT_ROOT / "material_candidate_screening"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "continuous_geometry_refinement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--stage08a-dir", type=Path, default=DEFAULT_STAGE08A_DIR)
    parser.add_argument("--stage08b-dir", type=Path, default=DEFAULT_STAGE08B_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3.0e-2)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--raw-restart-std", type=float, default=0.30)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--softmin-temperature", type=float, default=0.05)
    parser.add_argument("--absorption-penalty-weight", type=float, default=25.0)
    parser.add_argument("--flatness-weight", type=float, default=0.03)
    parser.add_argument("--mean-scattering-weight", type=float, default=0.05)
    parser.add_argument("--singularity-barrier-weight", type=float, default=0.05)
    parser.add_argument("--singularity-margin", type=float, default=1.0e-3)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--active-checkpoint-every", type=int, default=25)
    parser.add_argument("--selected-for-integer", type=int, default=100)
    parser.add_argument("--max-per-sequence", type=int, default=3)
    parser.add_argument("--axial-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, dict):
        raise TypeError(f"Expected checkpoint dictionary in {path}.")
    return value


def save_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temp)
    os.replace(temp, path)


def load_stage08b(path: Path, physics: Mapping[str, Any], protocol: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    summary_path = path / "material_screening_run_summary.json"
    selected_path = path / "selected_continuous_refinement_seeds.npz"
    summary = load_json(summary_path)
    if summary.get("status") != "passed":
        raise RuntimeError("Stage 08B is not a passed run.")
    recorded = summary.get("outputs", {}).get("selected_seeds_npz", {})
    if recorded and recorded.get("sha256") != sha256_file(selected_path):
        raise RuntimeError("Stage-08B selected seed NPZ hash mismatch.")
    with np.load(selected_path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if str(np.asarray(data["protocol_sha256"]).item()) != str(protocol["protocol_sha256"]):
        raise RuntimeError("Stage-08B seeds were generated under another protocol lock.")
    if not np.array_equal(data["wavelengths_nm"].astype(np.float64), np.asarray(physics["wavelengths"], dtype=np.float64)):
        raise RuntimeError("Stage-08B and physics wavelength grids differ.")
    if data["geometry_nm"].shape != (200, 4) or data["material_indices"].shape != (200, 4):
        raise RuntimeError("Expected exactly 200 Stage-08B continuous seeds.")
    return data


def smooth_min_pair(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1.0e-6)
    values = torch.stack((a, b), dim=1)
    return -t * (torch.logsumexp(-values / t, dim=1) - math.log(2.0))


def evaluate_differentiable_state(
    geometry: torch.Tensor,
    materials: torch.Tensor,
    physics: Mapping[str, Any],
    wavelengths: torch.Tensor,
    *,
    negative_tolerance: float,
    energy_tolerance: float,
) -> Dict[str, torch.Tensor]:
    result = __import__(
        "core_shell_optical_design.inverse_design.forward_model_bridge",
        fromlist=["pymiediff_forward"],
    ).pymiediff_forward(
        geometry, materials, physics, backend="pena", precision="double"
    )
    qsca, qabs, qext = result["q_sca"], result["q_abs"], result["q_ext"]
    valid, residual, minimum = physical_validity(
        qsca, qabs, qext,
        negative_tolerance=float(negative_tolerance),
        energy_tolerance=float(energy_tolerance),
    )
    clean_qsca = torch.clamp(qsca, min=0.0)
    clean_qabs = torch.clamp(qabs, min=0.0)
    clean_qext = torch.clamp(qext, min=0.0)
    metrics = spectral_metrics_torch(clean_qsca, clean_qabs, clean_qext, geometry.sum(dim=1))
    metrics.update(
        {
            "qsca": clean_qsca,
            "qabs": clean_qabs,
            "qext": clean_qext,
            "valid": valid,
            "energy_residual": residual,
            "minimum_efficiency": minimum,
            "minimum_abs_sin": torch_min_abs_sin(geometry, wavelengths),
        }
    )
    return metrics


def initialize_best(count: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "feasible": torch.zeros(count, dtype=torch.bool, device=device),
        "violation": torch.full((count,), torch.inf, dtype=torch.float64, device=device),
        "tail": torch.full((count,), -torch.inf, dtype=torch.float64, device=device),
        "mean": torch.full((count,), -torch.inf, dtype=torch.float64, device=device),
        "cv": torch.full((count,), torch.inf, dtype=torch.float64, device=device),
        "geometry": torch.zeros((count, 4), dtype=torch.float64, device=device),
        "step": torch.full((count,), -1, dtype=torch.long, device=device),
    }


def update_best(
    best: Dict[str, torch.Tensor],
    *,
    feasible: torch.Tensor,
    violation: torch.Tensor,
    tail: torch.Tensor,
    mean: torch.Tensor,
    cv: torch.Tensor,
    geometry: torch.Tensor,
    step: int,
) -> None:
    better_feasible = feasible & ~best["feasible"]
    both_feasible = feasible & best["feasible"]
    better_tail = tail > best["tail"] + 1.0e-12
    tied_tail = torch.abs(tail - best["tail"]) <= 1.0e-12
    better_mean = mean > best["mean"] + 1.0e-12
    tied_mean = torch.abs(mean - best["mean"]) <= 1.0e-12
    better_cv = cv < best["cv"] - 1.0e-12
    feasible_update = both_feasible & (better_tail | (tied_tail & (better_mean | (tied_mean & better_cv))))
    both_infeasible = ~feasible & ~best["feasible"]
    infeasible_update = both_infeasible & (
        (violation < best["violation"] - 1.0e-12)
        | ((torch.abs(violation - best["violation"]) <= 1.0e-12) & better_tail)
    )
    update = better_feasible | feasible_update | infeasible_update
    if bool(update.any()):
        best["feasible"][update] = feasible[update]
        best["violation"][update] = violation[update]
        best["tail"][update] = tail[update]
        best["mean"][update] = mean[update]
        best["cv"][update] = cv[update]
        best["geometry"][update] = geometry[update].detach()
        best["step"][update] = int(step)


def optimize_sequence(
    *,
    sequence_id: int,
    seed_indices: np.ndarray,
    seed_geometry: np.ndarray,
    material_indices: np.ndarray,
    physics: Mapping[str, Any],
    protocol: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    tier = protocol["selected_absorption_tier"]
    numerics = protocol["numerics"]
    count_seed = len(seed_geometry)
    trajectory_seed_local = np.repeat(np.arange(count_seed, dtype=np.int64), int(args.restarts))
    restart_index = np.tile(np.arange(int(args.restarts), dtype=np.int64), count_seed)
    base = torch.as_tensor(seed_geometry, dtype=torch.float64, device=device)
    base_raw = robust_raw_from_geometry(base)
    raw_rows: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + int(sequence_id) * 1000003)
    for seed_local in range(count_seed):
        for restart in range(int(args.restarts)):
            value = base_raw[seed_local].clone()
            if restart > 0 and float(args.raw_restart_std) > 0.0:
                noise = torch.randn(4, generator=generator, dtype=torch.float64) * float(args.raw_restart_std)
                value = value + noise.to(device)
            raw_rows.append(value)
    raw = torch.nn.Parameter(torch.stack(raw_rows, dim=0))
    materials = torch.as_tensor(
        np.repeat(material_indices[None, :], len(raw_rows), axis=0),
        dtype=torch.long,
        device=device,
    )
    optimizer = torch.optim.Adam([raw], lr=float(args.learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.steps)), eta_min=float(args.minimum_learning_rate)
    )
    best = initialize_best(len(raw_rows), device)
    history: list[dict[str, float | int]] = []
    wavelengths = torch.as_tensor(physics["wavelengths"], dtype=torch.float64, device=device)
    sobol = torch.quasirandom.SobolEngine(
        dimension=4, scramble=True, seed=int(args.seed) + 7919 * int(sequence_id)
    )
    perturbations = 2.0 * sobol.draw(int(args.steps)).double().to(device) - 1.0
    active_path = args.output_dir / "checkpoints" / f"sequence_{sequence_id:03d}_active.pt"
    completed_path = args.output_dir / "checkpoints" / f"sequence_{sequence_id:03d}_complete.npz"
    start_step = 0
    if args.resume and completed_path.is_file():
        with np.load(completed_path, allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    if args.resume and active_path.is_file():
        payload = torch_load(active_path, device)
        if str(payload["protocol_sha256"]) != str(protocol["protocol_sha256"]):
            raise RuntimeError(f"Active checkpoint protocol mismatch for sequence {sequence_id}.")
        raw.data.copy_(payload["raw"].to(device=device, dtype=torch.float64))
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        best = {key: value.to(device) for key, value in payload["best"].items()}
        history = list(payload["history"])
        start_step = int(payload["step"]) + 1
        print(f"Sequence {sequence_id:03d}: resuming at step {start_step}/{args.steps}")

    for step in range(start_step, int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        geometry = robust_geometry_from_raw(raw)
        perturbed = geometry + perturbations[step][None, :]
        try:
            nominal = evaluate_differentiable_state(
                geometry, materials, physics, wavelengths,
                negative_tolerance=float(numerics["negative_efficiency_tolerance"]),
                energy_tolerance=float(numerics["energy_residual_tolerance"]),
            )
            shifted = evaluate_differentiable_state(
                perturbed, materials, physics, wavelengths,
                negative_tolerance=float(numerics["negative_efficiency_tolerance"]),
                energy_tolerance=float(numerics["energy_residual_tolerance"]),
            )
        except Exception:
            with torch.no_grad():
                raw.add_(1.0e-3 * torch.sign(torch.randn_like(raw)))
            scheduler.step()
            continue

        robust_tail = smooth_min_pair(
            nominal["tail_mean_qsca"], shifted["tail_mean_qsca"], float(args.softmin_temperature)
        )
        robust_mean = smooth_min_pair(
            nominal["mean_qsca"], shifted["mean_qsca"], float(args.softmin_temperature)
        )
        robust_cv = torch.maximum(nominal["qsca_cv"], shifted["qsca_cv"])
        violation_nominal = absorption_violation_torch(nominal, tier)
        violation_shifted = absorption_violation_torch(shifted, tier)
        violation = torch.maximum(violation_nominal, violation_shifted)
        violation_score = torch.square(violation).sum(dim=1)
        min_sine = torch.minimum(nominal["minimum_abs_sin"], shifted["minimum_abs_sin"])
        singularity_barrier = torch.square(
            torch.relu(float(args.singularity_margin) - min_sine) / float(args.singularity_margin)
        )
        physics_ok = nominal["valid"] & shifted["valid"] & (
            min_sine > float(numerics["solver_singularity_tolerance"])
        )
        absorption_ok = (violation <= 0.0).all(dim=1)
        feasible = physics_ok & absorption_ok
        loss_per = (
            -robust_tail
            - float(args.mean_scattering_weight) * robust_mean
            + float(args.flatness_weight) * robust_cv
            + float(args.absorption_penalty_weight) * violation_score
            + float(args.singularity_barrier_weight) * singularity_barrier
        )
        invalid_penalty = (~physics_ok).to(loss_per.dtype) * 100.0
        loss = (loss_per + invalid_penalty).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage-08C loss for sequence {sequence_id}, step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], float(args.gradient_clip))
        optimizer.step()
        scheduler.step()
        update_best(
            best,
            feasible=feasible.detach(),
            violation=violation_score.detach(),
            tail=robust_tail.detach(),
            mean=robust_mean.detach(),
            cv=robust_cv.detach(),
            geometry=geometry.detach(),
            step=step + 1,
        )
        if (step + 1) % int(args.print_every) == 0 or step == 0 or step + 1 == int(args.steps):
            print(
                f"Sequence {sequence_id:03d} step {step + 1:4d}/{args.steps}: "
                f"feasible={int(feasible.sum())}/{len(feasible)}, "
                f"best-tail={float(torch.max(best['tail']).cpu()):.6f}, "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )
        if (step + 1) % int(args.print_every) == 0 or step + 1 == int(args.steps):
            history.append(
                {
                    "sequence_id": int(sequence_id),
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "feasible_count": int(feasible.sum().detach().cpu()),
                    "median_tail": float(torch.median(robust_tail.detach()).cpu()),
                    "best_recorded_tail": float(torch.max(best["tail"]).cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
        if (step + 1) % int(args.active_checkpoint_every) == 0:
            save_torch_atomic(
                active_path,
                {
                    "protocol_sha256": protocol["protocol_sha256"],
                    "step": int(step),
                    "raw": raw.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best": {key: value.detach().cpu() for key, value in best.items()},
                    "history": history,
                },
            )

    # A pathological solver failure across every optimization step must not
    # export the zero-filled initializer.  Preserve the final legal geometry
    # for downstream exact axial validation; such a trajectory remains marked
    # infeasible and cannot win the final ranking.
    uninitialized = best["step"] < 0
    if bool(uninitialized.any()):
        with torch.no_grad():
            fallback_geometry = robust_geometry_from_raw(raw)
        best["geometry"][uninitialized] = fallback_geometry[uninitialized]
        best["violation"][uninitialized] = torch.inf
        best["tail"][uninitialized] = -torch.inf
        best["mean"][uninitialized] = -torch.inf
        best["cv"][uninitialized] = torch.inf

    output = {
        "sequence_id": np.full(len(raw_rows), int(sequence_id), dtype=np.int64),
        "source_seed_index": np.asarray(seed_indices, dtype=np.int64)[trajectory_seed_local],
        "seed_local_index": trajectory_seed_local,
        "restart_index": restart_index,
        "material_indices": np.repeat(material_indices[None, :], len(raw_rows), axis=0).astype(np.int64),
        "best_geometry_nm": best["geometry"].detach().cpu().numpy(),
        "optimization_feasible": best["feasible"].detach().cpu().numpy(),
        "optimization_violation": best["violation"].detach().cpu().numpy(),
        "optimization_robust_tail": best["tail"].detach().cpu().numpy(),
        "optimization_robust_mean": best["mean"].detach().cpu().numpy(),
        "optimization_robust_cv": best["cv"].detach().cpu().numpy(),
        "best_step": best["step"].detach().cpu().numpy(),
        "history_json": np.asarray(__import__("json").dumps(history, ensure_ascii=False)),
        "protocol_sha256": np.asarray(protocol["protocol_sha256"]),
    }
    save_npz_atomic(completed_path, **output)
    if active_path.exists():
        active_path.unlink()
    return output


def diversity_select(frame: pd.DataFrame, count: int, max_per_sequence: int) -> pd.DataFrame:
    ordered = frame.sort_values(
        [
            "axial_absorption_feasible",
            "axial_normalized_absorption_violation",
            "axial_tail_mean_qsca",
            "axial_mean_qsca",
            "axial_qsca_cv",
            "sequence_id",
            "trajectory_index",
        ],
        ascending=[False, True, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    chosen: list[int] = []
    counts: dict[int, int] = {}
    # First retain one trajectory from every sequence.
    for idx, row in ordered.iterrows():
        seq = int(row["sequence_id"])
        if counts.get(seq, 0) == 0:
            chosen.append(idx)
            counts[seq] = 1
    for idx, row in ordered.iterrows():
        if len(chosen) >= int(count):
            break
        if idx in chosen:
            continue
        seq = int(row["sequence_id"])
        if counts.get(seq, 0) >= int(max_per_sequence):
            continue
        chosen.append(idx)
        counts[seq] = counts.get(seq, 0) + 1
    if len(chosen) < int(count):
        raise RuntimeError("Diversity constraints cannot supply the requested Stage-08D candidate count.")
    selected = ordered.iloc[chosen[: int(count)]].copy().reset_index(drop=True)
    selected["stage08d_seed_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
    return selected


def main() -> None:
    args = parse_args()
    if args.restarts < 1 or args.steps < 1:
        raise ValueError("Restart and step counts must be positive.")
    if args.minimum_learning_rate <= 0 or args.learning_rate < args.minimum_learning_rate:
        raise ValueError("Invalid learning-rate schedule.")
    set_seed(int(args.seed))
    prepare_output_dir(
        args.output_dir,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        summary_name="robust_refinement_run_summary.json",
    )
    started = time.perf_counter()
    device = resolve_device(args.device)
    physics = load_physics_assets(args.dataset)
    protocol_path = args.stage08a_dir / "robust_design_protocol_lock.json"
    protocol = verify_protocol_lock(protocol_path)
    if protocol["input_hashes"]["dataset_sha256"] != sha256_file(args.dataset):
        raise RuntimeError("Dataset differs from Stage-08 protocol lock.")
    formal = protocol["formal_search"]
    if int(args.restarts) != int(formal["continuous_restarts_per_seed"]):
        raise RuntimeError("Restart count differs from preregistered protocol.")
    if int(args.steps) != int(formal["continuous_steps"]):
        raise RuntimeError("Continuous step count differs from preregistered protocol.")
    if int(args.selected_for_integer) != int(formal["continuous_selected_for_integer"]):
        raise RuntimeError("Stage-08D seed count differs from preregistered protocol.")
    seeds = load_stage08b(args.stage08b_dir, physics, protocol)
    geometry = seeds["geometry_nm"].astype(np.float64)
    materials = seeds["material_indices"].astype(np.int64)
    sequence_id = seeds["sequence_id"].astype(np.int64)
    names = np.asarray(physics["material_names"]).astype(str)

    print("Stage 08C constrained continuous PyMieDiff refinement")
    print(f"Device: {device}; seeds={len(geometry)}; restarts={args.restarts}; steps={args.steps}")
    print(f"Selected absorption tier: {protocol['selected_absorption_tier']['name']}")

    outputs: list[Dict[str, np.ndarray]] = []
    for seq in sorted(np.unique(sequence_id).tolist()):
        rows = np.flatnonzero(sequence_id == seq)
        output = optimize_sequence(
            sequence_id=int(seq),
            seed_indices=seeds["selected_seed_index"][rows].astype(np.int64),
            seed_geometry=geometry[rows],
            material_indices=materials[rows[0]],
            physics=physics,
            protocol=protocol,
            device=device,
            args=args,
        )
        outputs.append(output)

    all_geometry = np.concatenate([item["best_geometry_nm"] for item in outputs], axis=0)
    all_materials = np.concatenate([item["material_indices"] for item in outputs], axis=0)
    all_sequence = np.concatenate([item["sequence_id"] for item in outputs], axis=0)
    all_source_seed = np.concatenate([item["source_seed_index"] for item in outputs], axis=0)
    all_restart = np.concatenate([item["restart_index"] for item in outputs], axis=0)
    all_best_step = np.concatenate([item["best_step"] for item in outputs], axis=0)
    opt_feasible = np.concatenate([item["optimization_feasible"] for item in outputs], axis=0)
    opt_violation = np.concatenate([item["optimization_violation"] for item in outputs], axis=0)
    opt_tail = np.concatenate([item["optimization_robust_tail"] for item in outputs], axis=0)
    opt_mean = np.concatenate([item["optimization_robust_mean"] for item in outputs], axis=0)
    opt_cv = np.concatenate([item["optimization_robust_cv"] for item in outputs], axis=0)
    trajectory_index = np.arange(len(all_geometry), dtype=np.int64)
    if len(all_geometry) != 200 * int(args.restarts):
        raise RuntimeError("Unexpected number of Stage-08C trajectories.")

    axial = axial_perturbations(include_nominal=True)
    state_geometry, state_materials, parent_index, state_index = expand_perturbations(
        all_geometry, all_materials, axial
    )
    axial_checkpoint = args.output_dir / "checkpoints" / "all_trajectory_axial_metrics.npz"
    state_metrics, _ = evaluate_design_pool(
        geometry=state_geometry,
        materials=state_materials,
        physics=physics,
        device=device,
        batch_size=int(args.axial_batch_size),
        singularity_tolerance=float(protocol["numerics"]["solver_singularity_tolerance"]),
        negative_tolerance=float(protocol["numerics"]["negative_efficiency_tolerance"]),
        energy_tolerance=float(protocol["numerics"]["energy_residual_tolerance"]),
        checkpoint_path=axial_checkpoint,
        checkpoint_every_batches=10,
        resume=bool(args.resume),
        store_spectra=False,
        print_label="Stage 08C axial verification",
    )
    axial_agg = aggregate_perturbation_metrics(state_metrics, parent_index, prefix="axial_")
    frame = pd.DataFrame(
        {
            "trajectory_index": trajectory_index,
            "sequence_id": all_sequence,
            "source_seed_index": all_source_seed,
            "restart_index": all_restart,
            "best_step": all_best_step,
            "m0": all_materials[:, 0],
            "m1": all_materials[:, 1],
            "m2": all_materials[:, 2],
            "m3": all_materials[:, 3],
            "materials": sequence_strings(all_materials, names),
            "r_core": all_geometry[:, 0],
            "t1": all_geometry[:, 1],
            "t2": all_geometry[:, 2],
            "t3": all_geometry[:, 3],
            "outer_radius": all_geometry.sum(axis=1),
            "optimization_feasible": opt_feasible,
            "optimization_violation": opt_violation,
            "optimization_robust_tail": opt_tail,
            "optimization_robust_mean": opt_mean,
            "optimization_robust_cv": opt_cv,
        }
    ).merge(axial_agg, left_on="trajectory_index", right_on="parent_index", validate="one_to_one")
    frame.drop(columns=["parent_index"], inplace=True)
    tier = protocol["selected_absorption_tier"]
    frame["axial_absorption_feasible"] = (
        tier_feasible(frame, tier, prefix="axial_") & frame["axial_all_states_valid"].astype(bool).to_numpy()
    )
    frame["axial_normalized_absorption_violation"] = normalized_absorption_violation_numpy(
        frame, tier, prefix="axial_"
    )
    selected = diversity_select(
        frame, int(args.selected_for_integer), int(args.max_per_sequence)
    )

    all_csv = args.output_dir / "continuous_trajectory_best.csv"
    selected_csv = args.output_dir / "selected_integer_search_parents.csv"
    atomic_write_csv(all_csv, frame.sort_values(["sequence_id", "trajectory_index"]))
    atomic_write_csv(selected_csv, selected)
    selected_npz = args.output_dir / "selected_integer_search_parents.npz"
    save_npz_atomic(
        selected_npz,
        wavelengths_nm=np.asarray(physics["wavelengths"], dtype=np.float64),
        stage08d_seed_rank=selected["stage08d_seed_rank"].to_numpy(dtype=np.int64),
        trajectory_index=selected["trajectory_index"].to_numpy(dtype=np.int64),
        sequence_id=selected["sequence_id"].to_numpy(dtype=np.int64),
        material_indices=selected[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64),
        continuous_geometry_nm=selected[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.float64),
        axial_tail_mean_qsca=selected["axial_tail_mean_qsca"].to_numpy(dtype=np.float64),
        axial_mean_qsca=selected["axial_mean_qsca"].to_numpy(dtype=np.float64),
        axial_qsca_cv=selected["axial_qsca_cv"].to_numpy(dtype=np.float64),
        axial_band_absorption_fraction=selected["axial_band_absorption_fraction"].to_numpy(dtype=np.float64),
        axial_absorption_fraction_p95=selected["axial_absorption_fraction_p95"].to_numpy(dtype=np.float64),
        axial_max_qabs=selected["axial_max_qabs"].to_numpy(dtype=np.float64),
        protocol_sha256=np.asarray(protocol["protocol_sha256"]),
    )

    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "selected_absorption_tier": tier,
        "counts": {
            "stage08b_seeds": 200,
            "restarts_per_seed": int(args.restarts),
            "continuous_trajectories": int(len(frame)),
            "axial_feasible_trajectories": int(frame["axial_absorption_feasible"].sum()),
            "selected_stage08d_candidates": int(len(selected)),
            "selected_material_sequences": int(selected["sequence_id"].nunique()),
        },
        "policy": {
            "materials_fixed_during_refinement": True,
            "pymiediff_double_precision": True,
            "nominal_and_deterministic_continuous_perturbation_per_step": True,
            "final_trajectory_ranking_uses_nine_axial_states": True,
            "stage03_to_stage06_candidates_read": False,
        },
        "inputs": {
            "dataset": artifact(args.dataset),
            "protocol_lock": artifact(protocol_path),
            "stage08b_selected_seeds": artifact(args.stage08b_dir / "selected_continuous_refinement_seeds.npz"),
        },
        "outputs": {
            "all_trajectory_best": artifact(all_csv),
            "selected_continuous_csv": artifact(selected_csv),
            "selected_continuous_npz": artifact(selected_npz),
        },
    }
    save_json_atomic(args.output_dir / "robust_refinement_run_summary.json", summary)

    print("Stage 08C constrained continuous refinement PASSED")
    print(f"Continuous trajectories: {len(frame)}")
    print(f"Axial-feasible trajectories: {int(frame['axial_absorption_feasible'].sum())}")
    print(f"Selected Stage 08D candidates: {len(selected)}")


if __name__ == "__main__":
    main()
