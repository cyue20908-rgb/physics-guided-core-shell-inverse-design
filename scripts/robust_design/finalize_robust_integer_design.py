# -*- coding: utf-8 -*-
"""Stage 08D: integer-nanometre robust finalization and one-design freeze.

This stage consumes only the 100 continuous candidates selected by Stage 08C.
For every parent it enumerates a local ±2 nm integer neighbourhood, evaluates
all unique nominal structures with double-precision PyMieDiff, selects a
preregistered nominal pool, and then evaluates all 3^4 = 81 independent ±1 nm
manufacturing states for every selected nominal structure.

The final structure is selected deterministically under the Stage-08 protocol:

1. every one of 81 states must be solver/physics valid and satisfy the frozen
   low-absorption tier;
2. maximize the worst-state mean of the lowest 10% of Qsca wavelengths;
3. among candidates within 1% of that optimum, maximize worst-state mean Qsca;
4. among candidates within 1% of that optimum, minimize worst-state Qsca CV;
5. minimize robust absorption metrics, then outer radius; use deterministic
   sequence/geometry tie breakers.

Exactly one integer four-layer structure is frozen.  No physical interpretation
or multipole/near-field analysis is performed in Stage 08.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any, Dict, Mapping
import hashlib
import json
import time

import numpy as np
import pandas as pd

from core_shell_optical_design.robust_design.robust_design_core import (
    DEFAULT_DATASET,
    RESULT_ROOT,
    ROBUST_LOWER,
    ROBUST_UPPER,
    ROBUST_MAX_OUTER,
    artifact,
    atomic_write_csv,
    boundary_clearance_nm,
    evaluate_design_pool,
    full_perturbations,
    load_json,
    load_physics_assets,
    normalized_absorption_violation_numpy,
    prepare_output_dir,
    resolve_device,
    robust_geometry_mask,
    save_json_atomic,
    save_npz_atomic,
    sequence_strings,
    set_seed,
    sha256_file,
    stable_hash,
    tier_feasible,
    verify_protocol_lock,
)

STAGE = "finalize_robust_integer_design"
VERSION = "goal-free-robust-design-2026-07-25-stage08D-v1"
DEFAULT_STAGE08A_DIR = RESULT_ROOT / "absorption_constraint_calibration"
DEFAULT_STAGE08C_DIR = RESULT_ROOT / "continuous_geometry_refinement"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "full_factorial_robustness_evaluation"
GEOMETRY_COLUMNS = ["r_core", "t1", "t2", "t3"]
MATERIAL_COLUMNS = ["m0", "m1", "m2", "m3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--stage08a-dir", type=Path, default=DEFAULT_STAGE08A_DIR)
    parser.add_argument("--stage08c-dir", type=Path, default=DEFAULT_STAGE08C_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--integer-radius", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--robust-preselect", type=int, default=800)
    parser.add_argument("--max-preselect-per-sequence", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_stage08c(
    stage08c_dir: Path,
    physics: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    summary_path = Path(stage08c_dir) / "robust_refinement_run_summary.json"
    selected_path = Path(stage08c_dir) / "selected_integer_search_parents.npz"
    summary = load_json(summary_path)
    if summary.get("status") != "passed":
        raise RuntimeError("Stage 08C is not a passed run.")
    recorded = summary.get("outputs", {}).get("selected_continuous_npz", {})
    if recorded and str(recorded.get("sha256", "")) != sha256_file(selected_path):
        raise RuntimeError("Stage-08C selected-candidate NPZ hash mismatch.")
    with np.load(selected_path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "wavelengths_nm",
        "stage08d_seed_rank",
        "trajectory_index",
        "sequence_id",
        "material_indices",
        "continuous_geometry_nm",
        "protocol_sha256",
    }
    missing = sorted(required - set(data))
    if missing:
        raise KeyError(f"Stage-08C selected NPZ is missing: {missing}")
    if str(np.asarray(data["protocol_sha256"]).item()) != str(protocol["protocol_sha256"]):
        raise RuntimeError("Stage-08C candidates were generated under another protocol lock.")
    if not np.array_equal(
        np.asarray(data["wavelengths_nm"], dtype=np.float64),
        np.asarray(physics["wavelengths"], dtype=np.float64),
    ):
        raise RuntimeError("Stage-08C and forward-physics wavelength grids differ.")
    geometry = np.asarray(data["continuous_geometry_nm"], dtype=np.float64)
    materials = np.asarray(data["material_indices"], dtype=np.int64)
    formal_count = int(protocol["formal_search"]["continuous_selected_for_integer"])
    if geometry.shape != (formal_count, 4) or materials.shape != (formal_count, 4):
        raise RuntimeError(
            f"Expected {formal_count} Stage-08C candidates, received "
            f"geometry={geometry.shape}, materials={materials.shape}."
        )
    if not np.all(robust_geometry_mask(geometry)):
        raise RuntimeError("Stage-08C contains geometry outside the robust nominal domain.")
    return data, summary


def enumerate_integer_pool(
    continuous_geometry: np.ndarray,
    materials: np.ndarray,
    sequence_id: np.ndarray,
    source_rank: np.ndarray,
    radius_nm: int,
) -> pd.DataFrame:
    offsets = np.asarray(
        list(product(range(-int(radius_nm), int(radius_nm) + 1), repeat=4)),
        dtype=np.int64,
    )
    records: dict[tuple[int, int, int, int, int, int, int, int], Dict[str, Any]] = {}
    for parent in range(len(continuous_geometry)):
        center = np.rint(continuous_geometry[parent]).astype(np.int64)
        neighborhood = center[None, :] + offsets
        legal = robust_geometry_mask(neighborhood.astype(np.float64))
        neighborhood = neighborhood[legal]
        for row in neighborhood:
            key = tuple(int(x) for x in materials[parent]) + tuple(int(x) for x in row)
            distance = float(np.linalg.norm(row.astype(np.float64) - continuous_geometry[parent]))
            candidate = {
                "source_parent_index": int(parent),
                "source_stage08d_seed_rank": int(source_rank[parent]),
                "sequence_id": int(sequence_id[parent]),
                "m0": int(materials[parent, 0]),
                "m1": int(materials[parent, 1]),
                "m2": int(materials[parent, 2]),
                "m3": int(materials[parent, 3]),
                "r_core": int(row[0]),
                "t1": int(row[1]),
                "t2": int(row[2]),
                "t3": int(row[3]),
                "distance_from_continuous_parent_nm": distance,
            }
            previous = records.get(key)
            if previous is None or (
                candidate["source_stage08d_seed_rank"],
                candidate["distance_from_continuous_parent_nm"],
                candidate["source_parent_index"],
            ) < (
                previous["source_stage08d_seed_rank"],
                previous["distance_from_continuous_parent_nm"],
                previous["source_parent_index"],
            ):
                records[key] = candidate
    if not records:
        raise RuntimeError("Integer neighbourhood enumeration produced no legal structure.")
    frame = pd.DataFrame(list(records.values()))
    frame = frame.sort_values(
        [
            "source_stage08d_seed_rank",
            "distance_from_continuous_parent_nm",
            "sequence_id",
            *GEOMETRY_COLUMNS,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "integer_candidate_index", np.arange(len(frame), dtype=np.int64))
    geometry = frame[GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    if not np.all(robust_geometry_mask(geometry)):
        raise RuntimeError("Integer pool contains a structure outside the robust domain.")
    if not np.array_equal(geometry, np.rint(geometry)):
        raise RuntimeError("Integer pool contains a non-integer geometry.")
    return frame


def select_nominal_robust_pool(
    frame: pd.DataFrame,
    *,
    count: int,
    max_per_sequence: int,
) -> pd.DataFrame:
    ordered = frame.sort_values(
        [
            "nominal_absorption_feasible",
            "nominal_absorption_violation",
            "tail_mean_qsca",
            "mean_qsca",
            "qsca_cv",
            "outer_radius",
            "sequence_id",
            *GEOMETRY_COLUMNS,
        ],
        ascending=[False, True, False, False, True, True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    target = min(int(count), len(ordered))
    selected_rows: list[int] = []
    selected_set: set[int] = set()
    counts: dict[int, int] = {}

    # One strongest design from every represented sequence first.
    for idx, row in ordered.iterrows():
        seq = int(row["sequence_id"])
        if counts.get(seq, 0) == 0:
            selected_rows.append(int(idx))
            selected_set.add(int(idx))
            counts[seq] = 1
            if len(selected_rows) >= target:
                break

    # Diversity-capped fill.
    if len(selected_rows) < target:
        for idx, row in ordered.iterrows():
            idx = int(idx)
            if idx in selected_set:
                continue
            seq = int(row["sequence_id"])
            if counts.get(seq, 0) >= int(max_per_sequence):
                continue
            selected_rows.append(idx)
            selected_set.add(idx)
            counts[seq] = counts.get(seq, 0) + 1
            if len(selected_rows) >= target:
                break

    # Deterministic global fill if the cap prevents the preregistered count.
    if len(selected_rows) < target:
        for idx in range(len(ordered)):
            if idx in selected_set:
                continue
            selected_rows.append(idx)
            selected_set.add(idx)
            if len(selected_rows) >= target:
                break

    selected = ordered.iloc[selected_rows].copy().reset_index(drop=True)
    selected.insert(0, "robust_parent_index", np.arange(len(selected), dtype=np.int64))
    selected["nominal_preselection_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
    return selected


def build_unique_robust_state_pool(
    parents: pd.DataFrame,
    perturbations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    unique: dict[tuple[int, int, int, int, int, int, int, int], int] = {}
    unique_geometry: list[np.ndarray] = []
    unique_materials: list[np.ndarray] = []
    map_rows: list[Dict[str, Any]] = []
    for parent_index, parent in parents.iterrows():
        base_geometry = parent[GEOMETRY_COLUMNS].to_numpy(dtype=np.int64)
        material = parent[MATERIAL_COLUMNS].to_numpy(dtype=np.int64)
        states = base_geometry[None, :] + np.asarray(perturbations, dtype=np.int64)
        # Robust nominal domain guarantees all ±1 states are in the original domain.
        for state_index, state_geometry in enumerate(states):
            key = tuple(int(x) for x in material) + tuple(int(x) for x in state_geometry)
            unique_index = unique.get(key)
            if unique_index is None:
                unique_index = len(unique_geometry)
                unique[key] = unique_index
                unique_geometry.append(state_geometry.astype(np.float64))
                unique_materials.append(material.copy())
            delta = perturbations[state_index]
            map_rows.append(
                {
                    "robust_parent_index": int(parent_index),
                    "state_index": int(state_index),
                    "unique_state_index": int(unique_index),
                    "delta_r_core": int(delta[0]),
                    "delta_t1": int(delta[1]),
                    "delta_t2": int(delta[2]),
                    "delta_t3": int(delta[3]),
                    "state_r_core": int(state_geometry[0]),
                    "state_t1": int(state_geometry[1]),
                    "state_t2": int(state_geometry[2]),
                    "state_t3": int(state_geometry[3]),
                }
            )
    return (
        np.asarray(unique_geometry, dtype=np.float64),
        np.asarray(unique_materials, dtype=np.int64),
        pd.DataFrame(map_rows),
    )


def aggregate_full_robustness(
    parents: pd.DataFrame,
    mapping: pd.DataFrame,
    unique_metrics: pd.DataFrame,
    tier: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = mapping.merge(
        unique_metrics.drop(columns=["row_index"]).rename_axis("unique_state_index").reset_index(),
        on="unique_state_index",
        how="left",
        validate="many_to_one",
    )
    rows: list[Dict[str, Any]] = []
    for parent_index, group in state.groupby("robust_parent_index", sort=True):
        if len(group) != 81:
            raise RuntimeError(f"Robust parent {parent_index} has {len(group)} states, expected 81.")
        valid = group["solver_valid"].astype(bool) & group["physics_valid"].astype(bool)
        feasible = valid & tier_feasible(group, tier)
        valid_group = group.loc[valid]
        if valid_group.empty:
            robust_values = {
                "robust_tail_mean_qsca": np.nan,
                "robust_qsca_p10": np.nan,
                "robust_mean_qsca": np.nan,
                "robust_min_qsca": np.nan,
                "robust_qsca_cv": np.nan,
                "robust_band_absorption_fraction": np.nan,
                "robust_absorption_fraction_p95": np.nan,
                "robust_max_qabs": np.nan,
                "robust_mean_qabs": np.nan,
                "robust_single_scattering_albedo": np.nan,
                "robust_energy_residual": np.nan,
                "robust_minimum_abs_sin": np.nan,
            }
        else:
            robust_values = {
                "robust_tail_mean_qsca": float(valid_group["tail_mean_qsca"].min()),
                "robust_qsca_p10": float(valid_group["qsca_p10"].min()),
                "robust_mean_qsca": float(valid_group["mean_qsca"].min()),
                "robust_min_qsca": float(valid_group["min_qsca"].min()),
                "robust_qsca_cv": float(valid_group["qsca_cv"].max()),
                "robust_band_absorption_fraction": float(valid_group["band_absorption_fraction"].max()),
                "robust_absorption_fraction_p95": float(valid_group["absorption_fraction_p95"].max()),
                "robust_max_qabs": float(valid_group["max_qabs"].max()),
                "robust_mean_qabs": float(valid_group["mean_qabs"].max()),
                "robust_single_scattering_albedo": float(valid_group["single_scattering_albedo"].min()),
                "robust_energy_residual": float(valid_group["energy_residual"].max()),
                "robust_minimum_abs_sin": float(valid_group["minimum_abs_sin"].min()),
            }
        tail_row = valid_group.loc[valid_group["tail_mean_qsca"].idxmin()] if valid.any() else group.iloc[0]
        absorption_row = valid_group.loc[valid_group["band_absorption_fraction"].idxmax()] if valid.any() else group.iloc[0]
        rows.append(
            {
                "robust_parent_index": int(parent_index),
                "all_81_states_valid": bool(valid.all()),
                "valid_state_count": int(valid.sum()),
                "all_81_states_absorption_feasible": bool(feasible.all()),
                "absorption_feasible_state_count": int(feasible.sum()),
                **robust_values,
                "worst_tail_state_index": int(tail_row["state_index"]),
                "worst_tail_delta_r_core": int(tail_row["delta_r_core"]),
                "worst_tail_delta_t1": int(tail_row["delta_t1"]),
                "worst_tail_delta_t2": int(tail_row["delta_t2"]),
                "worst_tail_delta_t3": int(tail_row["delta_t3"]),
                "worst_absorption_state_index": int(absorption_row["state_index"]),
                "worst_absorption_delta_r_core": int(absorption_row["delta_r_core"]),
                "worst_absorption_delta_t1": int(absorption_row["delta_t1"]),
                "worst_absorption_delta_t2": int(absorption_row["delta_t2"]),
                "worst_absorption_delta_t3": int(absorption_row["delta_t3"]),
            }
        )
    robust = parents.merge(pd.DataFrame(rows), on="robust_parent_index", validate="one_to_one")
    robust["full_robust_feasible"] = (
        robust["all_81_states_valid"].astype(bool)
        & robust["all_81_states_absorption_feasible"].astype(bool)
    )
    robust["robust_absorption_violation"] = normalized_absorption_violation_numpy(
        robust, tier, prefix="robust_"
    )
    return robust, state


def select_unique_final(robust: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    feasible = robust.loc[robust["full_robust_feasible"]].copy()
    if feasible.empty:
        raise RuntimeError(
            "No integer candidate satisfies the frozen absorption tier in all 81 manufacturing states. "
            "The preregistered protocol forbids post-hoc threshold relaxation."
        )
    primary_max = float(feasible["robust_tail_mean_qsca"].max())
    primary_cut = 0.99 * primary_max
    stage1 = feasible.loc[feasible["robust_tail_mean_qsca"] >= primary_cut - 1.0e-12].copy()
    secondary_max = float(stage1["robust_mean_qsca"].max())
    secondary_cut = 0.99 * secondary_max
    stage2 = stage1.loc[stage1["robust_mean_qsca"] >= secondary_cut - 1.0e-12].copy()
    tertiary_min = float(stage2["robust_qsca_cv"].min())
    tertiary_tolerance = max(1.0e-12, 0.01 * max(abs(tertiary_min), 1.0e-6))
    stage3 = stage2.loc[stage2["robust_qsca_cv"] <= tertiary_min + tertiary_tolerance].copy()
    ordered = stage3.sort_values(
        [
            "robust_band_absorption_fraction",
            "robust_absorption_fraction_p95",
            "robust_max_qabs",
            "outer_radius",
            "boundary_clearance_nm",
            "sequence_id",
            *GEOMETRY_COLUMNS,
        ],
        ascending=[True, True, True, True, False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = ordered.iloc[[0]].copy()
    audit = {
        "feasible_candidate_count": int(len(feasible)),
        "primary_max": primary_max,
        "primary_99pct_cut": primary_cut,
        "primary_near_optimal_count": int(len(stage1)),
        "secondary_max_within_primary_set": secondary_max,
        "secondary_99pct_cut": secondary_cut,
        "secondary_near_optimal_count": int(len(stage2)),
        "tertiary_min_within_secondary_set": tertiary_min,
        "tertiary_tolerance": tertiary_tolerance,
        "tertiary_near_optimal_count": int(len(stage3)),
        "deterministic_final_tie_breaker_count": int(len(ordered)),
    }
    return selected, audit


def self_hash_json(payload: Mapping[str, Any], field: str) -> Dict[str, Any]:
    output = dict(payload)
    output.pop(field, None)
    output[field] = stable_hash(output)
    return output


def main() -> None:
    args = parse_args()
    if args.integer_radius < 0 or args.integer_radius > 10:
        raise ValueError("--integer-radius must be in 0..10.")
    if args.batch_size < 1 or args.robust_preselect < 1:
        raise ValueError("Batch size and robust preselection count must be positive.")
    if args.max_preselect_per_sequence < 1:
        raise ValueError("--max-preselect-per-sequence must be positive.")

    set_seed(int(args.seed))
    prepare_output_dir(
        args.output_dir,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        summary_name="robust_design_run_summary.json",
    )
    started = time.perf_counter()
    device = resolve_device(args.device)
    physics = load_physics_assets(args.dataset)
    protocol_path = args.stage08a_dir / "robust_design_protocol_lock.json"
    protocol = verify_protocol_lock(protocol_path)
    if protocol["input_hashes"]["dataset_sha256"] != sha256_file(args.dataset):
        raise RuntimeError("Dataset differs from the Stage-08 protocol lock.")
    formal = protocol["formal_search"]
    if int(args.integer_radius) != int(formal["integer_neighborhood_radius_nm"]):
        raise RuntimeError("Integer-neighbourhood radius differs from the protocol lock.")
    if int(args.robust_preselect) != int(formal["nominal_candidates_for_full_robustness"]):
        raise RuntimeError("Robust preselection count differs from the protocol lock.")
    tier = dict(protocol["selected_absorption_tier"])
    source, source_summary = load_stage08c(args.stage08c_dir, physics, protocol)
    continuous_geometry = source["continuous_geometry_nm"].astype(np.float64)
    materials = source["material_indices"].astype(np.int64)
    sequence_id = source["sequence_id"].astype(np.int64)
    source_rank = source["stage08d_seed_rank"].astype(np.int64)
    names = np.asarray(physics["material_names"]).astype(str)

    print("Stage 08D integer-nanometre and full ±1 nm robust finalization")
    print(f"Device: {device}; PyMieDiff precision: double")
    print(f"Continuous parents: {len(continuous_geometry)}")
    print(f"Integer neighbourhood radius: ±{args.integer_radius} nm")
    print(f"Selected absorption tier: {tier['name']}")

    integer_pool = enumerate_integer_pool(
        continuous_geometry,
        materials,
        sequence_id,
        source_rank,
        int(args.integer_radius),
    )
    integer_geometry = integer_pool[GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    integer_materials = integer_pool[MATERIAL_COLUMNS].to_numpy(dtype=np.int64)
    integer_pool["materials"] = sequence_strings(integer_materials, names)
    integer_pool["outer_radius"] = integer_geometry.sum(axis=1)
    integer_pool["boundary_clearance_nm"] = boundary_clearance_nm(integer_geometry)

    nominal_checkpoint = args.output_dir / "checkpoints" / "integer_nominal_metrics.npz"
    nominal_metrics, _ = evaluate_design_pool(
        geometry=integer_geometry,
        materials=integer_materials,
        physics=physics,
        device=device,
        batch_size=int(args.batch_size),
        singularity_tolerance=float(protocol["numerics"]["solver_singularity_tolerance"]),
        negative_tolerance=float(protocol["numerics"]["negative_efficiency_tolerance"]),
        energy_tolerance=float(protocol["numerics"]["energy_residual_tolerance"]),
        singularity_policy="symmetric_limit",
        singularity_limit_epsilon_nm=float(protocol["numerics"]["integer_singularity_limit_epsilon_nm"]),
        singularity_limit_rtol=float(protocol["numerics"]["integer_singularity_limit_rtol"]),
        checkpoint_path=nominal_checkpoint,
        checkpoint_every_batches=10,
        resume=bool(args.resume),
        store_spectra=False,
        print_label="Stage 08D nominal integer evaluation",
    )
    nominal = pd.concat(
        [integer_pool.reset_index(drop=True), nominal_metrics.drop(columns=["row_index"])],
        axis=1,
    )
    nominal["nominal_absorption_feasible"] = (
        nominal["solver_valid"].astype(bool).to_numpy()
        & nominal["physics_valid"].astype(bool).to_numpy()
        & tier_feasible(nominal, tier)
    )
    nominal["nominal_absorption_violation"] = normalized_absorption_violation_numpy(nominal, tier)
    nominal_csv = args.output_dir / "all_unique_nominal_integer_candidates.csv"
    atomic_write_csv(nominal_csv, nominal)

    robust_parents = select_nominal_robust_pool(
        nominal,
        count=int(args.robust_preselect),
        max_per_sequence=int(args.max_preselect_per_sequence),
    )
    if len(robust_parents) != int(args.robust_preselect):
        raise RuntimeError(
            f"Expected {args.robust_preselect} robust parents, received {len(robust_parents)}."
        )
    parent_csv = args.output_dir / "selected_full_robustness_parents.csv"
    atomic_write_csv(parent_csv, robust_parents)

    perturbations = full_perturbations().astype(np.int64)
    unique_state_geometry, unique_state_materials, state_mapping = build_unique_robust_state_pool(
        robust_parents, perturbations
    )
    robust_checkpoint = args.output_dir / "checkpoints" / "unique_robust_state_metrics.npz"
    unique_metrics, _ = evaluate_design_pool(
        geometry=unique_state_geometry,
        materials=unique_state_materials,
        physics=physics,
        device=device,
        batch_size=int(args.batch_size),
        singularity_tolerance=float(protocol["numerics"]["solver_singularity_tolerance"]),
        negative_tolerance=float(protocol["numerics"]["negative_efficiency_tolerance"]),
        energy_tolerance=float(protocol["numerics"]["energy_residual_tolerance"]),
        singularity_policy="symmetric_limit",
        singularity_limit_epsilon_nm=float(protocol["numerics"]["integer_singularity_limit_epsilon_nm"]),
        singularity_limit_rtol=float(protocol["numerics"]["integer_singularity_limit_rtol"]),
        checkpoint_path=robust_checkpoint,
        checkpoint_every_batches=10,
        resume=bool(args.resume),
        store_spectra=False,
        print_label="Stage 08D full 81-state evaluation",
    )
    robust, state_metrics = aggregate_full_robustness(
        robust_parents,
        state_mapping,
        unique_metrics,
        tier,
    )
    robust_csv = args.output_dir / "all_full_robust_candidate_metrics.csv"
    state_csv = args.output_dir / "all_full_robust_state_metrics.csv"
    atomic_write_csv(robust_csv, robust)
    atomic_write_csv(state_csv, state_metrics)

    final, selection_audit = select_unique_final(robust)
    final_row = final.iloc[0]
    final_parent = int(final_row["robust_parent_index"])
    final_states = state_metrics.loc[
        state_metrics["robust_parent_index"] == final_parent
    ].sort_values("state_index", kind="mergesort").reset_index(drop=True)
    if len(final_states) != 81:
        raise RuntimeError("Final design does not have exactly 81 manufacturing states.")

    # Fresh, independent recomputation of the frozen design and all 81 states,
    # with spectra retained for downstream physical analysis.
    final_nominal_geometry = final[GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)
    final_materials = final[MATERIAL_COLUMNS].to_numpy(dtype=np.int64)
    final_state_geometry = final_nominal_geometry + perturbations.astype(np.float64)
    final_state_materials = np.repeat(final_materials, 81, axis=0)
    final_checkpoint = args.output_dir / "checkpoints" / "final_81_state_spectra.npz"
    final_metrics, final_spectra = evaluate_design_pool(
        geometry=final_state_geometry,
        materials=final_state_materials,
        physics=physics,
        device=device,
        batch_size=int(args.batch_size),
        singularity_tolerance=float(protocol["numerics"]["solver_singularity_tolerance"]),
        negative_tolerance=float(protocol["numerics"]["negative_efficiency_tolerance"]),
        energy_tolerance=float(protocol["numerics"]["energy_residual_tolerance"]),
        singularity_policy="symmetric_limit",
        singularity_limit_epsilon_nm=float(protocol["numerics"]["integer_singularity_limit_epsilon_nm"]),
        singularity_limit_rtol=float(protocol["numerics"]["integer_singularity_limit_rtol"]),
        checkpoint_path=final_checkpoint,
        checkpoint_every_batches=10,
        resume=bool(args.resume),
        store_spectra=True,
        print_label="Stage 08D frozen-design recomputation",
    )
    if final_spectra is None:
        raise RuntimeError("Final spectrum storage unexpectedly failed.")
    final_state_table = pd.DataFrame(
        {
            "state_index": np.arange(81, dtype=np.int64),
            "delta_r_core": perturbations[:, 0],
            "delta_t1": perturbations[:, 1],
            "delta_t2": perturbations[:, 2],
            "delta_t3": perturbations[:, 3],
            "r_core": final_state_geometry[:, 0].astype(np.int64),
            "t1": final_state_geometry[:, 1].astype(np.int64),
            "t2": final_state_geometry[:, 2].astype(np.int64),
            "t3": final_state_geometry[:, 3].astype(np.int64),
        }
    )
    final_state_table = pd.concat(
        [final_state_table, final_metrics.drop(columns=["row_index"])], axis=1
    )
    final_state_table["absorption_feasible"] = (
        final_state_table["solver_valid"].astype(bool).to_numpy()
        & final_state_table["physics_valid"].astype(bool).to_numpy()
        & tier_feasible(final_state_table, tier)
    )
    if not bool(final_state_table["absorption_feasible"].all()):
        raise RuntimeError("Fresh final recomputation failed the frozen absorption constraints.")
    final_state_csv = args.output_dir / "selected_design_robustness_states.csv"
    atomic_write_csv(final_state_csv, final_state_table)

    spectra_path = args.output_dir / "selected_design_spectra.npz"
    save_npz_atomic(
        spectra_path,
        wavelengths_nm=np.asarray(physics["wavelengths"], dtype=np.float64),
        material_indices=final_materials.astype(np.int64),
        material_names=np.asarray(names[final_materials[0]], dtype="U64"),
        nominal_geometry_nm=final_nominal_geometry.astype(np.int64),
        perturbations_nm=perturbations.astype(np.int64),
        state_geometry_nm=final_state_geometry.astype(np.int64),
        qsca=np.asarray(final_spectra["qsca"], dtype=np.float64),
        qabs=np.asarray(final_spectra["qabs"], dtype=np.float64),
        qext=np.asarray(final_spectra["qext"], dtype=np.float64),
        protocol_sha256=np.asarray(protocol["protocol_sha256"]),
    )

    final_csv = args.output_dir / "selected_robust_design.csv"
    final_export = final.copy()
    final_export.insert(0, "final_rank", 1)
    atomic_write_csv(final_csv, final_export)

    nominal_state = final_state_table.loc[
        (final_state_table[["delta_r_core", "delta_t1", "delta_t2", "delta_t3"]] == 0).all(axis=1)
    ].iloc[0]
    worst_tail_state = final_state_table.loc[final_state_table["tail_mean_qsca"].idxmin()]
    worst_abs_state = final_state_table.loc[final_state_table["band_absorption_fraction"].idxmax()]
    material_indices_final = [int(final_row[col]) for col in MATERIAL_COLUMNS]
    material_names_final = [str(names[index]) for index in material_indices_final]
    geometry_final = [int(final_row[col]) for col in GEOMETRY_COLUMNS]

    final_design_json_path = args.output_dir / "selected_robust_design.json"
    final_design_payload = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "claim_scope": (
            "unique protocol-selected optimum among the preregistered Stage-08 searched candidates; "
            "not a mathematical proof over every possible integer geometry in the full design space"
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "selected_absorption_tier": tier,
        "structure": {
            "sequence_id": int(final_row["sequence_id"]),
            "material_indices": material_indices_final,
            "material_names_core_to_outer": material_names_final,
            "materials": "|".join(material_names_final),
            "r_core_nm": geometry_final[0],
            "t1_nm": geometry_final[1],
            "t2_nm": geometry_final[2],
            "t3_nm": geometry_final[3],
            "interface_radii_nm": [
                geometry_final[0],
                geometry_final[0] + geometry_final[1],
                geometry_final[0] + geometry_final[1] + geometry_final[2],
                sum(geometry_final),
            ],
            "outer_radius_nm": int(sum(geometry_final)),
            "boundary_clearance_nm": float(final_row["boundary_clearance_nm"]),
        },
        "nominal_performance": {
            "tail_mean_qsca": float(nominal_state["tail_mean_qsca"]),
            "qsca_p10": float(nominal_state["qsca_p10"]),
            "mean_qsca": float(nominal_state["mean_qsca"]),
            "min_qsca": float(nominal_state["min_qsca"]),
            "qsca_cv": float(nominal_state["qsca_cv"]),
            "band_absorption_fraction": float(nominal_state["band_absorption_fraction"]),
            "absorption_fraction_p95": float(nominal_state["absorption_fraction_p95"]),
            "max_qabs": float(nominal_state["max_qabs"]),
            "mean_qabs": float(nominal_state["mean_qabs"]),
            "single_scattering_albedo": float(nominal_state["single_scattering_albedo"]),
            "mean_csca_nm2": float(nominal_state["mean_csca_nm2"]),
        },
        "worst_case_over_81_states": {
            "tail_mean_qsca": float(final_row["robust_tail_mean_qsca"]),
            "qsca_p10": float(final_row["robust_qsca_p10"]),
            "mean_qsca": float(final_row["robust_mean_qsca"]),
            "min_qsca": float(final_row["robust_min_qsca"]),
            "qsca_cv": float(final_row["robust_qsca_cv"]),
            "band_absorption_fraction": float(final_row["robust_band_absorption_fraction"]),
            "absorption_fraction_p95": float(final_row["robust_absorption_fraction_p95"]),
            "max_qabs": float(final_row["robust_max_qabs"]),
            "mean_qabs": float(final_row["robust_mean_qabs"]),
            "single_scattering_albedo": float(final_row["robust_single_scattering_albedo"]),
            "all_81_states_valid": bool(final_row["all_81_states_valid"]),
            "all_81_states_absorption_feasible": bool(final_row["all_81_states_absorption_feasible"]),
        },
        "worst_tail_perturbation_nm": {
            "delta_r_core": int(worst_tail_state["delta_r_core"]),
            "delta_t1": int(worst_tail_state["delta_t1"]),
            "delta_t2": int(worst_tail_state["delta_t2"]),
            "delta_t3": int(worst_tail_state["delta_t3"]),
            "tail_mean_qsca": float(worst_tail_state["tail_mean_qsca"]),
        },
        "worst_absorption_perturbation_nm": {
            "delta_r_core": int(worst_abs_state["delta_r_core"]),
            "delta_t1": int(worst_abs_state["delta_t1"]),
            "delta_t2": int(worst_abs_state["delta_t2"]),
            "delta_t3": int(worst_abs_state["delta_t3"]),
            "band_absorption_fraction": float(worst_abs_state["band_absorption_fraction"]),
            "max_qabs": float(worst_abs_state["max_qabs"]),
        },
        "selection_audit": selection_audit,
        "search_counts": {
            "continuous_parents": int(len(continuous_geometry)),
            "unique_nominal_integer_candidates": int(len(nominal)),
            "nominal_feasible_candidates": int(nominal["nominal_absorption_feasible"].sum()),
            "full_robustness_parents": int(len(robust_parents)),
            "unique_robust_states_evaluated": int(len(unique_state_geometry)),
            "full_robust_feasible_candidates": int(robust["full_robust_feasible"].sum()),
        },
        "numerics": protocol["numerics"],
        "artifacts": {
            "final_csv": artifact(final_csv),
            "final_81_state_csv": artifact(final_state_csv),
            "final_spectra_npz": artifact(spectra_path),
        },
    }
    final_design_payload = self_hash_json(final_design_payload, "record_sha256")
    save_json_atomic(final_design_json_path, final_design_payload)

    freeze_path = args.output_dir / "selected_design_freeze_record.json"
    freeze_payload = {
        "stage": STAGE,
        "version": VERSION,
        "status": "frozen",
        "protocol_sha256": protocol["protocol_sha256"],
        "selected_structure": final_design_payload["structure"],
        "selected_metrics": {
            "robust_tail_mean_qsca": float(final_row["robust_tail_mean_qsca"]),
            "robust_mean_qsca": float(final_row["robust_mean_qsca"]),
            "robust_qsca_cv": float(final_row["robust_qsca_cv"]),
            "robust_band_absorption_fraction": float(final_row["robust_band_absorption_fraction"]),
            "robust_absorption_fraction_p95": float(final_row["robust_absorption_fraction_p95"]),
            "robust_max_qabs": float(final_row["robust_max_qabs"]),
        },
        "input_artifacts": {
            "dataset": artifact(args.dataset),
            "protocol_lock": artifact(protocol_path),
            "stage08c_summary": artifact(args.stage08c_dir / "robust_refinement_run_summary.json"),
            "stage08c_selected_candidates": artifact(
                args.stage08c_dir / "selected_integer_search_parents.npz"
            ),
        },
        "output_artifacts": {
            "final_design_json": artifact(final_design_json_path),
            "final_design_csv": artifact(final_csv),
            "final_81_states_csv": artifact(final_state_csv),
            "final_spectra_npz": artifact(spectra_path),
            "all_nominal_candidates": artifact(nominal_csv),
            "all_robust_candidate_metrics": artifact(robust_csv),
        },
    }
    freeze_payload = self_hash_json(freeze_payload, "freeze_record_sha256")
    save_json_atomic(freeze_path, freeze_payload)

    completion_path = args.output_dir / "robust_design_workflow_complete.txt"
    completion_path.write_text(
        "Stage 08 goal-free robust design PASSED\n"
        f"Materials: {'|'.join(material_names_final)}\n"
        f"Geometry [r_core,t1,t2,t3] nm: {geometry_final}\n"
        f"Worst-81-state low-tail Qsca: {float(final_row['robust_tail_mean_qsca']):.10g}\n"
        f"Worst-81-state mean Qsca: {float(final_row['robust_mean_qsca']):.10g}\n"
        f"Worst-81-state band absorption fraction: {float(final_row['robust_band_absorption_fraction']):.10g}\n"
        "Exactly one integer structure has been frozen.\n",
        encoding="utf-8",
    )

    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "selected_absorption_tier": tier,
        "counts": final_design_payload["search_counts"],
        "selected_structure": final_design_payload["structure"],
        "selected_robust_metrics": final_design_payload["worst_case_over_81_states"],
        "selection_audit": selection_audit,
        "policy": {
            "all_400_material_sequences_screened_in_stage08b": True,
            "materials_fixed_in_continuous_and_integer_refinement": True,
            "final_grid_nm": 1,
            "full_pm1nm_states_per_candidate": 81,
            "absorption_is_hard_constraint": True,
            "exactly_one_structure_frozen": True,
            "physical_analysis_performed": False,
            "stage03_to_stage06_candidates_read": False,
        },
        "outputs": {
            "final_design_json": artifact(final_design_json_path),
            "final_design_csv": artifact(final_csv),
            "final_spectra_npz": artifact(spectra_path),
            "final_81_states_csv": artifact(final_state_csv),
            "freeze_record": artifact(freeze_path),
            "completion_marker": artifact(completion_path),
        },
    }
    save_json_atomic(args.output_dir / "robust_design_run_summary.json", summary)

    print("Stage 08D integer robust finalization PASSED")
    print(f"Materials: {'|'.join(material_names_final)}")
    print(f"Geometry [r_core,t1,t2,t3] nm: {geometry_final}")
    print(f"Outer radius: {sum(geometry_final)} nm")
    print(f"Worst-81-state low-tail Qsca: {float(final_row['robust_tail_mean_qsca']):.6f}")
    print(f"Worst-81-state mean Qsca: {float(final_row['robust_mean_qsca']):.6f}")
    print(
        "Worst-81-state band absorption fraction: "
        f"{float(final_row['robust_band_absorption_fraction']):.6e}"
    )
    print(f"Frozen result: {freeze_path}")
    print("Exactly one integer structure was frozen; no physical analysis was performed.")


if __name__ == "__main__":
    main()
