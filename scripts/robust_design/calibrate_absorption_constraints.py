# -*- coding: utf-8 -*-
"""Stage 08A: preregister the goal-free robust design and select absorption tier.

A controlled Sobol geometry bank is evaluated for all 400 legal material
sequences. Every nominal design is tested at the nominal geometry and the eight
axial ±1 nm perturbations. The strictest absorption tier with at least the
preregistered number of feasible designs and distinct material sequences is
frozen into robust_design_protocol_lock.json.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import platform
import time

import numpy as np
import pandas as pd

from core_shell_optical_design.robust_design.robust_design_core import (
    ABSORPTION_TIERS,
    DEFAULT_DATASET,
    PREPARED_DATA,
    RESULT_ROOT,
    aggregate_perturbation_metrics,
    artifact,
    atomic_write_csv,
    axial_perturbations,
    evaluate_design_pool,
    expand_perturbations,
    load_physics_assets,
    load_prepared_registry,
    prepare_output_dir,
    resolve_device,
    save_json_atomic,
    sequence_strings,
    set_seed,
    sha256_file,
    sobol_geometries,
    stable_hash,
    tier_feasible,
    validate_common_self_test,
    verify_frozen_assets,
    write_protocol_lock,
)

STAGE = "calibrate_absorption_constraints"
VERSION = "goal-free-robust-design-2026-07-25-stage08A-v1"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "absorption_constraint_calibration"


@dataclass(frozen=True)
class Protocol:
    calibration_geometries_per_sequence: int = 16
    calibration_perturbation_policy: str = "nominal_plus_eight_axial_pm1nm"
    minimum_feasible_designs: int = 100
    minimum_feasible_sequences: int = 20
    wavelength_weighting: str = "uniform_on_101_point_400_1000nm_grid"
    primary_metric: str = "worst_81_state_bottom_10pct_mean_Qsca"
    secondary_metric: str = "worst_81_state_mean_Qsca"
    tertiary_metric: str = "worst_81_state_Qsca_coefficient_of_variation"
    absorption_policy: str = "all_81_states_must_satisfy_selected_tier"
    nominal_robust_domain: str = "r=121..219,t=21..79,sum<=296nm"
    manufacturing_perturbation: str = "independent integer delta in {-1,0,+1}nm for four geometry variables"
    solver_backend: str = "pena"
    solver_precision: str = "double"
    solver_singularity_tolerance: float = 1.0e-6
    negative_efficiency_tolerance: float = 1.0e-7
    energy_residual_tolerance: float = 1.0e-8
    integer_singularity_policy: str = "symmetric_outer_radius_limit"
    integer_singularity_limit_epsilon_nm: float = 1.0e-3
    integer_singularity_limit_rtol: float = 1.0e-5
    batch_size: int = 64
    checkpoint_every_batches: int = 10
    seed: int = 20260801


def parse_args() -> argparse.Namespace:
    p = Protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--prepared-data", type=Path, default=PREPARED_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--geometries-per-sequence", type=int, default=p.calibration_geometries_per_sequence)
    parser.add_argument("--minimum-feasible-designs", type=int, default=p.minimum_feasible_designs)
    parser.add_argument("--minimum-feasible-sequences", type=int, default=p.minimum_feasible_sequences)
    parser.add_argument("--batch-size", type=int, default=p.batch_size)
    parser.add_argument("--seed", type=int, default=p.seed)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.geometries_per_sequence < 1:
        raise ValueError("--geometries-per-sequence must be positive.")
    if args.minimum_feasible_designs < 1 or args.minimum_feasible_sequences < 1:
        raise ValueError("Feasibility-count thresholds must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        validate_common_self_test()
        print("Stage 08 common self-test PASSED")
        return
    validate_args(args)
    set_seed(int(args.seed))
    prepare_output_dir(
        args.output_dir,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        summary_name="absorption_calibration_run_summary.json",
    )
    started = time.perf_counter()
    device = resolve_device(args.device)
    physics = load_physics_assets(args.dataset)
    prepared = load_prepared_registry(args.prepared_data, physics)
    provenance = verify_frozen_assets(dataset=args.dataset, prepared_data=args.prepared_data)
    sequences = np.asarray(prepared["legal_material_sequences"], dtype=np.int64)
    names = np.asarray(physics["material_names"]).astype(str)

    shared_geometry = sobol_geometries(
        int(args.geometries_per_sequence), seed=int(args.seed), skip=0
    )
    nominal_geometry = np.tile(shared_geometry, (len(sequences), 1))
    nominal_materials = np.repeat(sequences, int(args.geometries_per_sequence), axis=0)
    sequence_id = np.repeat(np.arange(len(sequences), dtype=np.int64), int(args.geometries_per_sequence))
    geometry_id = np.tile(np.arange(int(args.geometries_per_sequence), dtype=np.int64), len(sequences))

    perturbations = axial_perturbations(include_nominal=True)
    state_geometry, state_materials, parent_index, state_index = expand_perturbations(
        nominal_geometry, nominal_materials, perturbations
    )
    checkpoint = args.output_dir / "checkpoints" / "calibration_state_metrics.npz"

    print("Stage 08A preregistration and reachability audit")
    print(f"Device: {device}; PyMieDiff precision: double")
    print(f"Legal material sequences: {len(sequences)}")
    print(f"Calibration nominal designs: {len(nominal_geometry)}")
    print(f"Calibration states (nominal + axial ±1 nm): {len(state_geometry)}")

    state_metrics, _ = evaluate_design_pool(
        geometry=state_geometry,
        materials=state_materials,
        physics=physics,
        device=device,
        batch_size=int(args.batch_size),
        singularity_tolerance=Protocol.solver_singularity_tolerance,
        negative_tolerance=Protocol.negative_efficiency_tolerance,
        energy_tolerance=Protocol.energy_residual_tolerance,
        checkpoint_path=checkpoint,
        checkpoint_every_batches=Protocol.checkpoint_every_batches,
        resume=bool(args.resume),
        store_spectra=False,
        print_label="Stage 08A PyMieDiff calibration",
    )
    state_metrics["parent_index"] = parent_index
    state_metrics["state_index"] = state_index
    state_metrics["delta_r_core"] = perturbations[state_index, 0]
    state_metrics["delta_t1"] = perturbations[state_index, 1]
    state_metrics["delta_t2"] = perturbations[state_index, 2]
    state_metrics["delta_t3"] = perturbations[state_index, 3]

    robust = aggregate_perturbation_metrics(state_metrics, parent_index, prefix="axial_")
    nominal = pd.DataFrame(
        {
            "parent_index": np.arange(len(nominal_geometry), dtype=np.int64),
            "sequence_id": sequence_id,
            "geometry_id": geometry_id,
            "m0": nominal_materials[:, 0],
            "m1": nominal_materials[:, 1],
            "m2": nominal_materials[:, 2],
            "m3": nominal_materials[:, 3],
            "materials": sequence_strings(nominal_materials, names),
            "r_core": nominal_geometry[:, 0],
            "t1": nominal_geometry[:, 1],
            "t2": nominal_geometry[:, 2],
            "t3": nominal_geometry[:, 3],
            "outer_radius": nominal_geometry.sum(axis=1),
        }
    )
    calibration = nominal.merge(robust, on="parent_index", validate="one_to_one")

    tier_rows: list[Dict[str, Any]] = []
    selected_tier: Dict[str, Any] | None = None
    for rank, tier in enumerate(ABSORPTION_TIERS, start=1):
        feasible = tier_feasible(calibration, tier, prefix="axial_") & calibration["axial_all_states_valid"].astype(bool).to_numpy()
        feasible_designs = int(feasible.sum())
        feasible_sequences = int(calibration.loc[feasible, "sequence_id"].nunique())
        tier_rows.append(
            {
                "strictness_rank": rank,
                **tier,
                "feasible_design_count": feasible_designs,
                "feasible_sequence_count": feasible_sequences,
                "selection_requirements_met": bool(
                    feasible_designs >= int(args.minimum_feasible_designs)
                    and feasible_sequences >= int(args.minimum_feasible_sequences)
                ),
            }
        )
        if selected_tier is None and tier_rows[-1]["selection_requirements_met"]:
            selected_tier = dict(tier)

    tier_frame = pd.DataFrame(tier_rows)
    if selected_tier is None:
        atomic_write_csv(args.output_dir / "absorption_tier_reachability.csv", tier_frame)
        atomic_write_csv(args.output_dir / "calibration_nominal_designs.csv", calibration)
        raise RuntimeError(
            "No preregistered low-absorption tier has enough feasible calibration designs. "
            "Do not weaken thresholds after viewing candidates; revise the design-space protocol explicitly."
        )

    selected_feasible = tier_feasible(calibration, selected_tier, prefix="axial_") & calibration["axial_all_states_valid"].astype(bool).to_numpy()
    calibration["selected_tier_feasible"] = selected_feasible
    atomic_write_csv(args.output_dir / "calibration_nominal_designs.csv", calibration)
    atomic_write_csv(args.output_dir / "calibration_state_metrics.csv", state_metrics)
    atomic_write_csv(args.output_dir / "absorption_tier_reachability.csv", tier_frame)

    protocol = {
        "stage": STAGE,
        "version": VERSION,
        "status": "locked",
        "created_from_calibration_only": True,
        "objective": {
            "primary": "maximize minimum over 81 manufacturing states of mean of lowest ceil(10%*Nlambda) Qsca values",
            "secondary": "maximize minimum over 81 states of mean Qsca among candidates within 1% of primary optimum",
            "tertiary": "minimize maximum over 81 states of Qsca coefficient of variation",
            "absorption": "hard feasibility in every one of 81 states",
            "tie_breakers": [
                "minimum robust band absorption fraction",
                "minimum robust absorption-fraction P95",
                "minimum robust maximum Qabs",
                "smaller outer radius",
                "larger boundary clearance",
                "smaller material sequence ID",
                "lexicographic integer geometry",
            ],
        },
        "geometry": {
            "original_bounds_nm": {"r_core": [120, 220], "thickness": [20, 80], "outer_radius_max": 300},
            "robust_nominal_bounds_nm": {"r_core": [121, 219], "thickness": [21, 79], "outer_radius_max": 296},
            "continuous_parameterization_outer_radius_max_nm": 295.5,
            "reason_for_outer_radius_296": "all four geometry variables may simultaneously receive +1 nm, so nominal sum must be <=296 nm",
            "reason_for_continuous_295p5": "avoid placing continuous seeds exactly on removable integer Peña-recursion singularities; exact integer singular points are evaluated by a symmetric limiting procedure in Stage 08D",
            "integer_final_grid_nm": 1,
            "manufacturing_states": 81,
        },
        "selected_absorption_tier": selected_tier,
        "tier_selection_rule": {
            "ordered_tiers": [dict(item) for item in ABSORPTION_TIERS],
            "minimum_feasible_designs": int(args.minimum_feasible_designs),
            "minimum_feasible_sequences": int(args.minimum_feasible_sequences),
            "selected_strictest_tier_meeting_both": True,
            "calibration_uses_nominal_plus_eight_axial_perturbations": True,
            "final_feasibility_uses_all_81_perturbations": True,
        },
        "formal_search": {
            "stage08B_geometries_per_sequence": 20,
            "stage08B_sobol_per_sequence": 16,
            "stage08B_inverse_flat_levels": [2.0, 3.0, 4.0, 5.0],
            "selected_material_sequences": 40,
            "selected_seeds_per_sequence": 5,
            "continuous_restarts_per_seed": 4,
            "continuous_steps": 300,
            "continuous_selected_for_integer": 100,
            "integer_neighborhood_radius_nm": 2,
            "nominal_candidates_for_full_robustness": 800,
        },
        "numerics": {
            "pymiediff_backend": "pena",
            "precision": "double",
            "solver_singularity_tolerance": Protocol.solver_singularity_tolerance,
            "negative_efficiency_tolerance": Protocol.negative_efficiency_tolerance,
            "energy_residual_tolerance": Protocol.energy_residual_tolerance,
            "integer_singularity_policy": Protocol.integer_singularity_policy,
            "integer_singularity_limit_epsilon_nm": Protocol.integer_singularity_limit_epsilon_nm,
            "integer_singularity_limit_rtol": Protocol.integer_singularity_limit_rtol,
        },
        "seeds": {
            "calibration": int(args.seed),
            "formal_screen": 20260802,
            "continuous_refinement": 20260803,
            "integer_finalize": 20260804,
        },
        "inputs": provenance,
        "input_hashes": {
            "dataset_sha256": sha256_file(args.dataset),
            "prepared_data_sha256": sha256_file(args.prepared_data),
        },
        "calibration": {
            "geometries_per_sequence": int(args.geometries_per_sequence),
            "nominal_designs": int(len(nominal_geometry)),
            "evaluated_states": int(len(state_geometry)),
            "selected_tier_feasible_designs": int(selected_feasible.sum()),
            "selected_tier_feasible_sequences": int(calibration.loc[selected_feasible, "sequence_id"].nunique()),
        },
    }
    lock_path = args.output_dir / "robust_design_protocol_lock.json"
    locked = write_protocol_lock(lock_path, protocol)
    selected_path = args.output_dir / "selected_absorption_tier.json"
    save_json_atomic(selected_path, {"selected_absorption_tier": selected_tier, "protocol_sha256": locked["protocol_sha256"]})

    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "python": platform.python_version(),
        "selected_absorption_tier": selected_tier,
        "counts": {
            "legal_material_sequences": int(len(sequences)),
            "nominal_calibration_designs": int(len(nominal_geometry)),
            "calibration_states": int(len(state_geometry)),
            "selected_tier_feasible_designs": int(selected_feasible.sum()),
            "selected_tier_feasible_sequences": int(calibration.loc[selected_feasible, "sequence_id"].nunique()),
        },
        "outputs": {
            "protocol_lock": artifact(lock_path),
            "selected_absorption_tier": artifact(selected_path),
            "calibration_nominal_designs": artifact(args.output_dir / "calibration_nominal_designs.csv"),
            "calibration_state_metrics": artifact(args.output_dir / "calibration_state_metrics.csv"),
            "tier_reachability": artifact(args.output_dir / "absorption_tier_reachability.csv"),
        },
    }
    save_json_atomic(args.output_dir / "absorption_calibration_run_summary.json", summary)

    print("Stage 08A preregistration and reachability PASSED")
    print(f"Selected absorption tier: {selected_tier['name']}")
    print(f"Feasible calibration designs: {int(selected_feasible.sum())}")
    print(f"Feasible material sequences: {int(calibration.loc[selected_feasible, 'sequence_id'].nunique())}")
    print(f"Protocol lock: {lock_path}")


if __name__ == "__main__":
    main()
