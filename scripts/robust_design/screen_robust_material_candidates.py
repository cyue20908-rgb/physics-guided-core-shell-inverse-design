# -*- coding: utf-8 -*-
"""Stage 08B: exact screening of all 400 material sequences for a goal-free task.

Each legal sequence receives 20 deterministic continuous geometry seeds:
16 shared Sobol geometries and four frozen inverse-network predictions obtained
from constant Qsca initialization curves at levels 2, 3, 4 and 5. All 8000
seeds are evaluated directly with double-precision PyMieDiff, including Qabs.
The best 40 material sequences and five seeds per sequence are exported for
Stage 08C. The inverse network is used only to diversify initialization.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping
import time

import numpy as np
import pandas as pd
import torch

from core_shell_optical_design.robust_design.robust_design_core import (
    DEFAULT_DATASET,
    DEFAULT_PGFF_DIR,
    PREPARED_DATA,
    RESULT_ROOT,
    STAGE02_DIR,
    artifact,
    atomic_write_csv,
    evaluate_design_pool,
    load_json,
    load_physics_assets,
    load_prepared_registry,
    normalized_absorption_violation_numpy,
    prepare_output_dir,
    project_robust_geometry_numpy,
    resolve_device,
    save_json_atomic,
    save_npz_atomic,
    sequence_strings,
    set_seed,
    sha256_file,
    sobol_geometries,
    tier_feasible,
    verify_frozen_assets,
    verify_protocol_lock,
)
from core_shell_optical_design.inverse_design.spectrum_conditioned_inverse_network import SpectralMaterialInverseNet

STAGE = "screen_robust_material_candidates"
VERSION = "goal-free-robust-design-2026-07-25-stage08B-v1"
DEFAULT_STAGE08A_DIR = RESULT_ROOT / "absorption_constraint_calibration"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "material_candidate_screening"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--prepared-data", type=Path, default=PREPARED_DATA)
    parser.add_argument("--stage02-dir", type=Path, default=STAGE02_DIR)
    parser.add_argument("--pgff-dir", type=Path, default=DEFAULT_PGFF_DIR)
    parser.add_argument("--stage08a-dir", type=Path, default=DEFAULT_STAGE08A_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--sobol-per-sequence", type=int, default=16)
    parser.add_argument("--inverse-levels", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--selected-sequences", type=int, default=40)
    parser.add_argument("--seeds-per-sequence", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inverse-batch-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_torch_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, dict):
        raise TypeError(f"Expected checkpoint dictionary in {path}.")
    return value


def construct_inverse_model(
    checkpoint: Mapping[str, Any],
    prepared: Mapping[str, np.ndarray],
    device: torch.device,
) -> SpectralMaterialInverseNet:
    if checkpoint.get("model_class") != "SpectralMaterialInverseNet":
        raise RuntimeError("Unexpected Stage-02 inverse model class.")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise RuntimeError("Stage-02 checkpoint has no model_config.")
    model = SpectralMaterialInverseNet(
        n_wavelengths=int(config["n_wavelengths"]),
        n_materials=int(config["n_materials"]),
        qsca_mean=torch.as_tensor(prepared["qsca_mean_train"], dtype=torch.float32),
        qsca_std=torch.as_tensor(prepared["qsca_std_train"], dtype=torch.float32),
        width=int(config["width"]),
        depth=int(config["depth"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def inverse_seed_geometries(
    *,
    model: SpectralMaterialInverseNet,
    sequences: np.ndarray,
    wavelengths: np.ndarray,
    levels: list[float],
    device: torch.device,
    amp: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_geometry: list[np.ndarray] = []
    all_level: list[np.ndarray] = []
    materials_cpu = torch.as_tensor(sequences, dtype=torch.long)
    for level in levels:
        targets_cpu = torch.full(
            (len(sequences), len(wavelengths)), float(level), dtype=torch.float32
        )
        pieces: list[np.ndarray] = []
        for start in range(0, len(sequences), int(batch_size)):
            stop = min(start + int(batch_size), len(sequences))
            with torch.inference_mode(), autocast_context(device, amp):
                output = model(
                    targets_cpu[start:stop].to(device),
                    materials_cpu[start:stop].to(device),
                )["geometry"]
            pieces.append(output.float().cpu().numpy().astype(np.float64))
        geometry = project_robust_geometry_numpy(np.concatenate(pieces, axis=0))
        all_geometry.append(geometry)
        all_level.append(np.full(len(sequences), float(level), dtype=np.float64))
    # level-major -> reshape later by sequence
    return np.concatenate(all_geometry, axis=0), np.concatenate(all_level, axis=0)


def main() -> None:
    args = parse_args()
    if args.sobol_per_sequence < 1 or len(args.inverse_levels) < 1:
        raise ValueError("Sobol and inverse seed counts must be positive.")
    if args.selected_sequences < 1 or args.selected_sequences > 400:
        raise ValueError("--selected-sequences must be in 1..400.")
    if args.seeds_per_sequence < 1:
        raise ValueError("--seeds-per-sequence must be positive.")
    if args.amp and args.device == "cpu":
        raise ValueError("--amp cannot be combined with --device cpu.")

    set_seed(int(args.seed))
    prepare_output_dir(
        args.output_dir,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        summary_name="material_screening_run_summary.json",
    )
    started = time.perf_counter()
    device = resolve_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    physics = load_physics_assets(args.dataset)
    prepared = load_prepared_registry(args.prepared_data, physics)
    provenance = verify_frozen_assets(
        dataset=args.dataset,
        prepared_data=args.prepared_data,
        stage02_dir=args.stage02_dir,
        pgff_dir=args.pgff_dir,
    )
    protocol_path = args.stage08a_dir / "robust_design_protocol_lock.json"
    protocol = verify_protocol_lock(protocol_path)
    if protocol["input_hashes"]["dataset_sha256"] != sha256_file(args.dataset):
        raise RuntimeError("Dataset differs from Stage-08 protocol lock.")
    if protocol["input_hashes"]["prepared_data_sha256"] != sha256_file(args.prepared_data):
        raise RuntimeError("Prepared data differs from Stage-08 protocol lock.")
    tier = dict(protocol["selected_absorption_tier"])
    formal = protocol["formal_search"]
    if int(args.sobol_per_sequence) != int(formal["stage08B_sobol_per_sequence"]):
        raise RuntimeError("Sobol count differs from preregistered Stage-08 protocol.")
    if [float(x) for x in args.inverse_levels] != [float(x) for x in formal["stage08B_inverse_flat_levels"]]:
        raise RuntimeError("Inverse flat levels differ from preregistered Stage-08 protocol.")
    if int(args.selected_sequences) != int(formal["selected_material_sequences"]):
        raise RuntimeError("Selected material-sequence count differs from protocol.")
    if int(args.seeds_per_sequence) != int(formal["selected_seeds_per_sequence"]):
        raise RuntimeError("Seeds-per-sequence differs from protocol.")

    sequences = np.asarray(prepared["legal_material_sequences"], dtype=np.int64)
    names = np.asarray(physics["material_names"]).astype(str)
    sequence_text = sequence_strings(sequences, names)
    inverse_checkpoint = Path(args.stage02_dir) / "best_model.pt"
    checkpoint = load_torch_checkpoint(inverse_checkpoint, device)
    inverse = construct_inverse_model(checkpoint, prepared, device)

    shared_sobol = sobol_geometries(
        int(args.sobol_per_sequence), seed=int(args.seed), skip=4096
    )
    sobol_geometry = np.tile(shared_sobol, (len(sequences), 1))
    sobol_materials = np.repeat(sequences, int(args.sobol_per_sequence), axis=0)
    sobol_sequence_id = np.repeat(np.arange(len(sequences), dtype=np.int64), int(args.sobol_per_sequence))
    sobol_local_id = np.tile(np.arange(int(args.sobol_per_sequence), dtype=np.int64), len(sequences))
    sobol_source = np.asarray([f"sobol_{i:02d}" for i in sobol_local_id], dtype="U32")
    sobol_level = np.full(len(sobol_geometry), np.nan, dtype=np.float64)

    inverse_geometry_level_major, inverse_levels_level_major = inverse_seed_geometries(
        model=inverse,
        sequences=sequences,
        wavelengths=np.asarray(physics["wavelengths"], dtype=np.float64),
        levels=[float(x) for x in args.inverse_levels],
        device=device,
        amp=amp,
        batch_size=int(args.inverse_batch_size),
    )
    # Convert level-major arrays to sequence-major for deterministic 20-seed blocks.
    inverse_geometry = inverse_geometry_level_major.reshape(len(args.inverse_levels), len(sequences), 4).transpose(1, 0, 2).reshape(-1, 4)
    inverse_levels = inverse_levels_level_major.reshape(len(args.inverse_levels), len(sequences)).transpose(1, 0).reshape(-1)
    inverse_materials = np.repeat(sequences, len(args.inverse_levels), axis=0)
    inverse_sequence_id = np.repeat(np.arange(len(sequences), dtype=np.int64), len(args.inverse_levels))
    inverse_source = np.asarray([f"inverse_flat_{value:g}" for value in inverse_levels], dtype="U32")

    geometry_parts: list[np.ndarray] = []
    material_parts: list[np.ndarray] = []
    sequence_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    level_parts: list[np.ndarray] = []
    # Interleave per sequence so the table contract is easy to audit.
    for seq_id in range(len(sequences)):
        srows = np.flatnonzero(sobol_sequence_id == seq_id)
        irows = np.flatnonzero(inverse_sequence_id == seq_id)
        geometry_parts.extend([sobol_geometry[srows], inverse_geometry[irows]])
        material_parts.extend([sobol_materials[srows], inverse_materials[irows]])
        sequence_parts.extend([sobol_sequence_id[srows], inverse_sequence_id[irows]])
        source_parts.extend([sobol_source[srows], inverse_source[irows]])
        level_parts.extend([sobol_level[srows], inverse_levels[irows]])
    geometry = np.concatenate(geometry_parts, axis=0)
    materials = np.concatenate(material_parts, axis=0)
    sequence_id = np.concatenate(sequence_parts, axis=0)
    source = np.concatenate(source_parts, axis=0)
    flat_level = np.concatenate(level_parts, axis=0)
    expected_per_sequence = int(args.sobol_per_sequence) + len(args.inverse_levels)
    if len(geometry) != 400 * expected_per_sequence:
        raise RuntimeError("Unexpected Stage-08B seed pool size.")

    print("Stage 08B exact 400-sequence material screening")
    print(f"Device: {device}; AMP for inverse initialization: {amp}")
    print(f"Candidate designs: {len(geometry)} ({expected_per_sequence} per sequence)")
    print(f"Selected absorption tier: {tier['name']}")
    checkpoint_path = args.output_dir / "checkpoints" / "formal_screen_metrics.npz"
    metrics, _ = evaluate_design_pool(
        geometry=geometry,
        materials=materials,
        physics=physics,
        device=device,
        batch_size=int(args.batch_size),
        singularity_tolerance=float(protocol["numerics"]["solver_singularity_tolerance"]),
        negative_tolerance=float(protocol["numerics"]["negative_efficiency_tolerance"]),
        energy_tolerance=float(protocol["numerics"]["energy_residual_tolerance"]),
        checkpoint_path=checkpoint_path,
        checkpoint_every_batches=10,
        resume=bool(args.resume),
        store_spectra=False,
        print_label="Stage 08B PyMieDiff screen",
    )
    frame = pd.DataFrame(
        {
            "candidate_index": np.arange(len(geometry), dtype=np.int64),
            "sequence_id": sequence_id,
            "seed_source": source,
            "inverse_flat_level": flat_level,
            "m0": materials[:, 0],
            "m1": materials[:, 1],
            "m2": materials[:, 2],
            "m3": materials[:, 3],
            "materials": sequence_text[sequence_id],
            "r_core": geometry[:, 0],
            "t1": geometry[:, 1],
            "t2": geometry[:, 2],
            "t3": geometry[:, 3],
            "outer_radius": geometry.sum(axis=1),
        }
    )
    frame = pd.concat([frame, metrics.drop(columns=["row_index"])], axis=1)
    frame["nominal_absorption_feasible"] = (
        tier_feasible(frame, tier) & frame["solver_valid"].astype(bool).to_numpy() & frame["physics_valid"].astype(bool).to_numpy()
    )
    frame["normalized_absorption_violation"] = normalized_absorption_violation_numpy(frame, tier)

    ranked_parts: list[pd.DataFrame] = []
    sequence_rows: list[Dict[str, Any]] = []
    for seq_id, group in frame.groupby("sequence_id", sort=True):
        group = group.copy()
        group = group.sort_values(
            [
                "nominal_absorption_feasible",
                "normalized_absorption_violation",
                "tail_mean_qsca",
                "mean_qsca",
                "qsca_cv",
                "candidate_index",
            ],
            ascending=[False, True, False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        group["rank_within_sequence"] = np.arange(1, len(group) + 1, dtype=np.int64)
        ranked_parts.append(group)
        best = group.iloc[0]
        sequence_rows.append(
            {
                "sequence_id": int(seq_id),
                "materials": str(best["materials"]),
                "best_candidate_index": int(best["candidate_index"]),
                "best_nominal_absorption_feasible": bool(best["nominal_absorption_feasible"]),
                "best_normalized_absorption_violation": float(best["normalized_absorption_violation"]),
                "best_tail_mean_qsca": float(best["tail_mean_qsca"]),
                "best_mean_qsca": float(best["mean_qsca"]),
                "best_qsca_cv": float(best["qsca_cv"]),
                "best_band_absorption_fraction": float(best["band_absorption_fraction"]),
                "best_absorption_fraction_p95": float(best["absorption_fraction_p95"]),
                "best_max_qabs": float(best["max_qabs"]),
            }
        )
    ranked = pd.concat(ranked_parts, ignore_index=True)
    sequence_summary = pd.DataFrame(sequence_rows).sort_values(
        [
            "best_nominal_absorption_feasible",
            "best_normalized_absorption_violation",
            "best_tail_mean_qsca",
            "best_mean_qsca",
            "best_qsca_cv",
            "sequence_id",
        ],
        ascending=[False, True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    sequence_summary["material_screen_rank"] = np.arange(1, len(sequence_summary) + 1, dtype=np.int64)
    selected_sequence_ids = sequence_summary.head(int(args.selected_sequences))["sequence_id"].to_numpy(dtype=np.int64)
    selected = ranked.loc[
        ranked["sequence_id"].isin(selected_sequence_ids)
        & (ranked["rank_within_sequence"] <= int(args.seeds_per_sequence))
    ].copy()
    rank_map = sequence_summary.set_index("sequence_id")["material_screen_rank"].to_dict()
    selected["material_screen_rank"] = selected["sequence_id"].map(rank_map).astype(np.int64)
    selected = selected.sort_values(
        ["material_screen_rank", "rank_within_sequence"], kind="mergesort"
    ).reset_index(drop=True)
    selected["selected_seed_index"] = np.arange(len(selected), dtype=np.int64)
    if len(selected) != int(args.selected_sequences) * int(args.seeds_per_sequence):
        raise RuntimeError("Unexpected number of selected continuous-refinement seeds.")

    all_csv = args.output_dir / "initial_candidate_metrics.csv"
    sequence_csv = args.output_dir / "material_sequence_screen_summary.csv"
    selected_csv = args.output_dir / "selected_continuous_refinement_seeds.csv"
    atomic_write_csv(all_csv, ranked.sort_values(["sequence_id", "rank_within_sequence"]))
    atomic_write_csv(sequence_csv, sequence_summary)
    atomic_write_csv(selected_csv, selected)
    selected_npz = args.output_dir / "selected_continuous_refinement_seeds.npz"
    save_npz_atomic(
        selected_npz,
        wavelengths_nm=np.asarray(physics["wavelengths"], dtype=np.float64),
        selected_seed_index=selected["selected_seed_index"].to_numpy(dtype=np.int64),
        material_screen_rank=selected["material_screen_rank"].to_numpy(dtype=np.int64),
        rank_within_sequence=selected["rank_within_sequence"].to_numpy(dtype=np.int64),
        sequence_id=selected["sequence_id"].to_numpy(dtype=np.int64),
        material_indices=selected[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64),
        geometry_nm=selected[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.float64),
        seed_source=selected["seed_source"].to_numpy(dtype="U32"),
        nominal_tail_mean_qsca=selected["tail_mean_qsca"].to_numpy(dtype=np.float64),
        nominal_mean_qsca=selected["mean_qsca"].to_numpy(dtype=np.float64),
        nominal_qsca_cv=selected["qsca_cv"].to_numpy(dtype=np.float64),
        nominal_band_absorption_fraction=selected["band_absorption_fraction"].to_numpy(dtype=np.float64),
        nominal_absorption_fraction_p95=selected["absorption_fraction_p95"].to_numpy(dtype=np.float64),
        nominal_max_qabs=selected["max_qabs"].to_numpy(dtype=np.float64),
        protocol_sha256=np.asarray(protocol["protocol_sha256"]),
    )

    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "amp_inverse_initialization": amp,
        "selected_absorption_tier": tier,
        "counts": {
            "legal_material_sequences": 400,
            "seeds_per_sequence": expected_per_sequence,
            "all_exact_candidates": int(len(frame)),
            "nominal_feasible_candidates": int(frame["nominal_absorption_feasible"].sum()),
            "selected_sequences": int(len(selected_sequence_ids)),
            "selected_stage08c_seeds": int(len(selected)),
        },
        "policy": {
            "all_400_sequences_enumerated": True,
            "pymiediff_used_for_all_screen_candidates": True,
            "qabs_used_for_hard_feasibility": True,
            "inverse_network_role": "initialization only",
            "pgff_used": False,
            "stage03_to_stage06_candidate_results_read": False,
        },
        "inputs": {
            **provenance,
            "protocol_lock": artifact(protocol_path),
        },
        "outputs": {
            "all_candidates": artifact(all_csv),
            "sequence_summary": artifact(sequence_csv),
            "selected_seeds_csv": artifact(selected_csv),
            "selected_seeds_npz": artifact(selected_npz),
        },
    }
    save_json_atomic(args.output_dir / "material_screening_run_summary.json", summary)

    print("Stage 08B exhaustive material screen PASSED")
    print(f"Exact candidates: {len(frame)}")
    print(f"Selected material sequences: {len(selected_sequence_ids)}")
    print(f"Selected Stage 08C seeds: {len(selected)}")
    print("PGFF was not used for ranking; all screening metrics are PyMieDiff metrics.")


if __name__ == "__main__":
    main()
