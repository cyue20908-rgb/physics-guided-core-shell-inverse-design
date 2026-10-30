# -*- coding: utf-8 -*-
"""Stage 01: freeze inverse-network preprocessing using the training split only."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from core_shell_optical_design.inverse_design.inverse_design_data import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_DATASET,
    DEFAULT_RESULT_ROOT,
    artifact,
    load_forward_dataset,
    load_split_bundle,
    save_json_atomic,
    save_npz_atomic,
    set_seed,
)
from core_shell_optical_design.inverse_design.geometry_parameterization import enumerate_legal_material_sequences

STAGE = "prepare_inverse_design_data"
VERSION = "tandem-mie-inverse-2026-07-24-step01-v1"
STD_FLOOR = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze tandem inverse preprocessing.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--allow-missing-train-sequences", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sequence_tuples(values: np.ndarray) -> set[tuple[int, int, int, int]]:
    return {tuple(int(x) for x in row) for row in np.asarray(values, dtype=np.int64)}


def split_rows(name: str, indices: np.ndarray, dataset) -> dict:
    qsca = np.asarray(dataset["Q_sca"][indices], dtype=np.float64)
    geometry = np.asarray(dataset["geometry"][indices], dtype=np.float64)
    sequences = sequence_tuples(dataset["material_indices"][indices])
    return {
        "split": name,
        "sample_count": int(len(indices)),
        "material_sequence_count": int(len(sequences)),
        "mean_qsca": float(qsca.mean()),
        "std_qsca": float(qsca.std()),
        "minimum_qsca": float(qsca.min()),
        "maximum_qsca": float(qsca.max()),
        "mean_outer_radius_nm": float(geometry.sum(axis=1).mean()),
        "minimum_outer_radius_nm": float(geometry.sum(axis=1).min()),
        "maximum_outer_radius_nm": float(geometry.sum(axis=1).max()),
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    set_seed(args.seed)
    output_dir = args.result_root / "inverse_training_data"
    manifest_path = output_dir / "inverse_design_training_data.npz"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Prepared data already exist: {manifest_path}. Use --overwrite for a deliberate rebuild."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = args.result_root / "environment_preflight" / "preflight_report.json"
    if not preflight.is_file():
        raise FileNotFoundError(
            f"Run Stage 00 successfully before Stage 01. Missing: {preflight}"
        )

    dataset = load_forward_dataset(args.dataset)
    split = load_split_bundle(dataset, args.baseline_dir)
    legal = enumerate_legal_material_sequences(dataset["material_names"])
    legal_array = legal[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64)
    legal_set = sequence_tuples(legal_array)
    train_set = sequence_tuples(dataset["material_indices"][split.train])
    missing_train = sorted(legal_set - train_set)
    if missing_train and not args.allow_missing_train_sequences:
        raise RuntimeError(
            f"Training split is missing {len(missing_train)} legal material sequences. "
            "The blind enumerator would query material conditions unseen during tandem training."
        )

    # The inverse-network input normalization is computed from TRAIN ONLY.
    train_qsca = np.asarray(dataset["Q_sca"][split.train], dtype=np.float64)
    qsca_mean = train_qsca.mean(axis=0)
    qsca_std_raw = train_qsca.std(axis=0)
    qsca_std = np.maximum(qsca_std_raw, STD_FLOOR)

    save_npz_atomic(
        manifest_path,
        train_indices=split.train,
        validation_indices=split.validation,
        test_indices=split.test,
        wavelengths_nm=np.asarray(dataset["wavelengths"], dtype=np.float64),
        qsca_mean_train=qsca_mean.astype(np.float32),
        qsca_std_train=qsca_std.astype(np.float32),
        qsca_std_raw_train=qsca_std_raw.astype(np.float32),
        material_names=np.asarray(dataset["material_names"], dtype=str),
        legal_material_sequences=legal_array,
        geometry_lower_nm=np.asarray([120.0, 20.0, 20.0, 20.0], dtype=np.float64),
        geometry_upper_nm=np.asarray([220.0, 80.0, 80.0, 80.0], dtype=np.float64),
        maximum_outer_radius_nm=np.asarray(300.0, dtype=np.float64),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    legal.to_csv(output_dir / "legal_material_sequences.csv", index=False, encoding="utf-8-sig")

    statistics = pd.DataFrame(
        [
            split_rows("train", split.train, dataset),
            split_rows("validation", split.validation, dataset),
            split_rows("test", split.test, dataset),
        ]
    )
    statistics.to_csv(output_dir / "split_statistics.csv", index=False, encoding="utf-8-sig")

    report = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": float(time.perf_counter() - start),
        "data_policy": {
            "inverse_network_training_split": "train only",
            "model_selection_split": "validation only",
            "test_split": "untouched until final trained-network evaluation",
            "blind_targets": "not read by this stage",
            "qsca_normalization_source": "training split only",
            "legacy_inverse_results_read": False,
        },
        "dataset": artifact(args.dataset),
        "baseline_artifacts": {
            "idx_train": artifact(args.baseline_dir / "idx_train.npy"),
            "idx_val": artifact(args.baseline_dir / "idx_val.npy"),
            "idx_test": artifact(args.baseline_dir / "idx_test.npy"),
            "normalization": artifact(args.baseline_dir / "normalization.npz"),
        },
        "outputs": {
            "prepared_data": artifact(manifest_path),
            "legal_material_sequences": artifact(output_dir / "legal_material_sequences.csv"),
            "split_statistics": artifact(output_dir / "split_statistics.csv"),
        },
        "dimensions": {
            "n_samples": int(len(dataset["geometry"])),
            "n_wavelengths": int(len(dataset["wavelengths"])),
            "n_materials": int(len(dataset["material_names"])),
            "n_legal_material_sequences": int(len(legal)),
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "normalization": {
            "std_floor": STD_FLOOR,
            "floored_wavelength_count": int(np.sum(qsca_std_raw < STD_FLOOR)),
            "minimum_saved_std": float(qsca_std.min()),
            "maximum_saved_std": float(qsca_std.max()),
        },
        "material_coverage": {
            "training_sequence_count": int(len(train_set)),
            "missing_training_sequences": [list(x) for x in missing_train],
        },
    }
    save_json_atomic(output_dir / "preparation_report.json", report)
    print("Stage 01 preparation PASSED")
    print(f"Prepared data: {manifest_path}")
    print(f"Legal material sequences: {len(legal)}")
    print(f"Training material-sequence coverage: {len(train_set)}/{len(legal_set)}")
    print("Qsca input normalization source: training split only")


if __name__ == "__main__":
    main()
