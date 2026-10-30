# -*- coding: utf-8 -*-
"""Audit the frozen forward models and verify both gradient paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time

import numpy as np
import torch

from core_shell_optical_design.inverse_design.inverse_design_data import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_DATASET,
    DEFAULT_PGFF_DIR,
    DEFAULT_PGFF_MODEL_SOURCE,
    DEFAULT_RESULT_ROOT,
    artifact,
    load_forward_dataset,
    load_split_bundle,
    resolve_device,
    save_json_atomic,
    set_seed,
)
from core_shell_optical_design.inverse_design.forward_model_bridge import load_frozen_pgff, pgff_forward, pymiediff_forward
from core_shell_optical_design.inverse_design.geometry_parameterization import (
    enumerate_legal_material_sequences,
    geometry_bounds_mask,
    material_sequence_mask,
    self_test as geometry_self_test,
)

STAGE = "environment_preflight"
VERSION = "tandem-mie-inverse-2026-07-24-step01-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen forward assets and differentiability.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--pgff-dir", type=Path, default=DEFAULT_PGFF_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pymiediff-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--skip-pymiediff-gradient", action="store_true")
    parser.add_argument("--energy-tolerance", type=float, default=1.0e-8)
    return parser.parse_args()


def sequence_set(values: np.ndarray) -> set[tuple[int, int, int, int]]:
    return {tuple(int(x) for x in row) for row in np.asarray(values, dtype=np.int64)}


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    set_seed(args.seed)
    geometry_self_test()

    required = [
        args.dataset,
        args.baseline_dir / "idx_train.npy",
        args.baseline_dir / "idx_val.npy",
        args.baseline_dir / "idx_test.npy",
        args.baseline_dir / "normalization.npz",
        args.pgff_dir / "best_model.pt",
        args.pgff_dir / "hyperparameters.json",
        args.pgff_dir / "physics_normalization.npz",
        DEFAULT_PGFF_MODEL_SOURCE,
    ]
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Required frozen forward assets are missing:\n" + "\n".join(str(p) for p in missing))

    dataset = load_forward_dataset(args.dataset)
    split = load_split_bundle(dataset, args.baseline_dir)
    legal_sequences = enumerate_legal_material_sequences(dataset["material_names"])
    legal_array = legal_sequences[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64)

    all_set = sequence_set(dataset["material_indices"])
    train_set = sequence_set(dataset["material_indices"][split.train])
    val_set = sequence_set(dataset["material_indices"][split.validation])
    test_set = sequence_set(dataset["material_indices"][split.test])
    legal_set = sequence_set(legal_array)
    if all_set != legal_set:
        raise RuntimeError(
            f"Dataset sequence registry ({len(all_set)}) differs from exact legal registry ({len(legal_set)})."
        )

    device = resolve_device(args.device)
    pgff = load_frozen_pgff(dataset, args.baseline_dir, args.pgff_dir, device)
    sample_index = int(split.train[0])
    geometry = torch.as_tensor(
        dataset["geometry"][sample_index : sample_index + 1],
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)
    materials = torch.as_tensor(
        dataset["material_indices"][sample_index : sample_index + 1],
        dtype=torch.long,
        device=device,
    )
    prediction = pgff_forward(pgff, geometry, materials)
    pgff_loss = prediction["q_sca"].square().mean()
    pgff_loss.backward()
    pgff_gradient = geometry.grad.detach().cpu().numpy()
    if not np.isfinite(pgff_gradient).all() or float(np.linalg.norm(pgff_gradient)) <= 0.0:
        raise RuntimeError("Frozen PGFF does not provide a finite non-zero geometry gradient.")
    trainable_pgff_parameters = sum(int(p.requires_grad) for p in pgff.parameters())
    if trainable_pgff_parameters != 0:
        raise RuntimeError("PGFF parameters are not completely frozen.")

    pymiediff_report = {"checked": False}
    if not args.skip_pymiediff_gradient:
        mie_device = resolve_device(args.pymiediff_device)
        mie_geometry = torch.as_tensor(
            dataset["geometry"][sample_index : sample_index + 1],
            dtype=torch.float64,
            device=mie_device,
        ).requires_grad_(True)
        mie_materials = materials.to(mie_device)
        mie = pymiediff_forward(mie_geometry, mie_materials, dataset)
        residual = torch.max(torch.abs(mie["q_ext"] - mie["q_sca"] - mie["q_abs"]))
        if float(residual.detach().cpu()) > float(args.energy_tolerance):
            raise RuntimeError(
                f"PyMieDiff energy residual {float(residual):.3e} exceeds {args.energy_tolerance:.3e}."
            )
        mie_loss = mie["q_sca"].square().mean()
        mie_loss.backward()
        mie_gradient = mie_geometry.grad.detach().cpu().numpy()
        if not np.isfinite(mie_gradient).all() or float(np.linalg.norm(mie_gradient)) <= 0.0:
            raise RuntimeError("PyMieDiff does not provide a finite non-zero geometry gradient.")
        pymiediff_report = {
            "checked": True,
            "device": str(mie_device),
            "qsca_shape": list(mie["q_sca"].shape),
            "max_energy_residual": float(residual.detach().cpu()),
            "geometry_gradient": mie_gradient.tolist(),
            "gradient_l2_norm": float(np.linalg.norm(mie_gradient)),
        }

    output_dir = args.result_root / "environment_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": float(time.perf_counter() - start),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "strict_policy": {
            "legacy_inverse_results_read": False,
            "forward_assets_frozen": True,
            "test_split_used_for_training_or_normalization": False,
        },
        "dataset": {
            **artifact(args.dataset),
            "samples": int(len(dataset["geometry"])),
            "wavelengths": int(len(dataset["wavelengths"])),
            "wavelength_min_nm": float(dataset["wavelengths"].min()),
            "wavelength_max_nm": float(dataset["wavelengths"].max()),
            "materials": dataset["material_names"].tolist(),
            "max_energy_residual": float(dataset["max_energy_residual"]),
            "all_geometries_legal": bool(geometry_bounds_mask(dataset["geometry"]).all()),
            "all_material_sequences_legal": bool(
                material_sequence_mask(dataset["material_indices"], dataset["material_names"]).all()
            ),
        },
        "splits": {
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
            "legal_sequence_count": int(len(legal_set)),
            "sequence_coverage_train": int(len(train_set)),
            "sequence_coverage_validation": int(len(val_set)),
            "sequence_coverage_test": int(len(test_set)),
            "missing_train_sequences": [list(x) for x in sorted(legal_set - train_set)],
        },
        "pgff": {
            "checkpoint": artifact(args.pgff_dir / "best_model.pt"),
            "parameter_count": int(sum(p.numel() for p in pgff.parameters())),
            "trainable_parameter_tensor_count": int(trainable_pgff_parameters),
            "qsca_shape": list(prediction["q_sca"].shape),
            "geometry_gradient": pgff_gradient.tolist(),
            "gradient_l2_norm": float(np.linalg.norm(pgff_gradient)),
        },
        "pymiediff": pymiediff_report,
    }
    save_json_atomic(output_dir / "preflight_report.json", report)
    print("Stage 00 preflight PASSED")
    print(f"Dataset samples: {len(dataset['geometry'])}")
    print(f"Splits: train={len(split.train)}, val={len(split.validation)}, test={len(split.test)}")
    print(f"Legal material sequences: {len(legal_sequences)}")
    print(f"Training sequence coverage: {len(train_set)}/{len(legal_set)}")
    print(f"Frozen PGFF geometry-gradient norm: {np.linalg.norm(pgff_gradient):.6e}")
    if pymiediff_report["checked"]:
        print(f"PyMieDiff geometry-gradient norm: {pymiediff_report['gradient_l2_norm']:.6e}")
    print(f"Report: {output_dir / 'preflight_report.json'}")


if __name__ == "__main__":
    main()
