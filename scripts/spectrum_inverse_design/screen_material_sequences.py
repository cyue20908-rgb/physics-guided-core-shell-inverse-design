# -*- coding: utf-8 -*-
"""Stage 03: strict blind inference and exact screening of 400 materials.

This stage performs no optimization and no PyMieDiff evaluation.  For every
public Qsca-only blind target, it:

1. loads the frozen inverse network selected in Stage 02;
2. enumerates every legal four-layer material sequence from Stage 01;
3. predicts one strictly legal continuous geometry for each sequence;
4. reconstructs Qsca with the frozen PGFF surrogate;
5. ranks all 400 sequence-conditioned candidates using Qsca NMAE only;
6. freezes the top-K candidates for the next PyMieDiff geometry-refinement stage.

Blindness policy
----------------
Only ``task_id``, ``wavelengths_nm`` and ``target_qsca`` are consumed from the
public target NPZ.  Reference materials, reference geometry, Qabs and Qext are
forbidden.  The forward test split is not loaded for evaluation, and no legacy
inverse candidate/result file is consumed.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch

from core_shell_optical_design.inverse_design.inverse_design_data import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_DATASET,
    DEFAULT_PGFF_DIR,
    DEFAULT_RESULT_ROOT,
    artifact,
    load_forward_dataset,
    resolve_device,
    save_json_atomic,
    save_npz_atomic,
    set_seed,
    sha256_file,
)
from core_shell_optical_design.inverse_design.forward_model_bridge import load_frozen_pgff, pgff_forward
from core_shell_optical_design.inverse_design.geometry_parameterization import (
    GEOMETRY_CHECK_ATOL_NM,
    MAX_OUTER_RADIUS,
    geometry_bounds_mask,
    material_sequence_mask,
)
from core_shell_optical_design.inverse_design.spectrum_conditioned_inverse_network import SpectralMaterialInverseNet
from core_shell_optical_design.inverse_design.inverse_design_objectives import per_sample_metrics

STAGE = "material_sequence_screening"
VERSION = "tandem-mie-inverse-2026-07-24-stage03-v1"

DEFAULT_STAGE02_DIR = DEFAULT_RESULT_ROOT / "inverse_model_training"
DEFAULT_PREPARED_DATA = DEFAULT_RESULT_ROOT / "inverse_training_data" / "inverse_design_training_data.npz"
DEFAULT_PUBLIC_TARGETS = (
    DEFAULT_RESULT_ROOT / "blind_targets_public" / "blind_target_spectra.npz"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "material_sequence_screening"

# These keys would expose the hidden generating structure or non-Qsca spectra.
FORBIDDEN_TARGET_KEYS = {
    "reference_geometry",
    "reference_materials",
    "reference_material_indices",
    "reference_radii",
    "reference_qsca",
    "reference_qabs",
    "reference_qext",
    "material_indices",
    "materials",
    "radii",
    "geometry",
    "r_core",
    "t1",
    "t2",
    "t3",
    "m0",
    "m1",
    "m2",
    "m3",
    "target_qabs",
    "target_qext",
    "q_abs",
    "q_ext",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict blind Qsca inference with all 400 legal material sequences"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--pgff-dir", type=Path, default=DEFAULT_PGFF_DIR)
    parser.add_argument("--prepared-data", type=Path, default=DEFAULT_PREPARED_DATA)
    parser.add_argument("--stage02-dir", type=Path, default=DEFAULT_STAGE02_DIR)
    parser.add_argument("--public-targets", type=Path, default=DEFAULT_PUBLIC_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for inference.")
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expected-target-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 1 <= args.top_k <= 400:
        raise ValueError("--top-k must be between 1 and 400.")
    if args.expected_target_count < 0:
        raise ValueError("--expected-target-count cannot be negative.")
    if args.amp and args.device == "cpu":
        raise ValueError("--amp cannot be combined with --device cpu.")


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def load_torch_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, dict):
        raise TypeError(f"Expected checkpoint dictionary in {path}.")
    return value


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Stage 03 output directory is not empty: {path}\n"
                "Use --overwrite only when intentionally replacing this screen."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalized_key_set(keys: Iterable[str]) -> set[str]:
    return {str(key).strip().lower() for key in keys}


def spectrum_hash(wavelengths: np.ndarray, qsca: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(wavelengths, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(qsca, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def candidate_hash(
    target_hash: str,
    sequence: np.ndarray,
    geometry: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(target_hash).encode("utf-8"))
    digest.update(np.asarray(sequence, dtype="<i8").tobytes(order="C"))
    digest.update(np.round(np.asarray(geometry, dtype="<f8"), 7).tobytes(order="C"))
    return digest.hexdigest()[:24]


def load_public_targets(
    path: Path,
    dataset_wavelengths: np.ndarray,
    expected_count: int,
) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Public blind target file not found: {path}\n"
            "Provide --public-targets or place the Qsca-only NPZ at the default path."
        )
    if "private" in {part.strip().lower() for part in path.parts}:
        raise RuntimeError("Stage 03 refuses to read a path containing a 'private' directory.")
    if path.suffix.lower() != ".npz":
        raise ValueError("Stage 03 currently requires one combined public-target .npz file.")

    with np.load(path, allow_pickle=False) as data:
        keys = normalized_key_set(data.files)
        exposed = sorted(keys & FORBIDDEN_TARGET_KEYS)
        if exposed:
            raise RuntimeError(
                "Public target file exposes forbidden blind-reference fields: "
                + ", ".join(exposed)
            )
        required = {"task_id", "wavelengths_nm", "target_qsca"}
        missing = sorted(required - keys)
        if missing:
            raise KeyError(f"Public target NPZ is missing: {missing}")
        task_ids = np.asarray(data["task_id"]).astype(str)
        wavelengths = np.asarray(data["wavelengths_nm"], dtype=np.float64)
        qsca = np.asarray(data["target_qsca"], dtype=np.float64)

    if task_ids.ndim != 1:
        raise ValueError("task_id must be a one-dimensional array.")
    if wavelengths.ndim != 1 or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("wavelengths_nm must be a strictly increasing one-dimensional grid.")
    if qsca.shape != (len(task_ids), len(wavelengths)):
        raise ValueError(
            f"target_qsca shape {qsca.shape} does not match "
            f"({len(task_ids)}, {len(wavelengths)})."
        )
    if expected_count and len(task_ids) != int(expected_count):
        raise RuntimeError(
            f"Expected {expected_count} blind targets, found {len(task_ids)}."
        )
    if len(set(task_ids.tolist())) != len(task_ids):
        raise RuntimeError("Blind task IDs are not unique.")
    if not np.isfinite(qsca).all():
        raise ValueError("target_qsca contains NaN or Inf.")
    if float(qsca.min()) < -1.0e-8:
        raise ValueError("target_qsca contains a negative value below tolerance.")
    if not np.allclose(
        wavelengths,
        np.asarray(dataset_wavelengths, dtype=np.float64),
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise RuntimeError("Blind-target and forward-dataset wavelength grids differ.")
    if np.any(qsca.max(axis=1) <= 1.0e-8):
        raise ValueError("Every target spectrum must have a positive peak.")

    hashes = np.asarray(
        [spectrum_hash(wavelengths, row) for row in qsca], dtype="U64"
    )
    return {
        "task_id": task_ids.astype("U64"),
        "wavelengths_nm": wavelengths,
        "target_qsca": qsca,
        "target_hash": hashes,
        "source_keys": np.asarray(sorted(keys), dtype="U64"),
    }


def load_prepared_data(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "wavelengths_nm",
        "qsca_mean_train",
        "qsca_std_train",
        "material_names",
        "legal_material_sequences",
        "test_indices",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"Prepared data is missing: {missing}")
        output = {key: np.asarray(data[key]).copy() for key in data.files}
    return output


def verify_stage02(
    stage02_dir: Path,
    prepared_data_path: Path,
    pgff_checkpoint_path: Path,
) -> Tuple[Dict[str, Any], Path]:
    stage02_dir = Path(stage02_dir)
    summary_path = stage02_dir / "training_summary.json"
    best_path = stage02_dir / "best_model.pt"
    summary = load_json(summary_path)
    if summary.get("status") != "passed":
        raise RuntimeError("Stage 02 training_summary.json is not marked as passed.")
    policy = summary.get("data_policy", {})
    if policy.get("test_split_evaluated") is not False:
        raise RuntimeError("Stage 02 policy does not certify an untouched test split.")
    if policy.get("blind_targets_read") is not False:
        raise RuntimeError("Stage 02 policy does not certify blind-target isolation.")
    if not bool(summary.get("optimization", {}).get("pgff_parameters_frozen", False)):
        raise RuntimeError("Stage 02 summary does not certify frozen PGFF parameters.")
    if not best_path.is_file():
        raise FileNotFoundError(best_path)

    outputs = summary.get("outputs", {})
    saved_best_hash = outputs.get("best_model", {}).get("sha256")
    if saved_best_hash and sha256_file(best_path) != saved_best_hash:
        raise RuntimeError("Stage 02 best_model.pt SHA256 differs from training summary.")
    inputs = summary.get("inputs", {})
    saved_prepared_hash = inputs.get("prepared_data", {}).get("sha256")
    if saved_prepared_hash and sha256_file(prepared_data_path) != saved_prepared_hash:
        raise RuntimeError("inverse_design_training_data.npz differs from the Stage 02 input artifact.")
    saved_pgff_hash = inputs.get("pgff_checkpoint", {}).get("sha256")
    if saved_pgff_hash and sha256_file(pgff_checkpoint_path) != saved_pgff_hash:
        raise RuntimeError("PGFF best_model.pt differs from the Stage 02 input artifact.")
    return summary, best_path


def construct_inverse_model(
    checkpoint: Mapping[str, Any],
    prepared: Mapping[str, np.ndarray],
    device: torch.device,
) -> SpectralMaterialInverseNet:
    if checkpoint.get("model_class") != "SpectralMaterialInverseNet":
        raise RuntimeError("Unexpected Stage 02 inverse model class.")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise RuntimeError("Stage 02 checkpoint has no model_config dictionary.")
    required = {"n_wavelengths", "n_materials", "width", "depth", "dropout"}
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Stage 02 model_config is missing: {missing}")
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


def rank_one_target(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        by=["pgff_nmae", "pgff_nrmse", "pgff_peak_error_nm", "sequence_id"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "screen_rank", np.arange(1, len(ranked) + 1, dtype=np.int64))
    return ranked


def plot_target_screen(
    output_dir: Path,
    task_id: str,
    wavelengths: np.ndarray,
    target_qsca: np.ndarray,
    ranked: pd.DataFrame,
    qsca_by_sequence: np.ndarray,
    top_count: int = 5,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(wavelengths, target_qsca, linewidth=2.4, label="Target Qsca")
    for _, row in ranked.head(int(top_count)).iterrows():
        seq = int(row["sequence_id"])
        ax.plot(
            wavelengths,
            qsca_by_sequence[seq],
            linewidth=1.2,
            label=f"Rank {int(row['screen_rank'])}: {row['materials']}",
        )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Scattering efficiency, Qsca")
    ax.set_title(str(task_id))
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(task_id))
    fig.savefig(figure_dir / f"{safe_id}_pgff_screen.png", dpi=600, bbox_inches="tight")
    fig.savefig(figure_dir / f"{safe_id}_pgff_screen.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    start_time = time.perf_counter()
    set_seed(args.seed)
    prepare_output_dir(args.output_dir, args.overwrite)

    dataset = load_forward_dataset(args.dataset)
    prepared = load_prepared_data(args.prepared_data)
    if not np.array_equal(
        np.asarray(prepared["wavelengths_nm"], dtype=np.float64),
        np.asarray(dataset["wavelengths"], dtype=np.float64),
    ):
        raise RuntimeError("Prepared-data and forward-dataset wavelength grids differ.")
    if not np.array_equal(
        np.asarray(prepared["material_names"]).astype(str),
        np.asarray(dataset["material_names"]).astype(str),
    ):
        raise RuntimeError("Prepared-data and forward-dataset material registries differ.")

    legal_sequences = np.asarray(prepared["legal_material_sequences"], dtype=np.int64)
    if legal_sequences.shape != (400, 4):
        raise RuntimeError(f"Expected exactly (400,4) legal material sequences, got {legal_sequences.shape}.")
    if len(np.unique(legal_sequences, axis=0)) != 400:
        raise RuntimeError("Prepared material registry contains duplicate sequences.")
    if not np.all(material_sequence_mask(legal_sequences, dataset["material_names"])):
        raise RuntimeError("Prepared material registry contains an illegal sequence.")

    targets = load_public_targets(
        args.public_targets,
        dataset_wavelengths=np.asarray(dataset["wavelengths"], dtype=np.float64),
        expected_count=int(args.expected_target_count),
    )
    stage02_summary, best_model_path = verify_stage02(
        args.stage02_dir,
        args.prepared_data,
        args.pgff_dir / "best_model.pt",
    )

    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    checkpoint = load_torch_checkpoint(best_model_path, device)
    inverse = construct_inverse_model(checkpoint, prepared, device)
    pgff = load_frozen_pgff(dataset, args.baseline_dir, args.pgff_dir, device)
    if any(parameter.requires_grad for parameter in inverse.parameters()):
        raise RuntimeError("Stage 03 inverse network must be frozen.")
    if any(parameter.requires_grad for parameter in pgff.parameters()):
        raise RuntimeError("Stage 03 PGFF must be frozen.")

    wavelengths = np.asarray(targets["wavelengths_nm"], dtype=np.float64)
    wavelengths_tensor = torch.as_tensor(wavelengths, dtype=torch.float32, device=device)
    material_names = np.asarray(dataset["material_names"]).astype(str)
    material_strings = np.asarray(
        ["|".join(material_names[row].tolist()) for row in legal_sequences], dtype="U128"
    )

    all_frames: list[pd.DataFrame] = []
    all_pgff_qsca: list[np.ndarray] = []
    all_geometry: list[np.ndarray] = []
    all_task_index: list[np.ndarray] = []
    summary_rows: list[Dict[str, Any]] = []

    print("Stage 03 strict blind material screen")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Blind targets: {len(targets['task_id'])}; legal material sequences: {len(legal_sequences)}")
    print(f"Selection: top {args.top_k} PGFF-ranked sequences per target")

    materials_cpu = torch.as_tensor(legal_sequences, dtype=torch.long)
    for target_index, task_id in enumerate(targets["task_id"]):
        target_row = np.asarray(targets["target_qsca"][target_index], dtype=np.float32)
        target_tensor_cpu = torch.as_tensor(target_row, dtype=torch.float32).repeat(
            len(legal_sequences), 1
        )
        geometries: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []

        for begin in range(0, len(legal_sequences), int(args.batch_size)):
            end = min(begin + int(args.batch_size), len(legal_sequences))
            q_batch = target_tensor_cpu[begin:end].to(device=device, non_blocking=True)
            m_batch = materials_cpu[begin:end].to(device=device, non_blocking=True)
            with torch.inference_mode(), autocast_context(device, amp_enabled):
                geometry = inverse(q_batch, m_batch)["geometry"]
                qsca_pred = pgff_forward(pgff, geometry, m_batch)["q_sca"]
            geometries.append(geometry.float().cpu())
            predictions.append(qsca_pred.float().cpu())

        geometry_np = torch.cat(geometries, dim=0).numpy().astype(np.float64)
        qsca_np = torch.cat(predictions, dim=0).numpy().astype(np.float64)
        legal_mask = geometry_bounds_mask(geometry_np, atol=GEOMETRY_CHECK_ATOL_NM)
        if not np.all(legal_mask):
            bad = np.flatnonzero(~legal_mask)
            raise RuntimeError(
                f"Inverse network generated {len(bad)} illegal geometries for {task_id}; "
                f"first sequence IDs={bad[:10].tolist()}."
            )

        target_repeated = torch.as_tensor(target_row).repeat(len(legal_sequences), 1)
        metrics = per_sample_metrics(
            torch.as_tensor(qsca_np, dtype=torch.float32),
            target_repeated,
            torch.as_tensor(wavelengths, dtype=torch.float32),
        )
        nmae = metrics["nmae"].numpy().astype(np.float64)
        nrmse = metrics["nrmse"].numpy().astype(np.float64)
        r2 = metrics["r2"].numpy().astype(np.float64)
        peak_error = metrics["peak_error_nm"].numpy().astype(np.float64)
        outer = geometry_np.sum(axis=1)
        pred_peak_nm = wavelengths[np.argmax(qsca_np, axis=1)]
        target_hash_value = str(targets["target_hash"][target_index])
        candidate_ids = np.asarray(
            [
                candidate_hash(target_hash_value, legal_sequences[i], geometry_np[i])
                for i in range(len(legal_sequences))
            ],
            dtype="U24",
        )

        frame = pd.DataFrame(
            {
                "task_index": int(target_index),
                "task_id": str(task_id),
                "target_hash": target_hash_value,
                "candidate_uid": candidate_ids,
                "sequence_id": np.arange(len(legal_sequences), dtype=np.int64),
                "m0": legal_sequences[:, 0],
                "m1": legal_sequences[:, 1],
                "m2": legal_sequences[:, 2],
                "m3": legal_sequences[:, 3],
                "materials": material_strings,
                "r_core": geometry_np[:, 0],
                "t1": geometry_np[:, 1],
                "t2": geometry_np[:, 2],
                "t3": geometry_np[:, 3],
                "outer_radius": outer,
                "outer_boundary_margin_nm": MAX_OUTER_RADIUS - outer,
                "pgff_nmae": nmae,
                "pgff_nrmse": nrmse,
                "pgff_r2": r2,
                "pgff_peak_error_nm": peak_error,
                "target_peak_nm": float(wavelengths[int(np.argmax(target_row))]),
                "predicted_peak_nm": pred_peak_nm,
                "pgff_mean_qsca": qsca_np.mean(axis=1),
                "pgff_max_qsca": qsca_np.max(axis=1),
            }
        )
        ranked = rank_one_target(frame)
        ranked["selected_for_stage04"] = ranked["screen_rank"] <= int(args.top_k)
        all_frames.append(ranked)
        all_pgff_qsca.append(qsca_np)
        all_geometry.append(geometry_np)
        all_task_index.append(
            np.full(len(legal_sequences), target_index, dtype=np.int64)
        )

        best = ranked.iloc[0]
        summary_rows.append(
            {
                "task_index": int(target_index),
                "task_id": str(task_id),
                "target_hash": target_hash_value,
                "candidate_count": int(len(ranked)),
                "selected_count": int(ranked["selected_for_stage04"].sum()),
                "best_sequence_id": int(best["sequence_id"]),
                "best_materials": str(best["materials"]),
                "best_r_core": float(best["r_core"]),
                "best_t1": float(best["t1"]),
                "best_t2": float(best["t2"]),
                "best_t3": float(best["t3"]),
                "best_outer_radius": float(best["outer_radius"]),
                "best_pgff_nmae": float(best["pgff_nmae"]),
                "best_pgff_nrmse": float(best["pgff_nrmse"]),
                "best_pgff_r2": float(best["pgff_r2"]),
                "median_pgff_nmae": float(np.median(nmae)),
                "p10_pgff_nmae": float(np.quantile(nmae, 0.10)),
                "surrogate_nmae_le_0p05_count": int(np.sum(nmae <= 0.05)),
                "surrogate_nmae_le_0p02_count": int(np.sum(nmae <= 0.02)),
            }
        )
        if not args.no_plots:
            plot_target_screen(
                args.output_dir,
                str(task_id),
                wavelengths,
                target_row,
                ranked,
                qsca_np,
            )
        print(
            f"{task_id}: screened=400, selected={args.top_k}, "
            f"best PGFF NMAE={float(best['pgff_nmae']):.6f}, "
            f"materials={best['materials']}"
        )

    all_candidates = pd.concat(all_frames, ignore_index=True)
    all_candidates = all_candidates.sort_values(
        ["task_index", "screen_rank"], kind="mergesort"
    ).reset_index(drop=True)
    selected_candidates = all_candidates.loc[
        all_candidates["selected_for_stage04"]
    ].copy().reset_index(drop=True)
    target_summary = pd.DataFrame(summary_rows).sort_values("task_index").reset_index(drop=True)

    if len(all_candidates) != len(targets["task_id"]) * 400:
        raise RuntimeError("Unexpected total number of Stage 03 candidates.")
    if len(selected_candidates) != len(targets["task_id"]) * int(args.top_k):
        raise RuntimeError("Unexpected number of selected Stage 04 seeds.")
    if all_candidates["candidate_uid"].duplicated().any():
        raise RuntimeError("Candidate UIDs are not globally unique.")

    atomic_write_csv(args.output_dir / "material_sequence_candidates.csv", all_candidates)
    atomic_write_csv(args.output_dir / "selected_refinement_seeds.csv", selected_candidates)
    atomic_write_csv(args.output_dir / "material_sequence_screening_summary.csv", target_summary)

    # Freeze a sanitized target artifact containing Qsca only.
    frozen_target_path = args.output_dir / "blind_target_scattering_spectra.npz"
    save_npz_atomic(
        frozen_target_path,
        task_id=np.asarray(targets["task_id"]),
        wavelengths_nm=wavelengths,
        target_qsca=np.asarray(targets["target_qsca"], dtype=np.float64),
        target_hash=np.asarray(targets["target_hash"]),
    )

    qsca_stack = np.concatenate(all_pgff_qsca, axis=0).astype(np.float32)
    geometry_stack = np.concatenate(all_geometry, axis=0).astype(np.float32)
    task_index_stack = np.concatenate(all_task_index, axis=0)
    sequence_stack = np.tile(legal_sequences, (len(targets["task_id"]), 1))
    candidate_uid_stack = all_candidates.sort_values(
        ["task_index", "sequence_id"], kind="mergesort"
    )["candidate_uid"].to_numpy(dtype="U24")

    all_npz_path = args.output_dir / "material_sequence_candidate_spectra.npz"
    save_npz_atomic(
        all_npz_path,
        task_id=np.asarray(targets["task_id"]),
        wavelengths_nm=wavelengths,
        target_qsca=np.asarray(targets["target_qsca"], dtype=np.float32),
        target_hash=np.asarray(targets["target_hash"]),
        candidate_task_index=task_index_stack,
        candidate_sequence_id=np.tile(np.arange(400, dtype=np.int64), len(targets["task_id"])),
        candidate_uid=candidate_uid_stack,
        candidate_material_indices=sequence_stack,
        candidate_geometry=geometry_stack,
        candidate_pgff_qsca=qsca_stack,
    )

    selected_lookup = selected_candidates.sort_values(
        ["task_index", "screen_rank"], kind="mergesort"
    )
    selected_rows = (
        selected_lookup["task_index"].to_numpy(dtype=np.int64) * 400
        + selected_lookup["sequence_id"].to_numpy(dtype=np.int64)
    )
    selected_npz_path = args.output_dir / "selected_refinement_seeds.npz"
    save_npz_atomic(
        selected_npz_path,
        task_id=np.asarray(targets["task_id"]),
        wavelengths_nm=wavelengths,
        target_qsca=np.asarray(targets["target_qsca"], dtype=np.float32),
        target_hash=np.asarray(targets["target_hash"]),
        candidate_task_index=selected_lookup["task_index"].to_numpy(dtype=np.int64),
        candidate_screen_rank=selected_lookup["screen_rank"].to_numpy(dtype=np.int64),
        candidate_sequence_id=selected_lookup["sequence_id"].to_numpy(dtype=np.int64),
        candidate_uid=selected_lookup["candidate_uid"].to_numpy(dtype="U24"),
        candidate_material_indices=selected_lookup[["m0", "m1", "m2", "m3"]].to_numpy(dtype=np.int64),
        candidate_geometry=selected_lookup[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.float32),
        candidate_pgff_qsca=qsca_stack[selected_rows],
        candidate_pgff_nmae=selected_lookup["pgff_nmae"].to_numpy(dtype=np.float64),
    )

    best_values = target_summary["best_pgff_nmae"].to_numpy(dtype=np.float64)
    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "wall_time_seconds": float(time.perf_counter() - start_time),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "blind_policy": {
            "inverse_input": "target Qsca only",
            "consumed_target_fields": ["task_id", "wavelengths_nm", "target_qsca"],
            "reference_geometry_read": False,
            "reference_materials_read": False,
            "target_qabs_read": False,
            "target_qext_read": False,
            "forward_test_split_evaluated": False,
            "legacy_inverse_candidate_results_read": False,
            "stage03_uses_pymiediff": False,
        },
        "screening_protocol": {
            "legal_material_sequences_enumerated_exactly": 400,
            "inverse_predictions_per_target": 400,
            "geometry_predictions_per_material_sequence": 1,
            "ranking_primary_metric": "PGFF Qsca NMAE",
            "ranking_tie_breakers": [
                "PGFF Qsca NRMSE",
                "PGFF peak error",
                "sequence_id",
            ],
            "selected_per_target_for_stage04": int(args.top_k),
            "selection_uses_qabs": False,
            "selection_uses_reference_structure": False,
        },
        "dimensions": {
            "blind_targets": int(len(targets["task_id"])),
            "wavelengths": int(len(wavelengths)),
            "materials": int(len(material_names)),
            "legal_material_sequences": int(len(legal_sequences)),
            "all_candidates": int(len(all_candidates)),
            "selected_stage04_seeds": int(len(selected_candidates)),
            "test_samples_untouched": int(len(prepared["test_indices"])),
        },
        "screening_statistics": {
            "mean_best_pgff_nmae": float(best_values.mean()),
            "median_best_pgff_nmae": float(np.median(best_values)),
            "maximum_best_pgff_nmae": float(best_values.max()),
            "minimum_best_pgff_nmae": float(best_values.min()),
        },
        "stage02_provenance": {
            "best_epoch": int(stage02_summary["best_epoch"]),
            "best_validation_mean_nmae": float(stage02_summary["best_validation_mean_nmae"]),
            "inverse_model_config": checkpoint.get("model_config", {}),
        },
        "inputs": {
            "dataset": artifact(args.dataset),
            "prepared_data": artifact(args.prepared_data),
            "stage02_best_model": artifact(best_model_path),
            "stage02_training_summary": artifact(args.stage02_dir / "training_summary.json"),
            "pgff_checkpoint": artifact(args.pgff_dir / "best_model.pt"),
            "public_target_source": artifact(args.public_targets),
            "public_target_source_keys": targets["source_keys"].tolist(),
        },
        "outputs": {
            "sanitized_public_targets": artifact(frozen_target_path),
            "all_candidates_csv": artifact(args.output_dir / "material_sequence_candidates.csv"),
            "selected_candidates_csv": artifact(args.output_dir / "selected_refinement_seeds.csv"),
            "target_summary_csv": artifact(args.output_dir / "material_sequence_screening_summary.csv"),
            "all_candidate_spectra_npz": artifact(all_npz_path),
            "selected_stage04_seeds_npz": artifact(selected_npz_path),
        },
        "arguments": vars(args),
        "warning": (
            "Stage 03 ranks candidates with the frozen PGFF surrogate only. "
            "No Stage 03 metric is a final PyMieDiff accuracy claim."
        ),
    }
    save_json_atomic(args.output_dir / "material_sequence_screening_run_summary.json", summary)

    print("Stage 03 strict blind material screen PASSED")
    print(f"All candidates: {len(all_candidates)}")
    print(f"Selected Stage 04 seeds: {len(selected_candidates)}")
    print(f"Median best PGFF NMAE: {float(np.median(best_values)):.6f}")
    print("No private reference, Qabs target, Qext target, test split, or PyMieDiff was used.")


if __name__ == "__main__":
    main()
