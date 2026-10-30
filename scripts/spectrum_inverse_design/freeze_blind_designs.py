# -*- coding: utf-8 -*-
"""Pre-unblinding freeze and integrity audit.

This stage performs a final, private-reference-free audit of the strict-blind
integer results from Stage 05. It verifies the public benchmark lock, target
hashes, Stage-05 artifact hashes, integer geometry legality, material legality,
and independently recomputes the frozen spectral metrics from the saved
PyMieDiff spectra. It then creates an immutable-by-hash snapshot for the later
post-hoc unblind comparison.

IMPORTANT: this script never opens the private reference directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from core_shell_optical_design.project_paths import REPRODUCED_INVERSE_RESULTS_DIR


STAGE = "pre_unblinding_integrity_freeze"
VERSION = "tandem-mie-inverse-2026-07-28-stage06a-v2-target-transport-audit"

RESULT_ROOT = REPRODUCED_INVERSE_RESULTS_DIR
DEFAULT_PUBLIC_DIR = RESULT_ROOT / "blind_targets_public"
DEFAULT_STAGE05_DIR = RESULT_ROOT / "integer_geometry_search"
DEFAULT_OUTPUT_DIR = RESULT_ROOT / "pre_unblinding_freeze"

LOWER_BOUNDS = np.asarray([120.0, 20.0, 20.0, 20.0], dtype=np.float64)
UPPER_BOUNDS = np.asarray([220.0, 80.0, 80.0, 80.0], dtype=np.float64)
MAX_OUTER_RADIUS = 300.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and audit strict-blind integer-search results before unblinding"
    )
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--stage05-dir", type=Path, default=DEFAULT_STAGE05_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metric-atol", type=float, default=2.0e-10)
    parser.add_argument("--metric-rtol", type=float, default=2.0e-8)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp, path)


def save_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def spectrum_hash(wavelengths: np.ndarray, qsca: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(wavelengths, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(qsca, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def scalar_text(value: np.ndarray | Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    raise ValueError("Expected a scalar text value.")


def require_keys(source: Mapping[str, Any], names: Iterable[str], label: str) -> None:
    missing = [name for name in names if name not in source]
    if missing:
        raise KeyError(f"{label} is missing required fields: {missing}")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def legal_geometry_mask(geometry: np.ndarray, tol: float = 1.0e-7) -> np.ndarray:
    g = np.asarray(geometry, dtype=np.float64)
    return (
        np.all(g >= LOWER_BOUNDS[None, :] - tol, axis=1)
        & np.all(g <= UPPER_BOUNDS[None, :] + tol, axis=1)
        & (np.sum(g, axis=1) <= MAX_OUTER_RADIUS + tol)
    )


def legal_material_mask(materials: np.ndarray) -> np.ndarray:
    m = np.asarray(materials, dtype=np.int64)
    in_range = np.all((m >= 0) & (m <= 5), axis=1)
    shell_not_air = np.all(m[:, 1:] != 5, axis=1)
    adjacent_different = np.all(m[:, 1:] != m[:, :-1], axis=1)
    return in_range & shell_not_air & adjacent_different


def recompute_metrics(target: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if target.shape != pred.shape or target.ndim != 2:
        raise ValueError(f"Target/predicted spectra must share a 2D shape, got {target.shape} and {pred.shape}.")
    scale = np.max(target, axis=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Every target spectrum must have a finite positive peak.")
    error = pred - target
    nmae = np.mean(np.abs(error), axis=1) / scale
    nrmse = np.sqrt(np.mean(error * error, axis=1)) / scale
    centered = target - np.mean(target, axis=1, keepdims=True)
    ss_tot = np.sum(centered * centered, axis=1)
    ss_res = np.sum(error * error, axis=1)
    r2 = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, np.where(ss_res == 0.0, 1.0, -np.inf))
    return nmae, nrmse, r2



def verify_target_transport(
    public_target_qsca: np.ndarray,
    frozen_target_qsca: np.ndarray,
) -> Dict[str, Any]:
    """Verify the precision-only target transport used by Stages 03--05.

    Stage 03 intentionally serialized the downstream optimization target as
    float32. Stages 04 and 05 then loaded that array and promoted it to
    float64 for PyMieDiff calculations. Therefore the Stage-05 NPZ can differ
    bit-for-bit from the canonical public float64 target while still being an
    exact float32 round-trip of it. This function accepts only those two
    provenance-preserving cases; arbitrary numerical tolerance is not used.
    """
    public64 = np.asarray(public_target_qsca, dtype=np.float64)
    frozen64 = np.asarray(frozen_target_qsca, dtype=np.float64)
    if public64.shape != frozen64.shape:
        raise RuntimeError(
            f"Frozen/public target shape mismatch: {frozen64.shape} versus {public64.shape}."
        )

    public32_roundtrip = public64.astype(np.float32).astype(np.float64)
    if np.array_equal(frozen64, public64):
        mode = "exact_float64"
        expected = public64
    elif np.array_equal(frozen64, public32_roundtrip):
        mode = "exact_float32_roundtrip"
        expected = public32_roundtrip
    else:
        difference = np.abs(frozen64 - public64)
        roundtrip_difference = np.abs(frozen64 - public32_roundtrip)
        raise RuntimeError(
            "Frozen NPZ target spectra are neither exact public float64 values nor "
            "the exact float32 round-trip used by Stage 03. "
            f"max |frozen-public|={float(np.max(difference)):.3e}; "
            f"max |frozen-public32|={float(np.max(roundtrip_difference)):.3e}."
        )

    absolute_error = np.abs(expected - public64)
    peak = np.maximum(np.max(np.abs(public64), axis=1), 1.0e-300)
    per_target_max_abs = np.max(absolute_error, axis=1)
    per_target_max_peak_relative = per_target_max_abs / peak
    return {
        "mode": mode,
        "evaluation_target_qsca": frozen64,
        "per_target_max_abs_error": per_target_max_abs,
        "per_target_max_peak_relative_error": per_target_max_peak_relative,
        "maximum_abs_error": float(np.max(per_target_max_abs)),
        "maximum_peak_relative_error": float(np.max(per_target_max_peak_relative)),
    }

def verify_recorded_artifact(record: Mapping[str, Any], actual_path: Path, label: str) -> None:
    expected_hash = str(record.get("sha256", ""))
    expected_size = int(record.get("size_bytes", -1))
    actual_hash = sha256_file(actual_path)
    actual_size = int(actual_path.stat().st_size)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected_hash}, received {actual_hash}")
    if expected_size >= 0 and actual_size != expected_size:
        raise RuntimeError(f"{label} size mismatch: expected {expected_size}, received {actual_size}")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory already contains a freeze snapshot: {path}\n"
            "Do not overwrite a formal freeze unless you intentionally restart the protocol. "
            "Use --overwrite only before any unblinding has occurred."
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "snapshot").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.metric_atol) or args.metric_atol <= 0.0:
        raise ValueError("--metric-atol must be finite and positive.")
    if not np.isfinite(args.metric_rtol) or args.metric_rtol <= 0.0:
        raise ValueError("--metric-rtol must be finite and positive.")

    started = time.time()
    public_dir = Path(args.public_dir)
    stage05_dir = Path(args.stage05_dir)
    output_dir = Path(args.output_dir)

    public_npz_path = public_dir / "blind_target_spectra.npz"
    public_manifest_path = public_dir / "blind_benchmark_manifest.json"
    benchmark_lock_path = public_dir / "blind_benchmark_lock.json"
    stage05_summary_path = stage05_dir / "integer_search_run_summary.json"
    frozen_csv_path = stage05_dir / "frozen_blind_design_results.csv"
    frozen_npz_path = stage05_dir / "frozen_blind_design_results.npz"
    integer_summary_path = stage05_dir / "integer_design_target_summary.csv"

    for required in (
        public_npz_path,
        public_manifest_path,
        benchmark_lock_path,
        stage05_summary_path,
        frozen_csv_path,
        frozen_npz_path,
        integer_summary_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    print("Stage 06A pre-unblind freeze and integrity audit")
    print("Private reference directory is not accessed by this stage.")

    lock = load_json(benchmark_lock_path)
    manifest = load_json(public_manifest_path)
    stage05_summary = load_json(stage05_summary_path)
    public = load_npz(public_npz_path)
    frozen_npz = load_npz(frozen_npz_path)
    frozen_csv = pd.read_csv(frozen_csv_path, encoding="utf-8-sig")
    integer_summary = pd.read_csv(integer_summary_path, encoding="utf-8-sig")

    if lock.get("frozen") is not True:
        raise RuntimeError("blind_benchmark_lock.json is not marked frozen.")
    if sha256_file(public_npz_path) != str(lock.get("public_target_sha256", "")):
        raise RuntimeError("Public target file no longer matches blind_benchmark_lock.json.")
    if sha256_file(public_manifest_path) != str(lock.get("manifest_sha256", "")):
        raise RuntimeError("Public target manifest no longer matches blind_benchmark_lock.json.")
    if manifest.get("status") != "passed":
        raise RuntimeError("Blind target manifest is not passed.")
    if stage05_summary.get("status") != "passed":
        raise RuntimeError("Stage 05 summary is not passed.")

    require_keys(
        public,
        ("task_id", "wavelengths_nm", "target_qsca", "target_hash", "benchmark_id"),
        "public target NPZ",
    )
    require_keys(
        frozen_npz,
        (
            "task_id", "wavelengths_nm", "target_qsca", "target_hash",
            "candidate_task_index", "candidate_integer_rank", "candidate_design_uid",
            "candidate_sequence_id", "candidate_material_indices",
            "candidate_integer_geometry", "candidate_mie_qsca", "candidate_mie_qabs",
            "candidate_mie_qext", "candidate_mie_nmae", "candidate_mie_nrmse",
            "candidate_mie_r2", "candidate_standard_success", "candidate_high_accuracy",
        ),
        "Stage-05 frozen NPZ",
    )

    task_id = public["task_id"].astype(str)
    wavelengths = np.asarray(public["wavelengths_nm"], dtype=np.float64)
    target_qsca = np.asarray(public["target_qsca"], dtype=np.float64)
    target_hash = public["target_hash"].astype(str)
    benchmark_id = scalar_text(public["benchmark_id"])

    if len(task_id) != 10 or target_qsca.shape != (10, len(wavelengths)):
        raise RuntimeError(f"Expected 10 public targets, received task shape {task_id.shape}, spectra {target_qsca.shape}.")
    if len(set(task_id.tolist())) != 10:
        raise RuntimeError("Public task IDs are not unique.")
    if benchmark_id != str(lock.get("benchmark_id", "")):
        raise RuntimeError("Public benchmark ID does not match blind_benchmark_lock.json.")
    if benchmark_id != str(manifest.get("benchmark_id", "")):
        raise RuntimeError("Public benchmark ID does not match the target manifest.")

    recomputed_hash = np.asarray([spectrum_hash(wavelengths, row) for row in target_qsca])
    if not np.array_equal(recomputed_hash, target_hash):
        raise RuntimeError("One or more public target spectrum hashes are invalid.")

    # Verify Stage-05 output records against files currently on disk.
    output_records = stage05_summary.get("outputs", {})
    if not isinstance(output_records, dict):
        raise TypeError("Stage-05 outputs record is malformed.")
    mapping = {
        "frozen_blind_integer_results_csv": frozen_csv_path,
        "frozen_blind_integer_results_npz": frozen_npz_path,
        "integer_target_summary": integer_summary_path,
    }
    for key, path in mapping.items():
        record = output_records.get(key)
        if not isinstance(record, dict):
            raise KeyError(f"Stage-05 summary does not record output {key}.")
        verify_recorded_artifact(record, path, key)

    counts = stage05_summary.get("counts", {})
    if int(counts.get("frozen_results", -1)) != 10:
        raise RuntimeError("Stage 05 did not freeze exactly ten results.")

    candidate_task_index = np.asarray(frozen_npz["candidate_task_index"], dtype=np.int64)
    candidate_rank = np.asarray(frozen_npz["candidate_integer_rank"], dtype=np.int64)
    materials = np.asarray(frozen_npz["candidate_material_indices"], dtype=np.int64)
    geometry = np.asarray(frozen_npz["candidate_integer_geometry"], dtype=np.float64)
    pred_qsca = np.asarray(frozen_npz["candidate_mie_qsca"], dtype=np.float64)
    pred_qabs = np.asarray(frozen_npz["candidate_mie_qabs"], dtype=np.float64)
    pred_qext = np.asarray(frozen_npz["candidate_mie_qext"], dtype=np.float64)

    n_frozen = len(candidate_task_index)
    if n_frozen != 10:
        raise RuntimeError(f"Frozen NPZ contains {n_frozen} candidates instead of 10.")
    if not np.array_equal(np.sort(candidate_task_index), np.arange(10, dtype=np.int64)):
        raise RuntimeError("Frozen NPZ must contain exactly one candidate for every task index 0..9.")
    order = np.argsort(candidate_task_index)
    candidate_task_index = candidate_task_index[order]
    candidate_rank = candidate_rank[order]
    materials = materials[order]
    geometry = geometry[order]
    pred_qsca = pred_qsca[order]
    pred_qabs = pred_qabs[order]
    pred_qext = pred_qext[order]

    for key in (
        "candidate_design_uid", "candidate_sequence_id", "candidate_mie_nmae",
        "candidate_mie_nrmse", "candidate_mie_r2", "candidate_mie_peak_error_nm",
        "candidate_standard_success", "candidate_high_accuracy",
    ):
        frozen_npz[key] = np.asarray(frozen_npz[key])[order]

    if not np.all(candidate_rank == 1):
        raise RuntimeError("Every frozen candidate must have integer rank 1.")
    if geometry.shape != (10, 4) or materials.shape != (10, 4):
        raise RuntimeError("Frozen geometry/material arrays must have shape (10, 4).")
    if not np.allclose(geometry, np.rint(geometry), atol=0.0, rtol=0.0):
        raise RuntimeError("A frozen geometry is not on the 1-nm integer grid.")
    if not np.all(legal_geometry_mask(geometry)):
        raise RuntimeError("A frozen integer geometry violates the design bounds.")
    if not np.all(legal_material_mask(materials)):
        raise RuntimeError("A frozen material sequence violates the legal material rules.")

    if pred_qsca.shape != target_qsca.shape or pred_qabs.shape != target_qsca.shape or pred_qext.shape != target_qsca.shape:
        raise RuntimeError("Frozen PyMieDiff spectra have an unexpected shape.")
    if np.any(~np.isfinite(pred_qsca)) or np.any(~np.isfinite(pred_qabs)) or np.any(~np.isfinite(pred_qext)):
        raise RuntimeError("Frozen PyMieDiff spectra contain non-finite values.")
    if np.min(pred_qsca) < -1.0e-7 or np.min(pred_qabs) < -1.0e-7 or np.min(pred_qext) < -1.0e-7:
        raise RuntimeError("Frozen PyMieDiff spectra contain materially negative efficiencies.")
    energy_residual = np.max(np.abs(pred_qext - pred_qsca - pred_qabs), axis=1)
    if np.max(energy_residual) > 1.0e-8:
        raise RuntimeError(f"Frozen PyMieDiff energy residual is too large: {np.max(energy_residual):.3e}")

    if not np.array_equal(frozen_npz["task_id"].astype(str), task_id):
        raise RuntimeError("Frozen NPZ task IDs do not match public task IDs.")
    if not np.allclose(np.asarray(frozen_npz["wavelengths_nm"], dtype=np.float64), wavelengths, atol=0.0, rtol=0.0):
        raise RuntimeError("Frozen NPZ wavelength grid does not match the public target grid.")
    target_transport = verify_target_transport(
        target_qsca,
        np.asarray(frozen_npz["target_qsca"], dtype=np.float64),
    )
    evaluation_target_qsca = np.asarray(
        target_transport["evaluation_target_qsca"], dtype=np.float64
    )
    if not np.array_equal(frozen_npz["target_hash"].astype(str), target_hash):
        raise RuntimeError("Frozen NPZ target hashes do not match the canonical public targets.")

    # Reproduce Stage-05 metrics against the exact target representation that
    # entered Stages 04--05, then independently report canonical metrics
    # against the original public float64 benchmark.
    recomputed_nmae, recomputed_nrmse, recomputed_r2 = recompute_metrics(
        evaluation_target_qsca, pred_qsca
    )
    canonical_nmae, canonical_nrmse, canonical_r2 = recompute_metrics(
        target_qsca, pred_qsca
    )
    recorded_nmae = np.asarray(frozen_npz["candidate_mie_nmae"], dtype=np.float64)
    recorded_nrmse = np.asarray(frozen_npz["candidate_mie_nrmse"], dtype=np.float64)
    recorded_r2 = np.asarray(frozen_npz["candidate_mie_r2"], dtype=np.float64)
    for name, recorded, recomputed in (
        ("NMAE", recorded_nmae, recomputed_nmae),
        ("NRMSE", recorded_nrmse, recomputed_nrmse),
        ("R2", recorded_r2, recomputed_r2),
    ):
        if not np.allclose(recorded, recomputed, atol=args.metric_atol, rtol=args.metric_rtol):
            worst = float(np.max(np.abs(recorded - recomputed)))
            raise RuntimeError(f"Stored {name} does not match independent recomputation; max difference={worst:.3e}")

    standard = np.asarray(frozen_npz["candidate_standard_success"], dtype=bool)
    high_accuracy = np.asarray(frozen_npz["candidate_high_accuracy"], dtype=bool)
    expected_standard = (recomputed_nmae <= 0.05) & (recomputed_nrmse <= 0.05) & (recomputed_r2 >= 0.99)
    expected_high = (recomputed_nmae <= 0.02) & (recomputed_nrmse <= 0.02)
    canonical_standard = (canonical_nmae <= 0.05) & (canonical_nrmse <= 0.05) & (canonical_r2 >= 0.99)
    canonical_high = (canonical_nmae <= 0.02) & (canonical_nrmse <= 0.02)
    # Stage 05 may additionally enforce a peak criterion for applicable targets. It can only reduce success.
    if np.any(standard & ~expected_standard):
        raise RuntimeError("A stored standard-success flag contradicts the core metric thresholds.")
    if not np.array_equal(high_accuracy, expected_high):
        raise RuntimeError("Stored high-accuracy flags do not match the declared thresholds.")

    if len(frozen_csv) != 10 or len(integer_summary) != 10:
        raise RuntimeError("Stage-05 CSV outputs must each contain ten rows.")
    frozen_csv = frozen_csv.sort_values("task_index", kind="mergesort").reset_index(drop=True)
    integer_summary = integer_summary.sort_values("task_index", kind="mergesort").reset_index(drop=True)
    for frame_name, frame in (("frozen CSV", frozen_csv), ("integer summary", integer_summary)):
        if frame["task_id"].astype(str).tolist() != task_id.tolist():
            raise RuntimeError(f"{frame_name} task IDs do not match the public benchmark.")
        csv_geometry = frame[["r_core", "t1", "t2", "t3"]].to_numpy(dtype=np.float64)
        if not np.allclose(csv_geometry, geometry, atol=0.0, rtol=0.0):
            raise RuntimeError(f"{frame_name} geometry does not match the frozen NPZ.")

    audit = pd.DataFrame(
        {
            "task_index": np.arange(10, dtype=np.int64),
            "task_id": task_id,
            "target_hash": target_hash,
            "design_uid": frozen_npz["candidate_design_uid"].astype(str),
            "sequence_id": np.asarray(frozen_npz["candidate_sequence_id"], dtype=np.int64),
            "m0": materials[:, 0],
            "m1": materials[:, 1],
            "m2": materials[:, 2],
            "m3": materials[:, 3],
            "r_core": geometry[:, 0].astype(np.int64),
            "t1": geometry[:, 1].astype(np.int64),
            "t2": geometry[:, 2].astype(np.int64),
            "t3": geometry[:, 3].astype(np.int64),
            "outer_radius": np.sum(geometry, axis=1).astype(np.int64),
            "target_transport_mode": [str(target_transport["mode"])] * 10,
            "target_transport_max_abs_error": np.asarray(target_transport["per_target_max_abs_error"], dtype=np.float64),
            "target_transport_max_peak_relative_error": np.asarray(target_transport["per_target_max_peak_relative_error"], dtype=np.float64),
            "stage05_target_recomputed_mie_nmae": recomputed_nmae,
            "stage05_target_recomputed_mie_nrmse": recomputed_nrmse,
            "stage05_target_recomputed_mie_r2": recomputed_r2,
            "canonical_public_mie_nmae": canonical_nmae,
            "canonical_public_mie_nrmse": canonical_nrmse,
            "canonical_public_mie_r2": canonical_r2,
            "maximum_energy_residual": energy_residual,
            "stage05_standard_success": standard,
            "stage05_high_accuracy": high_accuracy,
            "canonical_public_standard_success": canonical_standard,
            "canonical_public_high_accuracy": canonical_high,
        }
    )

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    snapshot_dir = output_dir / "snapshot"
    snapshot_sources = {
        "blind_target_spectra.npz": public_npz_path,
        "blind_benchmark_manifest.json": public_manifest_path,
        "blind_benchmark_lock.json": benchmark_lock_path,
        "integer_search_run_summary.json": stage05_summary_path,
        "frozen_blind_design_results.csv": frozen_csv_path,
        "frozen_blind_design_results.npz": frozen_npz_path,
        "integer_design_target_summary.csv": integer_summary_path,
    }
    copied: Dict[str, Dict[str, Any]] = {}
    for name, source in snapshot_sources.items():
        destination = snapshot_dir / name
        shutil.copy2(source, destination)
        copied[name] = artifact(destination)

    audit_csv_path = output_dir / "preunblind_audit_per_target.csv"
    save_csv_atomic(audit_csv_path, audit)

    freeze_payload: Dict[str, Any] = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "created_unix_time": time.time(),
        "wall_time_seconds": time.time() - started,
        "python": platform.python_version(),
        "benchmark_id": benchmark_id,
        "private_reference_accessed": False,
        "policy": {
            "stage05_results_frozen_before_unblind": True,
            "public_benchmark_lock_verified": True,
            "stage05_output_hashes_verified": True,
            "target_hashes_recomputed": True,
            "stage03_float32_target_transport_verified_exactly": True,
            "spectral_metrics_independently_recomputed": True,
            "canonical_public_metrics_independently_recomputed": True,
            "integer_geometry_legality_verified": True,
            "material_sequence_legality_verified": True,
            "pymiediff_energy_closure_verified": True,
            "optimization_or_reranking_performed": False,
        },
        "counts": {
            "targets": 10,
            "frozen_results": 10,
            "standard_success": int(np.sum(standard)),
            "high_accuracy": int(np.sum(high_accuracy)),
        },
        "target_transport": {
            "mode": str(target_transport["mode"]),
            "maximum_absolute_qsca_difference_from_public_float64": float(target_transport["maximum_abs_error"]),
            "maximum_peak_relative_qsca_difference_from_public_float64": float(target_transport["maximum_peak_relative_error"]),
            "accepted_rule": "exact public float64 or exact public->float32->float64 round-trip only",
        },
        "aggregate_metrics": {
            "stage05_target_median_frozen_nmae": float(np.median(recomputed_nmae)),
            "stage05_target_mean_frozen_nmae": float(np.mean(recomputed_nmae)),
            "stage05_target_maximum_frozen_nmae": float(np.max(recomputed_nmae)),
            "stage05_target_median_frozen_nrmse": float(np.median(recomputed_nrmse)),
            "stage05_target_minimum_frozen_r2": float(np.min(recomputed_r2)),
            "canonical_public_median_frozen_nmae": float(np.median(canonical_nmae)),
            "canonical_public_mean_frozen_nmae": float(np.mean(canonical_nmae)),
            "canonical_public_maximum_frozen_nmae": float(np.max(canonical_nmae)),
            "canonical_public_median_frozen_nrmse": float(np.median(canonical_nrmse)),
            "canonical_public_minimum_frozen_r2": float(np.min(canonical_r2)),
            "maximum_energy_residual": float(np.max(energy_residual)),
        },
        "source_artifacts": {
            "public_targets": artifact(public_npz_path),
            "public_manifest": artifact(public_manifest_path),
            "benchmark_lock": artifact(benchmark_lock_path),
            "stage05_summary": artifact(stage05_summary_path),
            "stage05_frozen_csv": artifact(frozen_csv_path),
            "stage05_frozen_npz": artifact(frozen_npz_path),
            "stage05_integer_summary": artifact(integer_summary_path),
        },
        "snapshot_artifacts": copied,
        "audit_table": artifact(audit_csv_path),
    }

    # A content hash over the freeze facts, excluding its own file hash.
    canonical = json.dumps(freeze_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    freeze_payload["freeze_record_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    freeze_path = output_dir / "pre_unblinding_freeze_record.json"
    save_json_atomic(freeze_path, freeze_payload)

    instruction_path = output_dir / "unblinding_instruction.txt"
    with open(instruction_path, "w", encoding="utf-8") as handle:
        handle.write(
            "Stage 06A passed. The ten inverse results are frozen by SHA256 before private-reference access.\n"
            "Do not rerun or modify Stage 03-05 after opening the private reference.\n"
            "The next formal step is Stage 06B post-hoc unblind comparison.\n"
        )

    print("Stage 06A pre-unblind freeze PASSED")
    print(f"Benchmark ID: {benchmark_id}")
    print(f"Frozen results: 10; standard success: {int(np.sum(standard))}/10; high accuracy: {int(np.sum(high_accuracy))}/10")
    print(
        "Target transport: "
        f"{target_transport['mode']}; max public-float64 difference="
        f"{float(target_transport['maximum_abs_error']):.3e}"
    )
    print(f"Median Stage-05-target Mie NMAE: {float(np.median(recomputed_nmae)):.6f}")
    print(f"Median canonical-public Mie NMAE: {float(np.median(canonical_nmae)):.6f}")
    print(f"Freeze record: {freeze_path}")
    print("Private references were not read. Results may now be post-hoc unblinded.")


if __name__ == "__main__":
    main()
