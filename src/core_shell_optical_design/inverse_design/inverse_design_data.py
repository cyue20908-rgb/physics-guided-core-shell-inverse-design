# -*- coding: utf-8 -*-
"""Shared loading, validation, hashing, and path utilities.

This module consumes only the frozen dataset, split definitions, and forward
model assets stored in the organized reproducibility package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

from core_shell_optical_design.inverse_design.geometry_parameterization import (
    assert_legal_geometry,
    cumulative_to_geometry,
    material_sequence_mask,
)
from core_shell_optical_design.project_paths import (
    DATASET_PATH,
    DIRECT_MODEL_DIR,
    PHYSICS_GUIDED_MODEL_DIR,
    REPOSITORY_ROOT,
    REPRODUCED_INVERSE_RESULTS_DIR,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPOSITORY_ROOT

DEFAULT_DATASET = DATASET_PATH
DEFAULT_BASELINE_DIR = DIRECT_MODEL_DIR
DEFAULT_PGFF_DIR = PHYSICS_GUIDED_MODEL_DIR
DEFAULT_RESULT_ROOT = REPRODUCED_INVERSE_RESULTS_DIR
DEFAULT_PGFF_MODEL_SOURCE = (
    PROJECT_DIR
    / "src"
    / "core_shell_optical_design"
    / "forward_models"
    / "physics_guided_fourier_network.py"
)

DATASET_ENERGY_TOL = 1.0e-5
NEGATIVE_TOL = 1.0e-7


@dataclass(frozen=True)
class SplitBundle:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    geometry_mean: np.ndarray
    geometry_std: np.ndarray
    wavelengths: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(request: str) -> torch.device:
    request = str(request).lower()
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if request not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(request)


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
    os.replace(temp, path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def artifact(path: Path) -> Dict[str, Any]:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _load_config(npz: Mapping[str, Any]) -> Dict[str, Any]:
    if "config_json" in npz:
        value = npz["config_json"].item()
        return json.loads(str(value))
    if "config" in npz:
        value = npz["config"].item()
        if isinstance(value, dict):
            return value
    return {}


def load_forward_dataset(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Forward dataset not found: {path}")
    required = (
        "radii", "material_indices", "wavelengths", "Q_sca", "Q_abs", "Q_ext",
        "material_names", "material_n", "material_k",
    )
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Dataset is missing fields: {missing}")
        radii = data["radii"].astype(np.float64)
        material_indices = data["material_indices"].astype(np.int64)
        wavelengths = data["wavelengths"].astype(np.float64)
        qsca = data["Q_sca"].astype(np.float64)
        qabs = data["Q_abs"].astype(np.float64)
        qext = data["Q_ext"].astype(np.float64)
        material_names = data["material_names"].astype(str)
        material_n = data["material_n"].astype(np.float64)
        material_k = data["material_k"].astype(np.float64)
        sample_id = (
            data["sample_id"].astype(np.int64)
            if "sample_id" in data
            else np.arange(len(radii), dtype=np.int64)
        )
        config = _load_config(data)

    geometry = cumulative_to_geometry(radii)
    n_samples = len(radii)
    n_wavelengths = len(wavelengths)
    expected_geometry = (n_samples, 4)
    expected_spectrum = (n_samples, n_wavelengths)
    if radii.shape != expected_geometry or geometry.shape != expected_geometry:
        raise ValueError("radii/geometry shape mismatch.")
    if material_indices.shape != expected_geometry:
        raise ValueError("material_indices shape mismatch.")
    for name, values in (("Q_sca", qsca), ("Q_abs", qabs), ("Q_ext", qext)):
        if values.shape != expected_spectrum:
            raise ValueError(f"{name} shape mismatch: {values.shape} vs {expected_spectrum}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf.")
        if float(values.min()) < -NEGATIVE_TOL:
            raise ValueError(f"{name} contains a negative value below tolerance.")
    if wavelengths.ndim != 1 or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("wavelengths must be strictly increasing.")
    if material_n.shape != material_k.shape:
        raise ValueError("material_n and material_k shapes differ.")
    if material_n.shape != (len(material_names), n_wavelengths):
        raise ValueError("Material optical tables do not match materials/wavelengths.")
    if not np.isfinite(material_n).all() or not np.isfinite(material_k).all():
        raise ValueError("Material optical tables contain NaN or Inf.")
    if np.any(material_k < -1.0e-12):
        raise ValueError("material_k contains negative extinction coefficients.")

    assert_legal_geometry(geometry, label="forward dataset geometry")
    legal_material = material_sequence_mask(material_indices, material_names)
    if not np.all(legal_material):
        raise ValueError(
            f"Forward dataset contains {int((~legal_material).sum())} illegal material sequences."
        )
    residual = np.abs(qext - qsca - qabs)
    max_residual = float(residual.max())
    if max_residual > DATASET_ENERGY_TOL:
        raise ValueError(
            f"Dataset energy residual {max_residual:.3e} exceeds {DATASET_ENERGY_TOL:.3e}."
        )
    return {
        "path": path,
        "sample_id": sample_id,
        "radii": radii,
        "geometry": geometry,
        "material_indices": material_indices,
        "wavelengths": wavelengths,
        "Q_sca": qsca,
        "Q_abs": qabs,
        "Q_ext": qext,
        "material_names": material_names,
        "material_n": material_n,
        "material_k": material_k,
        "material_epsilon": (material_n + 1j * material_k) ** 2,
        "config": config,
        "max_energy_residual": max_residual,
    }


def load_split_bundle(dataset: Mapping[str, Any], baseline_dir: Path) -> SplitBundle:
    baseline_dir = Path(baseline_dir)
    paths = {
        "train": baseline_dir / "idx_train.npy",
        "validation": baseline_dir / "idx_val.npy",
        "test": baseline_dir / "idx_test.npy",
        "normalization": baseline_dir / "normalization.npz",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing baseline artifacts:\n" + "\n".join(str(p) for p in missing))
    train = np.load(paths["train"]).astype(np.int64)
    validation = np.load(paths["validation"]).astype(np.int64)
    test = np.load(paths["test"]).astype(np.int64)
    combined = np.concatenate((train, validation, test))
    n_samples = len(dataset["geometry"])
    if len(combined) != n_samples or len(np.unique(combined)) != n_samples:
        raise RuntimeError("Train/validation/test splits do not cover the dataset exactly once.")
    if combined.min() < 0 or combined.max() >= n_samples:
        raise RuntimeError("Split index is outside the dataset range.")
    if set(train.tolist()) & set(validation.tolist()):
        raise RuntimeError("Train and validation splits overlap.")
    if set(train.tolist()) & set(test.tolist()):
        raise RuntimeError("Train and test splits overlap.")
    if set(validation.tolist()) & set(test.tolist()):
        raise RuntimeError("Validation and test splits overlap.")

    with np.load(paths["normalization"], allow_pickle=False) as norm:
        for key in ("geometry_mean", "geometry_std", "wavelengths"):
            if key not in norm:
                raise KeyError(f"normalization.npz is missing {key!r}.")
        geometry_mean = norm["geometry_mean"].astype(np.float32)
        geometry_std = norm["geometry_std"].astype(np.float32)
        wavelengths = norm["wavelengths"].astype(np.float32)
    if not np.array_equal(wavelengths, np.asarray(dataset["wavelengths"], dtype=np.float32)):
        raise RuntimeError("Baseline and dataset wavelength grids differ.")

    # Reproduce the exact float32 preprocessing used by the PGFF training script.
    radii32 = np.asarray(dataset["radii"], dtype=np.float32)
    geometry32 = cumulative_to_geometry(radii32)
    mean32 = geometry32[train].mean(axis=0)
    std32 = geometry32[train].std(axis=0)
    if not np.allclose(geometry_mean, mean32, rtol=1.0e-6, atol=1.0e-6):
        raise RuntimeError("Stored geometry_mean is inconsistent with the training split.")
    if not np.allclose(geometry_std, std32, rtol=1.0e-6, atol=1.0e-6):
        raise RuntimeError("Stored geometry_std is inconsistent with the training split.")
    if np.any(geometry_std <= 0.0):
        raise RuntimeError("geometry_std must be strictly positive.")
    return SplitBundle(train, validation, test, geometry_mean, geometry_std, wavelengths)


def add_project_root_to_path() -> None:
    root = str(PROJECT_DIR / "src")
    if root not in sys.path:
        sys.path.insert(0, root)
