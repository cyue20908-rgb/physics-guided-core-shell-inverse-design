# -*- coding: utf-8 -*-
"""Frozen PGFF reconstruction and differentiable PyMieDiff forward wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import json
import math

import numpy as np
import torch

from core_shell_optical_design.inverse_design.inverse_design_data import (
    add_project_root_to_path,
    load_split_bundle,
)
from core_shell_optical_design.forward_models.physics_guided_fourier_network import (
    PGFFNet,
    PHYSICS_FEATURE_NAMES,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def load_frozen_pgff(
    dataset: Mapping[str, Any],
    baseline_dir: Path,
    pgff_dir: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Reconstruct the exact trained PGFFNet and freeze its parameters.

    Freezing parameters does *not* disable autograd with respect to continuous
    geometry inputs. The inverse network can therefore receive gradients through
    the frozen forward surrogate.
    """
    baseline_dir = Path(baseline_dir)
    pgff_dir = Path(pgff_dir)
    required = (
        pgff_dir / "best_model.pt",
        pgff_dir / "hyperparameters.json",
        pgff_dir / "physics_normalization.npz",
        baseline_dir / "normalization.npz",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing PGFF artifacts:\n" + "\n".join(str(p) for p in missing))

    add_project_root_to_path()

    split = load_split_bundle(dataset, baseline_dir)
    hyper = _load_json(pgff_dir / "hyperparameters.json")
    with np.load(pgff_dir / "physics_normalization.npz", allow_pickle=False) as physics:
        for key in ("mean", "std"):
            if key not in physics:
                raise KeyError(f"physics_normalization.npz is missing {key!r}.")
        physics_mean = physics["mean"].astype(np.float32)
        physics_std = physics["std"].astype(np.float32)
        if "feature_names" in physics:
            stored = physics["feature_names"].astype(str).tolist()
            if stored != list(PHYSICS_FEATURE_NAMES):
                raise RuntimeError(
                    "Stored physics feature names do not match the organized "
                    "physics-guided forward model."
                )
    if np.any(physics_std <= 0.0):
        raise RuntimeError("PGFF physics feature std must be positive.")

    model = PGFFNet(
        geometry_mean=torch.as_tensor(split.geometry_mean, dtype=torch.float32),
        geometry_std=torch.as_tensor(split.geometry_std, dtype=torch.float32),
        wavelengths_nm=torch.as_tensor(dataset["wavelengths"], dtype=torch.float32),
        material_n=torch.as_tensor(dataset["material_n"], dtype=torch.float32),
        material_k=torch.as_tensor(dataset["material_k"], dtype=torch.float32),
        physics_mean=torch.as_tensor(physics_mean, dtype=torch.float32),
        physics_std=torch.as_tensor(physics_std, dtype=torch.float32),
        hidden_dim=int(hyper.get("hidden_dim", 192)),
        n_fourier_frequencies=int(hyper.get("fourier_frequencies", 32)),
        fourier_scale=float(hyper.get("fourier_scale", 3.0)),
        fourier_seed=int(hyper.get("fourier_seed", 2026)),
        dropout=float(hyper.get("dropout", 0.0)),
    ).to(device)

    checkpoint_path = pgff_dir / "best_model.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pgff_forward(
    model: torch.nn.Module,
    geometry: torch.Tensor,
    material_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if geometry.ndim != 2 or geometry.shape[1] != 4:
        raise ValueError("geometry must have shape (B,4).")
    if material_indices.shape != geometry.shape:
        raise ValueError("material_indices must have the same (B,4) shape as geometry.")
    prediction = model(geometry.float(), material_indices.long())
    for key in ("q_sca", "q_abs", "q_ext"):
        if key not in prediction:
            raise KeyError(f"PGFF prediction is missing {key!r}.")
        if not torch.isfinite(prediction[key]).all():
            raise RuntimeError(f"PGFF prediction {key} contains NaN or Inf.")
    return prediction


def import_pymiediff():
    try:
        import pymiediff as pmd
    except ImportError as exc:
        raise ImportError(
            "PyMieDiff is not installed in the active environment. "
            "Run this project in the user's miediff environment."
        ) from exc
    return pmd


def pymiediff_forward(
    geometry: torch.Tensor,
    material_indices: torch.Tensor,
    dataset: Mapping[str, Any],
    *,
    backend: str = "pena",
    precision: str = "double",
) -> Dict[str, torch.Tensor]:
    """Differentiable four-layer PyMieDiff forward calculation."""
    if geometry.ndim != 2 or geometry.shape[1] != 4:
        raise ValueError("geometry must have shape (B,4).")
    if material_indices.shape != geometry.shape:
        raise ValueError("material_indices must have the same shape as geometry.")
    pmd = import_pymiediff()
    dtype = torch.float64 if precision == "double" else torch.float32
    geometry = geometry.to(dtype=dtype)
    materials = material_indices.to(device=geometry.device, dtype=torch.long)
    radii = torch.cumsum(geometry, dim=1)

    n = torch.as_tensor(dataset["material_n"], dtype=dtype, device=geometry.device)
    k = torch.as_tensor(dataset["material_k"], dtype=dtype, device=geometry.device)
    eps_table = torch.complex(n, k).square()
    eps_layers = eps_table[materials]
    wavelengths = torch.as_tensor(dataset["wavelengths"], dtype=dtype, device=geometry.device)
    k0 = 2.0 * math.pi / wavelengths
    eps_env = torch.tensor(1.0 + 0.0j, dtype=eps_layers.dtype, device=geometry.device)

    result = pmd.multishell.cross_sections(
        k0=k0,
        r_layers=radii,
        eps_layers=eps_layers,
        eps_env=eps_env,
        backend=str(backend),
        precision=str(precision),
        n_max=None,
    )
    batch = geometry.shape[0]
    n_wavelengths = len(wavelengths)
    output = {
        "q_sca": result["q_sca"].reshape(batch, n_wavelengths),
        "q_abs": result["q_abs"].reshape(batch, n_wavelengths),
        "q_ext": result["q_ext"].reshape(batch, n_wavelengths),
    }
    for key, values in output.items():
        if not torch.isfinite(values).all():
            raise RuntimeError(f"PyMieDiff {key} contains NaN or Inf.")
    return output
