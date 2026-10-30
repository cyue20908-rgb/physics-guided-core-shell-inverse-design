# -*- coding: utf-8 -*-
"""Strict geometry and material constraints for the new tandem inverse workflow.

Design convention
-----------------
geometry = [r_core, t1, t2, t3] in nanometres.
Cumulative radii are obtained with ``torch.cumsum`` / ``numpy.cumsum``.

Legal domain inherited from the frozen forward database:
    120 <= r_core <= 220 nm
     20 <= t1,t2,t3 <= 80 nm
    r_core + t1 + t2 + t3 <= 300 nm

Material rules:
    - exactly four layers;
    - Air may appear only in the core;
    - adjacent layers must use different materials.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
import torch

N_LAYERS = 4
R_CORE_MIN = 120.0
R_CORE_MAX = 220.0
T_MIN = 20.0
T_MAX = 80.0
MAX_OUTER_RADIUS = 300.0

# Keep differentiable outputs slightly inside the hard outer-radius boundary.
# Float32 has a spacing of about 3e-5 nm around 300 nm; an exact 300 nm
# construction can therefore be exported as 300.00003 nm after summation.
# A 1e-3 nm interior margin is physically negligible and prevents false
# boundary violations in every later stage.
GEOMETRY_INTERIOR_MARGIN_NM = 1.0e-3
GEOMETRY_CHECK_ATOL_NM = 1.0e-4

GEOMETRY_COLUMNS = ("r_core", "t1", "t2", "t3")
MATERIAL_COLUMNS = ("m0", "m1", "m2", "m3")

LOWER_BOUNDS = np.asarray([R_CORE_MIN, T_MIN, T_MIN, T_MIN], dtype=np.float64)
UPPER_BOUNDS = np.asarray([R_CORE_MAX, T_MAX, T_MAX, T_MAX], dtype=np.float64)
EXTRA_CAPACITY = UPPER_BOUNDS - LOWER_BOUNDS
TOTAL_EXTRA_BUDGET = MAX_OUTER_RADIUS - float(LOWER_BOUNDS.sum())


def cumulative_to_geometry(radii: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii)
    if radii.ndim != 2 or radii.shape[1] != N_LAYERS:
        raise ValueError(f"radii must have shape (N,{N_LAYERS}).")
    geometry = np.empty_like(radii)
    geometry[:, 0] = radii[:, 0]
    geometry[:, 1:] = np.diff(radii, axis=1)
    return geometry


def geometry_to_cumulative(geometry: np.ndarray) -> np.ndarray:
    geometry = np.asarray(geometry)
    if geometry.ndim != 2 or geometry.shape[1] != N_LAYERS:
        raise ValueError(f"geometry must have shape (N,{N_LAYERS}).")
    return np.cumsum(geometry, axis=1)


def geometry_bounds_mask(
    geometry: np.ndarray,
    atol: float = GEOMETRY_CHECK_ATOL_NM,
) -> np.ndarray:
    values = np.asarray(geometry, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != N_LAYERS:
        raise ValueError(f"geometry must have shape (N,{N_LAYERS}).")
    finite = np.isfinite(values).all(axis=1)
    lower_ok = np.all(values >= LOWER_BOUNDS[None, :] - atol, axis=1)
    upper_ok = np.all(values <= UPPER_BOUNDS[None, :] + atol, axis=1)
    outer_ok = values.sum(axis=1) <= MAX_OUTER_RADIUS + atol
    return finite & lower_ok & upper_ok & outer_ok


def assert_legal_geometry(geometry: np.ndarray, label: str = "geometry") -> None:
    mask = geometry_bounds_mask(geometry)
    if not np.all(mask):
        bad = np.flatnonzero(~mask)
        raise ValueError(
            f"{label} contains {len(bad)} illegal rows; first rows={bad[:10].tolist()}."
        )


def geometry_from_raw(raw: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Map unconstrained values to the exact legal continuous geometry domain.

    1. A sigmoid maps each coordinate into its independent lower/upper range.
    2. If the total extra radius exceeds the 120 nm shared budget, all four
       increments are scaled proportionally. This preserves positivity and the
       individual upper bounds while enforcing outer radius <= 300 nm.

    The map is differentiable almost everywhere and never relies on a penalty
    term to make an illegal geometry acceptable.
    """
    if raw.shape[-1] != N_LAYERS:
        raise ValueError(f"raw must end with dimension {N_LAYERS}.")
    lower = torch.as_tensor(LOWER_BOUNDS, dtype=raw.dtype, device=raw.device)
    capacity = torch.as_tensor(EXTRA_CAPACITY, dtype=raw.dtype, device=raw.device)
    safe_budget = TOTAL_EXTRA_BUDGET - GEOMETRY_INTERIOR_MARGIN_NM
    if safe_budget <= 0.0:
        raise RuntimeError("Geometry interior margin consumes the entire shared budget.")
    budget = torch.as_tensor(safe_budget, dtype=raw.dtype, device=raw.device)

    increments = capacity * torch.sigmoid(raw)
    increment_sum = increments.sum(dim=-1, keepdim=True)
    scale = torch.clamp(budget / increment_sum.clamp_min(float(eps)), max=1.0)
    geometry = lower + increments * scale
    return geometry


def raw_from_geometry(geometry: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Approximate inverse of :func:`geometry_from_raw` for legal geometries."""
    if geometry.shape[-1] != N_LAYERS:
        raise ValueError(f"geometry must end with dimension {N_LAYERS}.")
    lower = torch.as_tensor(LOWER_BOUNDS, dtype=geometry.dtype, device=geometry.device)
    capacity = torch.as_tensor(EXTRA_CAPACITY, dtype=geometry.dtype, device=geometry.device)
    fraction = ((geometry - lower) / capacity).clamp(float(eps), 1.0 - float(eps))
    return torch.log(fraction) - torch.log1p(-fraction)


def torch_geometry_is_legal(
    geometry: torch.Tensor,
    atol: float = GEOMETRY_CHECK_ATOL_NM,
) -> torch.Tensor:
    lower = torch.as_tensor(LOWER_BOUNDS, dtype=geometry.dtype, device=geometry.device)
    upper = torch.as_tensor(UPPER_BOUNDS, dtype=geometry.dtype, device=geometry.device)
    finite = torch.isfinite(geometry).all(dim=-1)
    lower_ok = (geometry >= lower - atol).all(dim=-1)
    upper_ok = (geometry <= upper + atol).all(dim=-1)
    outer_ok = geometry.sum(dim=-1) <= MAX_OUTER_RADIUS + atol
    return finite & lower_ok & upper_ok & outer_ok


def find_air_index(material_names: Sequence[str]) -> int:
    normalized = [str(name).strip().lower() for name in material_names]
    matches = [index for index, name in enumerate(normalized) if name == "air"]
    if len(matches) != 1:
        raise ValueError("material_names must contain exactly one material named 'Air'.")
    return matches[0]


def material_sequence_mask(
    material_indices: np.ndarray,
    material_names: Sequence[str],
) -> np.ndarray:
    indices = np.asarray(material_indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices[None, :]
    if indices.ndim != 2 or indices.shape[1] != N_LAYERS:
        raise ValueError(f"material_indices must have shape (N,{N_LAYERS}).")
    n_materials = len(material_names)
    air = find_air_index(material_names)
    return (
        np.all(indices >= 0, axis=1)
        & np.all(indices < n_materials, axis=1)
        & np.all(indices[:, 1:] != air, axis=1)
        & np.all(indices[:, 1:] != indices[:, :-1], axis=1)
    )


def enumerate_legal_material_sequences(material_names: Sequence[str]) -> pd.DataFrame:
    names = np.asarray(material_names, dtype=str)
    all_sequences = np.asarray(
        list(product(range(len(names)), repeat=N_LAYERS)), dtype=np.int64
    )
    legal = all_sequences[material_sequence_mask(all_sequences, names)]
    frame = pd.DataFrame(legal, columns=MATERIAL_COLUMNS)
    frame.insert(0, "sequence_id", np.arange(len(frame), dtype=np.int64))
    frame["materials"] = ["|".join(names[row].tolist()) for row in legal]
    if len(names) == 6 and len(frame) != 400:
        raise RuntimeError(f"Expected 400 legal sequences for six materials, got {len(frame)}.")
    return frame


def self_test() -> None:
    raw = torch.randn(4096, 4, dtype=torch.float64, requires_grad=True)
    geometry = geometry_from_raw(raw)
    if not bool(torch_geometry_is_legal(geometry).all()):
        raise AssertionError("Differentiable geometry map produced an illegal geometry.")
    loss = geometry.square().mean()
    loss.backward()
    if raw.grad is None or not torch.isfinite(raw.grad).all():
        raise AssertionError("Geometry parameterization gradient is invalid.")

    names = np.asarray(["Al2O3", "BaSO4", "SiO2", "TiO2", "PDMS", "Air"])
    sequences = enumerate_legal_material_sequences(names)
    if len(sequences) != 400:
        raise AssertionError("Material sequence enumeration failed.")


if __name__ == "__main__":
    self_test()
    print("geometry_parameterization self-test passed.")
