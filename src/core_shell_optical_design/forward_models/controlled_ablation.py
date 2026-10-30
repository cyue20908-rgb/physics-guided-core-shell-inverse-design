"""Frozen definitions for the parameter-matched PGFF ablation study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .physics_guided_fourier_network import PGFFNet, PHYSICS_FEATURE_NAMES


@dataclass(frozen=True)
class AblationVariant:
    """Information and conditioning switches for one controlled variant."""

    name: str
    coordinate_mode: str
    use_physics_features: bool
    conditioning_mode: str
    lambda_curve: float
    role: str


def variants_from_protocol(protocol: Mapping) -> dict[str, AblationVariant]:
    """Validate and materialize the variants in a frozen protocol mapping."""
    variants: dict[str, AblationVariant] = {}
    for name, payload in protocol["variants"].items():
        variants[name] = AblationVariant(
            name=name,
            coordinate_mode=str(payload["coordinate_mode"]),
            use_physics_features=bool(payload["use_physics_features"]),
            conditioning_mode=str(payload["conditioning_mode"]),
            lambda_curve=float(payload["lambda_curve"]),
            role=str(payload["role"]),
        )
    return variants


def build_ablation_model(
    *,
    variant: AblationVariant,
    geometry_mean: torch.Tensor,
    geometry_std: torch.Tensor,
    wavelengths_nm: torch.Tensor,
    material_n: torch.Tensor,
    material_k: torch.Tensor,
    physics_mean: torch.Tensor,
    physics_std: torch.Tensor,
    model_config: Mapping,
    fourier_seed: int,
) -> PGFFNet:
    """Build one variant while holding every registered module constant."""
    return PGFFNet(
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
        wavelengths_nm=wavelengths_nm,
        material_n=material_n,
        material_k=material_k,
        physics_mean=physics_mean,
        physics_std=physics_std,
        hidden_dim=int(model_config["hidden_dim"]),
        n_fourier_frequencies=int(model_config["fourier_frequencies"]),
        fourier_scale=float(model_config["fourier_scale"]),
        fourier_seed=int(fourier_seed),
        dropout=float(model_config["dropout"]),
        n_structure_blocks=int(model_config["n_structure_blocks"]),
        n_wavelength_blocks=int(model_config["n_wavelength_blocks"]),
        n_physics_blocks=int(model_config["n_physics_blocks"]),
        n_fusion_blocks=int(model_config["n_fusion_blocks"]),
        coordinate_mode=variant.coordinate_mode,
        use_physics_features=variant.use_physics_features,
        conditioning_mode=variant.conditioning_mode,
    )


def parameter_count(model: torch.nn.Module) -> int:
    """Return the registered trainable parameter count used for fairness."""
    return sum(parameter.numel() for parameter in model.parameters())


def input_informed_parameter_count(
    model: PGFFNet, variant: AblationVariant
) -> int:
    """Count parameters connected to a nonzero information-bearing input.

    Registered counts are exactly identical.  This secondary conservative
    audit subtracts first-layer weights multiplied by deterministic zero masks
    in the raw-coordinate and absent-physics controls.
    """
    count = parameter_count(model)
    if variant.coordinate_mode == "raw_k0":
        dormant_coordinate_columns = model.fourier.output_dim - 1
        count -= dormant_coordinate_columns * model.hidden_dim
    if not variant.use_physics_features:
        count -= len(PHYSICS_FEATURE_NAMES) * model.hidden_dim
    return count
