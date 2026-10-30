"""Physics-guided Fourier-feature forward surrogate for four-layer spheres.

The model predicts Q_sca and Q_abs at every wavelength and constructs
Q_ext = Q_sca + Q_abs exactly.  It uses the same global geometry/material
representation as the direct-spectrum MLP baseline, while adding only
explicit wavelength coordinates and input-derived Mie features.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PHYSICS_FEATURE_NAMES = (
    "x1", "x2", "x3", "x4",
    "n1", "n2", "n3", "n4",
    "k1", "k2", "k3", "k4",
    "dn12", "dn23", "dn34", "dn4_env",
    "dk12", "dk23", "dk34", "dk4_env",
    "phi1", "phi2", "phi3", "phi4",
)

COORDINATE_MODES = ("raw_k0", "fourier")
CONDITIONING_MODES = ("additive", "film")


class ResidualMLPBlock(nn.Module):
    """Pre-normalized residual MLP block."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class FixedFourierEncoder(nn.Module):
    """Fixed, reproducible random Fourier features for normalized k0."""

    def __init__(
        self,
        n_frequencies: int = 32,
        scale: float = 3.0,
        seed: int = 2026,
    ):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        frequencies = torch.randn(
            1, n_frequencies, generator=generator, dtype=torch.float32
        ) * float(scale)
        self.register_buffer("frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        return 2 * self.frequencies.shape[1]

    def forward(self, coordinate: torch.Tensor) -> torch.Tensor:
        # coordinate: (..., 1), normalized inverse wavelength / wave number.
        projected = 2.0 * math.pi * coordinate @ self.frequencies
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)


class Encoder(nn.Module):
    """Projection followed by residual MLP blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input(x))


class PositiveHead(nn.Module):
    """Small regression head with a non-negative Softplus output."""

    def __init__(self, hidden_dim: int, initial_value: float):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim // 2, 1)
        self.softplus = nn.Softplus()

        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        inverse_softplus = math.log(math.expm1(float(initial_value)))
        nn.init.constant_(self.output.bias, inverse_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.output(self.hidden(x))).squeeze(-1)


class PGFFNet(nn.Module):
    """Physics-Guided Fourier Feature Network.

    Parameters are registered with the training wavelength/material tables.
    For a new wavelength grid, construct a new model with interpolated n/k
    tables and load the learned weights with strict=False only after checking
    compatible architecture; the learned Fourier matrix is grid-independent.
    """

    def __init__(
        self,
        geometry_mean: torch.Tensor,
        geometry_std: torch.Tensor,
        wavelengths_nm: torch.Tensor,
        material_n: torch.Tensor,
        material_k: torch.Tensor,
        physics_mean: torch.Tensor,
        physics_std: torch.Tensor,
        hidden_dim: int = 256,
        n_fourier_frequencies: int = 32,
        fourier_scale: float = 3.0,
        fourier_seed: int = 2026,
        dropout: float = 0.0,
        n_structure_blocks: int = 2,
        n_wavelength_blocks: int = 2,
        n_physics_blocks: int = 2,
        n_fusion_blocks: int = 2,
        environment_n: float = 1.0,
        environment_k: float = 0.0,
        coordinate_mode: str = "fourier",
        use_physics_features: bool = True,
        conditioning_mode: str = "film",
    ):
        super().__init__()

        if coordinate_mode not in COORDINATE_MODES:
            raise ValueError(
                f"coordinate_mode must be one of {COORDINATE_MODES}, "
                f"got {coordinate_mode!r}."
            )
        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning_mode must be one of {CONDITIONING_MODES}, "
                f"got {conditioning_mode!r}."
            )

        geometry_mean = torch.as_tensor(geometry_mean, dtype=torch.float32)
        geometry_std = torch.as_tensor(geometry_std, dtype=torch.float32)
        wavelengths_nm = torch.as_tensor(wavelengths_nm, dtype=torch.float32)
        material_n = torch.as_tensor(material_n, dtype=torch.float32)
        material_k = torch.as_tensor(material_k, dtype=torch.float32)
        physics_mean = torch.as_tensor(physics_mean, dtype=torch.float32)
        physics_std = torch.as_tensor(physics_std, dtype=torch.float32)

        if geometry_mean.shape != (4,) or geometry_std.shape != (4,):
            raise ValueError("geometry_mean/std must each have shape (4,).")
        if material_n.shape != material_k.shape:
            raise ValueError("material_n and material_k must have identical shape.")
        if material_n.ndim != 2 or material_n.shape[1] != wavelengths_nm.numel():
            raise ValueError("material tables must have shape (N_material, N_wavelength).")
        if physics_mean.shape != (len(PHYSICS_FEATURE_NAMES),):
            raise ValueError("physics_mean has the wrong shape.")
        if physics_std.shape != physics_mean.shape:
            raise ValueError("physics_std has the wrong shape.")

        self.hidden_dim = int(hidden_dim)
        self.n_materials = int(material_n.shape[0])
        self.n_wavelengths = int(wavelengths_nm.numel())
        self.environment_n = float(environment_n)
        self.environment_k = float(environment_k)
        self.coordinate_mode = str(coordinate_mode)
        self.use_physics_features = bool(use_physics_features)
        self.conditioning_mode = str(conditioning_mode)

        self.register_buffer("geometry_mean", geometry_mean)
        self.register_buffer("geometry_std", geometry_std.clamp_min(1e-8))
        self.register_buffer("wavelengths_nm", wavelengths_nm)
        self.register_buffer("material_n", material_n)
        self.register_buffer("material_k", material_k)
        self.register_buffer("physics_mean", physics_mean)
        self.register_buffer("physics_std", physics_std.clamp_min(1e-8))

        k0 = 2.0 * math.pi / wavelengths_nm
        self.register_buffer("k0", k0)
        self.register_buffer("k0_min", k0.min())
        self.register_buffer("k0_max", k0.max())

        structure_input_dim = 4 + 4 * self.n_materials
        self.structure_encoder = Encoder(
            structure_input_dim, hidden_dim, n_structure_blocks, dropout
        )

        self.fourier = FixedFourierEncoder(
            n_frequencies=n_fourier_frequencies,
            scale=fourier_scale,
            seed=fourier_seed,
        )
        self.wavelength_encoder = Encoder(
            self.fourier.output_dim, hidden_dim, n_wavelength_blocks, dropout
        )
        self.physics_encoder = Encoder(
            len(PHYSICS_FEATURE_NAMES), hidden_dim, n_physics_blocks, dropout
        )

        self.film = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fusion_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(n_fusion_blocks)]
        )

        self.q_sca_head = PositiveHead(hidden_dim, initial_value=3.0)
        self.q_abs_head = PositiveHead(hidden_dim, initial_value=0.05)

    @staticmethod
    def geometry_to_cumulative_radii(
        geometry_nm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[r_core,t1,t2,t3] -> cumulative radii and layer thicknesses."""
        if geometry_nm.ndim != 2 or geometry_nm.shape[1] != 4:
            raise ValueError("geometry_nm must have shape (batch, 4).")
        cumulative_radii = torch.cumsum(geometry_nm, dim=-1)
        return cumulative_radii, geometry_nm

    def build_physics_features(
        self,
        geometry_nm: torch.Tensor,
        material_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return pointwise Mie-inspired features, shape (B, W, 24)."""
        if material_indices.shape != geometry_nm.shape:
            raise ValueError("material_indices must have shape (batch, 4).")
        if material_indices.min() < 0 or material_indices.max() >= self.n_materials:
            raise ValueError("material index out of range.")

        radii_nm, thickness_nm = self.geometry_to_cumulative_radii(geometry_nm)
        batch_size = geometry_nm.shape[0]

        # Advanced indexing gives (B, 4, W); transpose to (B, W, 4).
        n_layers = self.material_n[material_indices].permute(0, 2, 1)
        k_layers = self.material_k[material_indices].permute(0, 2, 1)

        k0 = self.k0.view(1, self.n_wavelengths, 1)
        radii = radii_nm.view(batch_size, 1, 4)
        thickness = thickness_nm.view(batch_size, 1, 4)

        size_parameters = k0 * radii

        dn_internal = n_layers[..., :-1] - n_layers[..., 1:]
        dn_outer = n_layers[..., -1:] - self.environment_n
        delta_n = torch.cat((dn_internal, dn_outer), dim=-1)

        dk_internal = k_layers[..., :-1] - k_layers[..., 1:]
        dk_outer = k_layers[..., -1:] - self.environment_k
        delta_k = torch.cat((dk_internal, dk_outer), dim=-1)

        optical_phase = k0 * n_layers * thickness

        features = torch.cat(
            (
                size_parameters,
                n_layers,
                k_layers,
                delta_n,
                delta_k,
                optical_phase,
            ),
            dim=-1,
        )
        return (features - self.physics_mean.view(1, 1, -1)) / self.physics_std.view(
            1, 1, -1
        )

    def forward(
        self,
        geometry_nm: torch.Tensor,
        material_indices: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> Dict[str, torch.Tensor]:
        geometry_nm = geometry_nm.float()
        material_indices = material_indices.long()
        batch_size = geometry_nm.shape[0]

        geometry_normalized = (geometry_nm - self.geometry_mean) / self.geometry_std
        material_one_hot = F.one_hot(
            material_indices, num_classes=self.n_materials
        ).float().reshape(batch_size, -1)
        structure_input = torch.cat((geometry_normalized, material_one_hot), dim=-1)
        structure_features = self.structure_encoder(structure_input)

        k0_normalized = (self.k0 - self.k0_min) / (self.k0_max - self.k0_min)
        if self.coordinate_mode == "fourier":
            coordinate_features = self.fourier(k0_normalized.view(-1, 1))
        else:
            # Zero-pad the single raw coordinate to preserve the exact input
            # width and registered parameter count of the Fourier branch.
            # No sinusoidal, polynomial, or duplicate wavelength channel is
            # added; only column zero carries information.
            coordinate_features = torch.zeros(
                self.n_wavelengths,
                self.fourier.output_dim,
                device=k0_normalized.device,
                dtype=k0_normalized.dtype,
            )
            coordinate_features[:, 0] = k0_normalized
        wavelength_features = self.wavelength_encoder(coordinate_features)
        wavelength_features = wavelength_features.unsqueeze(0).expand(batch_size, -1, -1)

        gamma, beta = self.film(structure_features).chunk(2, dim=-1)
        if self.conditioning_mode == "film":
            conditioned_wavelength = (
                (1.0 + gamma.unsqueeze(1)) * wavelength_features
                + beta.unsqueeze(1)
            )
        else:
            # Parameter-matched non-FiLM control: retain structure
            # conditioning and use every parameter in ``self.film``, but
            # remove the multiplicative structure-wavelength interaction.
            additive_structure = (gamma + beta) / math.sqrt(2.0)
            conditioned_wavelength = (
                wavelength_features + additive_structure.unsqueeze(1)
            )

        if self.use_physics_features:
            pointwise_context = self.build_physics_features(
                geometry_nm, material_indices
            )
        else:
            # Deterministic mask for the absent-physics control.  The module
            # topology and registered parameter count stay unchanged, but no
            # radii, n/k, contrast, phase, or duplicate wavelength signal can
            # enter through this branch.
            pointwise_context = torch.zeros(
                batch_size,
                self.n_wavelengths,
                len(PHYSICS_FEATURE_NAMES),
                device=geometry_nm.device,
                dtype=geometry_nm.dtype,
            )
        physics_features = self.physics_encoder(pointwise_context)

        fused = torch.cat((conditioned_wavelength, physics_features), dim=-1)
        fused = self.fusion_projection(fused)
        fused = self.fusion_blocks(fused)

        q_sca = self.q_sca_head(fused)
        q_abs = self.q_abs_head(fused)
        q_ext = q_sca + q_abs

        result = {"q_sca": q_sca, "q_abs": q_abs, "q_ext": q_ext}
        if return_auxiliary:
            result.update(
                {
                    "structure_features": structure_features,
                    "pointwise_context": pointwise_context,
                    # Backward-compatible key retained for existing callers.
                    "physics_features": pointwise_context,
                    "fused_features": fused,
                }
            )
        return result

