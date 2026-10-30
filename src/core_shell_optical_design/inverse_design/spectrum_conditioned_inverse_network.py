# -*- coding: utf-8 -*-
"""Inverse network architecture for the new frozen-PGFF tandem workflow."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_shell_optical_design.inverse_design.geometry_parameterization import (
    geometry_from_raw,
    raw_from_geometry,
)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width * 2)
        self.linear2 = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return residual + x


class SpectralMaterialInverseNet(nn.Module):
    """Map target Qsca + one legal material sequence to a continuous geometry.

    A single output per material sequence is sufficient for multi-solution
    discovery because strict blind inference enumerates all 400 legal sequences.
    """

    def __init__(
        self,
        *,
        n_wavelengths: int,
        n_materials: int,
        qsca_mean: torch.Tensor,
        qsca_std: torch.Tensor,
        width: int = 384,
        depth: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if qsca_mean.shape != (n_wavelengths,) or qsca_std.shape != (n_wavelengths,):
            raise ValueError("qsca_mean/qsca_std must match n_wavelengths.")
        if torch.any(qsca_std <= 0.0):
            raise ValueError("qsca_std must be strictly positive.")
        self.n_wavelengths = int(n_wavelengths)
        self.n_materials = int(n_materials)
        self.register_buffer("qsca_mean", qsca_mean.detach().float().clone())
        self.register_buffer("qsca_std", qsca_std.detach().float().clone())

        input_dim = self.n_wavelengths + 4 * self.n_materials
        self.input_layer = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(width, dropout=dropout) for _ in range(int(depth))]
        )
        self.output_norm = nn.LayerNorm(width)
        self.geometry_head = nn.Linear(width, 4)

        # Start from an interior legal geometry rather than the Rout=300 boundary.
        # [160, 40, 40, 40] nm has Rout=280 nm and leaves room for gradients
        # to move both inward and outward during the first optimization steps.
        nominal_geometry = torch.tensor([[160.0, 40.0, 40.0, 40.0]])
        nominal_raw = raw_from_geometry(nominal_geometry).squeeze(0)
        with torch.no_grad():
            self.geometry_head.bias.copy_(nominal_raw)
        nn.init.normal_(self.geometry_head.weight, mean=0.0, std=1.0e-3)

    def forward(
        self,
        target_qsca: torch.Tensor,
        material_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if target_qsca.ndim != 2 or target_qsca.shape[1] != self.n_wavelengths:
            raise ValueError(
                f"target_qsca must have shape (B,{self.n_wavelengths})."
            )
        if material_indices.ndim != 2 or material_indices.shape[1] != 4:
            raise ValueError("material_indices must have shape (B,4).")
        if material_indices.shape[0] != target_qsca.shape[0]:
            raise ValueError("target_qsca and material_indices batch sizes differ.")
        if material_indices.min() < 0 or material_indices.max() >= self.n_materials:
            raise ValueError("material_indices contains an out-of-range value.")

        normalized_spectrum = (target_qsca.float() - self.qsca_mean) / self.qsca_std
        material_one_hot = F.one_hot(
            material_indices.long(), num_classes=self.n_materials
        ).float().reshape(len(material_indices), -1)
        x = torch.cat((normalized_spectrum, material_one_hot), dim=1)
        x = F.gelu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        raw_geometry = self.geometry_head(self.output_norm(x))
        geometry = geometry_from_raw(raw_geometry)
        return {"geometry": geometry, "raw_geometry": raw_geometry}
