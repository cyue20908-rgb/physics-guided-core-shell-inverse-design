# -*- coding: utf-8 -*-
"""Spectral objectives and evaluation metrics for tandem inverse training.

The training objective is deliberately smooth (normalized Smooth-L1), while
model selection and reporting use the frozen physical metrics NMAE, NRMSE,
R², and peak-wavelength error. All normalizations are per target spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BatchMetricSums:
    sample_count: int
    nmae_sum: float
    nrmse_sum: float
    r2_sum: float
    r2_count: int
    peak_error_sum_nm: float
    geometry_mae_sum_nm: float


def target_scale(target_qsca: torch.Tensor, floor: float = 1.0e-6) -> torch.Tensor:
    """Return one positive normalization value per target spectrum."""
    if target_qsca.ndim != 2:
        raise ValueError("target_qsca must have shape (B,L).")
    return target_qsca.amax(dim=1, keepdim=True).clamp_min(float(floor))


def normalized_smooth_l1_loss(
    prediction_qsca: torch.Tensor,
    target_qsca: torch.Tensor,
    *,
    beta: float = 0.01,
    scale_floor: float = 1.0e-6,
) -> torch.Tensor:
    """Smooth cycle-consistency loss normalized by each target peak."""
    if prediction_qsca.shape != target_qsca.shape:
        raise ValueError("prediction_qsca and target_qsca shapes differ.")
    scale = target_scale(target_qsca, scale_floor)
    return F.smooth_l1_loss(
        prediction_qsca / scale,
        target_qsca / scale,
        beta=float(beta),
        reduction="mean",
    )


def normalized_geometry_anchor_loss(
    prediction_geometry: torch.Tensor,
    target_geometry: torch.Tensor,
) -> torch.Tensor:
    """Scale-free geometry anchor used only during the early training phase."""
    if prediction_geometry.shape != target_geometry.shape:
        raise ValueError("prediction_geometry and target_geometry shapes differ.")
    ranges = torch.as_tensor(
        [100.0, 60.0, 60.0, 60.0],
        dtype=prediction_geometry.dtype,
        device=prediction_geometry.device,
    )
    return F.smooth_l1_loss(
        prediction_geometry / ranges,
        target_geometry / ranges,
        beta=0.02,
        reduction="mean",
    )


def per_sample_metrics(
    prediction_qsca: torch.Tensor,
    target_qsca: torch.Tensor,
    wavelengths_nm: torch.Tensor,
    *,
    scale_floor: float = 1.0e-6,
) -> Dict[str, torch.Tensor]:
    """Compute frozen reporting metrics without reducing over the batch."""
    if prediction_qsca.shape != target_qsca.shape:
        raise ValueError("prediction_qsca and target_qsca shapes differ.")
    if wavelengths_nm.ndim != 1 or wavelengths_nm.numel() != target_qsca.shape[1]:
        raise ValueError("wavelengths_nm does not match the spectral dimension.")

    residual = prediction_qsca - target_qsca
    scale = target_scale(target_qsca, scale_floor).squeeze(1)
    nmae = residual.abs().mean(dim=1) / scale
    nrmse = residual.square().mean(dim=1).sqrt() / scale

    centered = target_qsca - target_qsca.mean(dim=1, keepdim=True)
    ss_res = residual.square().sum(dim=1)
    ss_tot = centered.square().sum(dim=1)
    valid_r2 = ss_tot > 1.0e-12
    r2 = torch.full_like(ss_res, torch.nan)
    r2[valid_r2] = 1.0 - ss_res[valid_r2] / ss_tot[valid_r2]

    target_peak_index = target_qsca.argmax(dim=1)
    prediction_peak_index = prediction_qsca.argmax(dim=1)
    target_peak_nm = wavelengths_nm[target_peak_index]
    prediction_peak_nm = wavelengths_nm[prediction_peak_index]
    peak_error_nm = (prediction_peak_nm - target_peak_nm).abs()
    return {
        "nmae": nmae,
        "nrmse": nrmse,
        "r2": r2,
        "peak_error_nm": peak_error_nm,
    }
