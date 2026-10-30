# -*- coding: utf-8 -*-
"""Train the wavenumber-conditioned MLP comparison model.

The model uses the same structure descriptor as the direct-spectrum MLP:

    - geometry: r_core, t1, t2, t3
    - materials: four layer-wise one-hot vectors

For every wavelength, it additionally receives the raw normalized vacuum
wavenumber k0 = 2*pi/lambda.  It does not use Fourier features, material n/k,
Mie-inspired physics features, auxiliary Q_abs/Q_ext tasks, or physical loss
constraints.  A shared pointwise MLP predicts Q_sca at every wavelength and
assembles the result into a full spectrum of shape (batch, n_wavelengths).

For a controlled ablation, train/validation/test indices and geometry
normalization are loaded from the direct-spectrum MLP result directory.
Output prediction and metric keys match the direct-spectrum model interface so
downstream evaluation remains consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core_shell_optical_design.project_paths import (
    DATASET_PATH,
    DIRECT_MODEL_DIR,
    REPRODUCED_RESULT_DIR,
)


# ==================== Default paths ====================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = DATASET_PATH
DEFAULT_DIRECT_MLP_DIR = DIRECT_MODEL_DIR
LEGACY_DIRECT_MLP_DIR = DIRECT_MODEL_DIR
DEFAULT_OUTPUT_DIR = (
    REPRODUCED_RESULT_DIR
    / "forward_models"
    / "wavenumber_conditioned_network"
)


# ==================== Arguments ====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a raw-k0 wavelength-conditioned MLP without Fourier features."
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--split_dir",
        type=Path,
        default=DEFAULT_DIRECT_MLP_DIR,
        help=(
            "Direct-spectrum MLP result directory containing idx_train.npy, "
            "idx_val.npy, idx_test.npy, and normalization.npz."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--scheduler_patience", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable automatic mixed precision on CUDA.",
    )
    return parser.parse_args()


# ==================== Reproducibility and I/O ====================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RawK0MLP")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=json_default)
    os.replace(temporary, path)


# ==================== Data preparation ====================

def load_config(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "config" in data:
        return data["config"].item()
    if "config_json" in data:
        return json.loads(str(data["config_json"].item()))
    raise KeyError("Dataset is missing config/config_json.")


def cumulative_radii_to_geometry(radii: np.ndarray) -> np.ndarray:
    """Convert cumulative radii [r1,r2,r3,r4] to [r_core,t1,t2,t3]."""
    if radii.ndim != 2 or radii.shape[1] != 4:
        raise ValueError(f"radii must have shape (N, 4), got {radii.shape}.")
    geometry = np.empty_like(radii, dtype=np.float32)
    geometry[:, 0] = radii[:, 0]
    geometry[:, 1:] = np.diff(radii, axis=1)
    return geometry


def build_one_hot(material_indices: np.ndarray, n_materials: int) -> np.ndarray:
    if material_indices.ndim != 2 or material_indices.shape[1] != 4:
        raise ValueError("material_indices must have shape (N, 4).")
    if material_indices.min() < 0 or material_indices.max() >= n_materials:
        raise ValueError("material_indices contains an out-of-range material index.")
    one_hot = np.eye(n_materials, dtype=np.float32)[material_indices]
    return one_hot.reshape(len(material_indices), 4 * n_materials)


def resolve_split_dir(requested: Path, logger: logging.Logger) -> Path:
    if requested.exists():
        return requested
    if requested == DEFAULT_DIRECT_MLP_DIR and LEGACY_DIRECT_MLP_DIR.exists():
        logger.warning(
            "Default direct-spectrum directory was not found; using legacy directory: %s",
            LEGACY_DIRECT_MLP_DIR,
        )
        return LEGACY_DIRECT_MLP_DIR
    raise FileNotFoundError(f"Direct-spectrum split directory not found: {requested}")


def validate_indices(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
) -> None:
    arrays = (train_indices, val_indices, test_indices)
    names = ("train", "validation", "test")
    for name, values in zip(names, arrays):
        if values.ndim != 1:
            raise ValueError(f"{name} indices must be one-dimensional.")
        if len(values) == 0:
            raise ValueError(f"{name} split is empty.")
        if values.min() < 0 or values.max() >= n_samples:
            raise ValueError(f"{name} split contains an out-of-range index.")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"{name} split contains duplicate indices.")

    train_set, val_set, test_set = map(set, (x.tolist() for x in arrays))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Train/validation/test splits overlap.")

    combined = np.concatenate(arrays)
    if len(combined) != n_samples or len(np.unique(combined)) != n_samples:
        raise RuntimeError("Saved splits do not cover the dataset exactly once.")


def normalize_k0(wavelengths_nm: np.ndarray) -> Tuple[np.ndarray, float, float]:
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    if wavelengths_nm.ndim != 1 or len(wavelengths_nm) < 2:
        raise ValueError("wavelengths must be a one-dimensional grid with at least two points.")
    if not np.isfinite(wavelengths_nm).all() or np.any(wavelengths_nm <= 0.0):
        raise ValueError("wavelengths must be finite and strictly positive.")
    if np.any(np.diff(wavelengths_nm) <= 0.0):
        raise ValueError("wavelengths must be strictly increasing.")

    k0 = 2.0 * np.pi / wavelengths_nm
    k0_min = float(k0.min())
    k0_max = float(k0.max())
    if not k0_max > k0_min:
        raise ValueError("The wavelength grid has zero k0 range.")
    normalized = (k0 - k0_min) / (k0_max - k0_min)
    return normalized.astype(np.float32), k0_min, k0_max


def load_and_prepare_data(
    data_path: Path,
    split_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    split_dir = resolve_split_dir(split_dir, logger)
    required_split_files = (
        "idx_train.npy",
        "idx_val.npy",
        "idx_test.npy",
        "normalization.npz",
    )
    for filename in required_split_files:
        path = split_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required direct-spectrum artifact not found: {path}")

    logger.info("Loading dataset: %s", data_path)
    logger.info("Reusing direct-spectrum split and normalization: %s", split_dir)

    with np.load(data_path, allow_pickle=True) as data:
        for key in ("radii", "material_indices", "wavelengths", "Q_sca"):
            if key not in data:
                raise KeyError(f"Dataset is missing required field: {key}")
        radii = data["radii"].astype(np.float32)
        material_indices = data["material_indices"].astype(np.int64)
        wavelengths = data["wavelengths"].astype(np.float32)
        q_sca = data["Q_sca"].astype(np.float32)
        config = load_config(data)

    materials = list(config["materials"])
    n_materials = len(materials)
    n_samples = len(radii)
    n_wavelengths = len(wavelengths)

    if material_indices.shape != (n_samples, 4):
        raise ValueError(f"material_indices has wrong shape: {material_indices.shape}")
    if q_sca.shape != (n_samples, n_wavelengths):
        raise ValueError(f"Q_sca has wrong shape: {q_sca.shape}")
    if not all(np.isfinite(x).all() for x in (radii, wavelengths, q_sca)):
        raise ValueError("Dataset contains NaN or Inf.")
    if q_sca.min() < -1e-6:
        raise ValueError("Q_sca contains clearly negative values.")

    geometry = cumulative_radii_to_geometry(radii)
    if np.any(geometry <= 0.0):
        raise ValueError("Converted core radius and shell thicknesses must be positive.")
    train_indices = np.load(split_dir / "idx_train.npy").astype(np.int64)
    val_indices = np.load(split_dir / "idx_val.npy").astype(np.int64)
    test_indices = np.load(split_dir / "idx_test.npy").astype(np.int64)
    validate_indices(train_indices, val_indices, test_indices, n_samples)

    with np.load(split_dir / "normalization.npz", allow_pickle=False) as norm:
        for key in ("geometry_mean", "geometry_std", "wavelengths"):
            if key not in norm:
                raise KeyError(f"Direct-spectrum normalization is missing: {key}")
        geometry_mean = norm["geometry_mean"].astype(np.float32)
        geometry_std = norm["geometry_std"].astype(np.float32)
        reference_wavelengths = norm["wavelengths"].astype(np.float32)

    if geometry_mean.shape != (4,) or geometry_std.shape != (4,):
        raise ValueError("geometry_mean and geometry_std must each have shape (4,).")
    if np.any(geometry_std <= 0.0):
        raise ValueError("geometry_std must be strictly positive.")
    if not np.array_equal(wavelengths, reference_wavelengths):
        raise RuntimeError("Dataset and direct-spectrum wavelength grids differ.")

    recomputed_mean = geometry[train_indices].mean(axis=0)
    recomputed_std = np.maximum(geometry[train_indices].std(axis=0), 1e-8)
    if not np.allclose(geometry_mean, recomputed_mean, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Stored geometry_mean is inconsistent with the training split.")
    if not np.allclose(geometry_std, recomputed_std, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Stored geometry_std is inconsistent with the training split.")

    geometry_normalized = (geometry - geometry_mean) / geometry_std
    material_one_hot = build_one_hot(material_indices, n_materials)
    structure_features = np.concatenate(
        (geometry_normalized, material_one_hot), axis=1
    ).astype(np.float32)
    expected_structure_dim = 4 + 4 * n_materials
    if structure_features.shape != (n_samples, expected_structure_dim):
        raise RuntimeError("Unexpected structure-feature shape.")

    k0_normalized, k0_min, k0_max = normalize_k0(wavelengths)

    # Preserve the original output interface while recording the new coordinate.
    np.save(output_dir / "idx_train.npy", train_indices)
    np.save(output_dir / "idx_val.npy", val_indices)
    np.save(output_dir / "idx_test.npy", test_indices)
    np.savez(
        output_dir / "normalization.npz",
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
        geometry_feature_names=np.asarray(
            ["r_core", "shell1_thickness", "shell2_thickness", "shell3_thickness"]
        ),
        material_names=np.asarray(materials),
        wavelengths=wavelengths,
        wavelength_coordinate=np.asarray("normalized_k0"),
        k0_normalized=k0_normalized,
        k0_min=np.asarray(k0_min, dtype=np.float64),
        k0_max=np.asarray(k0_max, dtype=np.float64),
    )

    split_info = {
        "n_total": int(n_samples),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(val_indices)),
        "n_test": int(len(test_indices)),
        "train_fraction": float(len(train_indices) / n_samples),
        "validation_fraction": float(len(val_indices) / n_samples),
        "test_fraction": float(len(test_indices) / n_samples),
        "stratification": "four-layer material sequence",
        "split_policy": "exactly reused from direct-spectrum MLP",
        "split_source_dir": str(split_dir),
    }
    save_json(output_dir / "split_info.json", split_info)

    logger.info(
        "Samples: total=%d, train=%d, validation=%d, test=%d",
        n_samples,
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )
    logger.info("Wavelength points: %d", n_wavelengths)
    logger.info("Structure input dimension: %d", expected_structure_dim)
    logger.info("Pointwise input dimension including raw k0: %d", expected_structure_dim + 1)
    logger.info("Normalized k0 range: [%.6f, %.6f]", k0_normalized.min(), k0_normalized.max())

    return {
        "structure_features": structure_features,
        "targets": q_sca,
        "wavelengths": wavelengths,
        "k0_normalized": k0_normalized,
        "k0_min": k0_min,
        "k0_max": k0_max,
        "materials": materials,
        "config": config,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "structure_input_dim": expected_structure_dim,
        "output_dim": n_wavelengths,
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "split_dir": split_dir,
    }


# ==================== DataLoader ====================

def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features[indices]),
        torch.from_numpy(targets[indices]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


# ==================== Model ====================

class RawK0MLP(nn.Module):
    """Shared pointwise MLP conditioned on one raw normalized k0 coordinate."""

    def __init__(
        self,
        structure_input_dim: int,
        wavelengths_nm: torch.Tensor,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        wavelengths_nm = torch.as_tensor(wavelengths_nm, dtype=torch.float32)
        if wavelengths_nm.ndim != 1 or wavelengths_nm.numel() < 2:
            raise ValueError("wavelengths_nm must be a one-dimensional grid.")
        if torch.any(wavelengths_nm <= 0.0):
            raise ValueError("wavelengths_nm must be strictly positive.")

        k0 = 2.0 * math.pi / wavelengths_nm
        k0_min = k0.min()
        k0_max = k0.max()
        if not bool(k0_max > k0_min):
            raise ValueError("wavelength grid has zero k0 range.")
        k0_normalized = (k0 - k0_min) / (k0_max - k0_min)

        self.structure_input_dim = int(structure_input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_wavelengths = int(wavelengths_nm.numel())
        self.register_buffer("wavelengths_nm", wavelengths_nm)
        self.register_buffer("k0_min", k0_min)
        self.register_buffer("k0_max", k0_max)
        self.register_buffer("k0_normalized", k0_normalized)

        # This factorized first layer is exactly equivalent to applying one
        # Linear(structure_input_dim + 1, hidden_dim) layer to concatenated
        # [structure, k0] inputs, while avoiding an explicit repeated tensor.
        self.structure_projection = nn.Linear(structure_input_dim, hidden_dim)
        self.coordinate_projection = nn.Linear(1, hidden_dim, bias=False)
        self.hidden_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_activation = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def wavelength_to_coordinate(self, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        wavelengths_nm = torch.as_tensor(
            wavelengths_nm,
            dtype=self.wavelengths_nm.dtype,
            device=self.wavelengths_nm.device,
        )
        if wavelengths_nm.ndim != 1:
            raise ValueError("wavelengths_nm must be one-dimensional.")
        if torch.any(wavelengths_nm <= 0.0):
            raise ValueError("wavelengths_nm must be strictly positive.")
        k0 = 2.0 * math.pi / wavelengths_nm
        return (k0 - self.k0_min) / (self.k0_max - self.k0_min)

    def forward(
        self,
        structure_features: torch.Tensor,
        wavelengths_nm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        structure_features = structure_features.float()
        if structure_features.ndim != 2:
            raise ValueError("structure_features must have shape (batch, feature).")
        if structure_features.shape[1] != self.structure_input_dim:
            raise ValueError("structure_features has the wrong feature dimension.")

        coordinate = (
            self.k0_normalized
            if wavelengths_nm is None
            else self.wavelength_to_coordinate(wavelengths_nm)
        )
        structure_hidden = self.structure_projection(structure_features)
        coordinate_hidden = self.coordinate_projection(coordinate.reshape(-1, 1))
        hidden = structure_hidden[:, None, :] + coordinate_hidden[None, :, :]
        hidden = self.hidden_layers(hidden)
        output = self.output_layer(hidden).squeeze(-1)
        return self.output_activation(output)


# ==================== Training and inference ====================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(features)
            loss = loss_function(predictions, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        current_batch = len(features)
        total_loss += float(loss.item()) * current_batch
        total_samples += current_batch
    if total_samples == 0:
        raise RuntimeError("Training loader is empty.")
    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(features)
            loss = loss_function(predictions, targets)
        current_batch = len(features)
        total_loss += float(loss.item()) * current_batch
        total_samples += current_batch
    if total_samples == 0:
        raise RuntimeError("Evaluation loader is empty.")
    return total_loss / total_samples


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions = []
    all_targets = []
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(features)
        all_predictions.append(predictions.float().cpu().numpy())
        all_targets.append(targets.numpy())
    if not all_predictions:
        raise RuntimeError("Prediction loader is empty.")
    return np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)


# ==================== Metrics ====================

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    wavelengths: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    if predictions.shape != targets.shape:
        raise ValueError("Prediction and target shapes differ.")
    if predictions.ndim != 2 or predictions.shape[1] != len(wavelengths):
        raise ValueError("Predictions do not match the wavelength grid.")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or Inf.")

    error = predictions - targets
    absolute_error = np.abs(error)
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(absolute_error))
    target_mean = float(np.mean(targets))
    target_range = float(np.max(targets) - np.min(targets))
    denominator = np.maximum(np.abs(targets), 1e-8)

    target_flat = targets.ravel()
    prediction_flat = predictions.ravel()
    ss_residual = float(np.sum((target_flat - prediction_flat) ** 2))
    ss_total = float(np.sum((target_flat - np.mean(target_flat)) ** 2))
    r2 = 1.0 - ss_residual / ss_total if ss_total > 0.0 else float("nan")

    sample_mae = np.mean(absolute_error, axis=1)
    true_peak_index = np.argmax(targets, axis=1)
    pred_peak_index = np.argmax(predictions, axis=1)
    peak_wavelength_mae = float(
        np.mean(np.abs(wavelengths[pred_peak_index] - wavelengths[true_peak_index]))
    )
    peak_amplitude_mae = float(
        np.mean(np.abs(np.max(predictions, axis=1) - np.max(targets, axis=1)))
    )
    per_wavelength_mae = np.mean(absolute_error, axis=0)
    per_wavelength_rmse = np.sqrt(np.mean(error**2, axis=0))

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": float(r2),
        "MAPE_percent": float(np.mean(absolute_error / denominator) * 100.0),
        "NRMSE_by_mean": rmse / target_mean if target_mean > 0.0 else float("nan"),
        "NRMSE_by_range": rmse / target_range if target_range > 0.0 else float("nan"),
        "maximum_absolute_error": float(np.max(absolute_error)),
        "median_sample_MAE": float(np.median(sample_mae)),
        "95th_percentile_sample_MAE": float(np.quantile(sample_mae, 0.95)),
        "peak_wavelength_MAE_nm": peak_wavelength_mae,
        "peak_amplitude_MAE": peak_amplitude_mae,
    }
    return metrics, per_wavelength_mae, per_wavelength_rmse, sample_mae


# ==================== Plots ====================

def plot_training_history(history: Dict[str, Iterable[float]], output_dir: Path) -> None:
    epochs = np.asarray(history["epoch"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(epochs, history["train_loss"], label="Training loss")
    ax.semilogy(epochs, history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(r"Raw-$k_0$ MLP training history")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=200)
    plt.close(fig)


def plot_wavelength_errors(
    wavelengths: np.ndarray,
    per_wavelength_mae: np.ndarray,
    per_wavelength_rmse: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelengths, per_wavelength_mae, label="MAE")
    ax.plot(wavelengths, per_wavelength_rmse, label="RMSE")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Prediction error")
    ax.set_title(r"Raw-$k_0$ MLP error by wavelength")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_wavelength.png", dpi=200)
    plt.close(fig)


def plot_sample_predictions(
    wavelengths: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    sample_mae: np.ndarray,
    output_dir: Path,
) -> None:
    order = np.argsort(sample_mae)
    selected = [
        int(order[0]),
        int(order[len(order) // 2]),
        int(order[-1]),
    ]
    rng = np.random.default_rng(2026)
    while len(selected) < min(6, len(predictions)):
        candidate = int(rng.integers(0, len(predictions)))
        if candidate not in selected:
            selected.append(candidate)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, index in zip(axes.ravel(), selected):
        ax.set_visible(True)
        ax.plot(wavelengths, targets[index], label="PyMieDiff", linewidth=1.8)
        ax.plot(
            wavelengths,
            predictions[index],
            "--",
            label=r"Raw-$k_0$ MLP",
            linewidth=1.5,
        )
        ax.set_title(f"Test row {index}, MAE={sample_mae[index]:.4f}")
        ax.set_ylabel(r"$Q_{\mathrm{sca}}$")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlabel("Wavelength (nm)")
    fig.suptitle(r"Raw-$k_0$ MLP test predictions")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_predictions.png", dpi=200)
    plt.close(fig)


# ==================== Main ====================

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")
    logger.info("=" * 72)
    logger.info("Raw-k0 wavelength-conditioned MLP (no Fourier, no physics features)")
    logger.info("=" * 72)
    logger.info("Device: %s", device)
    logger.info("AMP: %s", use_amp)

    bundle = load_and_prepare_data(
        data_path=args.data_path,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        logger=logger,
    )
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "features": bundle["structure_features"],
        "targets": bundle["targets"],
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "seed": args.seed,
    }
    train_loader = make_loader(
        indices=bundle["train_indices"], shuffle=True, **loader_kwargs
    )
    val_loader = make_loader(
        indices=bundle["val_indices"], shuffle=False, **loader_kwargs
    )
    test_loader = make_loader(
        indices=bundle["test_indices"], shuffle=False, **loader_kwargs
    )

    model = RawK0MLP(
        structure_input_dim=bundle["structure_input_dim"],
        wavelengths_nm=torch.from_numpy(bundle["wavelengths"]),
        hidden_dim=args.hidden_dim,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model parameters: %s", f"{parameter_count:,}")
    logger.info(
        "Architecture: structure(%d) + raw normalized k0(1) -> %d -> %d -> %d -> 1, shared over %d wavelengths",
        bundle["structure_input_dim"],
        args.hidden_dim,
        args.hidden_dim,
        args.hidden_dim,
        bundle["output_dim"],
    )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_model_path = args.output_dir / "best_model.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    training_start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            scaler,
            use_amp,
        )
        val_loss = evaluate_loss(model, val_loader, loss_function, device, use_amp)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        improved = val_loss < best_val_loss - 1e-10
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_class": "RawK0MLP",
                    "input_dim": bundle["structure_input_dim"] + 1,
                    "structure_input_dim": bundle["structure_input_dim"],
                    "output_dim": bundle["output_dim"],
                    "hidden_dim": args.hidden_dim,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "parameter_count": parameter_count,
                    "wavelength_coordinate": "normalized_k0",
                    "uses_fourier_features": False,
                    "uses_physics_features": False,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            logger.info(
                "Epoch %03d | train=%.8f | val=%.8f | lr=%.2e | best=%.8f",
                epoch,
                train_loss,
                val_loss,
                current_lr,
                best_val_loss,
            )
        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d; best epoch=%d", epoch, best_epoch)
            break

    training_seconds = time.time() - training_start
    logger.info("Training completed in %.2f seconds", training_seconds)
    logger.info("Best epoch: %d", best_epoch)
    logger.info("Best validation MSE: %.8f", best_val_loss)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions, targets = predict(model, test_loader, device, use_amp)
    metrics, per_wavelength_mae, per_wavelength_rmse, sample_mae = calculate_metrics(
        predictions, targets, bundle["wavelengths"]
    )

    logger.info("Test metrics:")
    for name, value in metrics.items():
        logger.info("  %s: %.8f", name, value)

    metrics.update(
        {
            "model_name": "RawK0MLP",
            "parameter_count": parameter_count,
            "best_epoch": best_epoch,
            "best_validation_MSE": best_val_loss,
            "training_seconds": training_seconds,
            "device": str(device),
            "used_amp": use_amp,
        }
    )
    save_json(args.output_dir / "test_metrics.json", metrics)

    hyperparameters = {
        **vars(args),
        "data_path": str(args.data_path),
        "split_dir": str(bundle["split_dir"]),
        "output_dir": str(args.output_dir),
        "model": {
            "name": "RawK0MLP",
            "structure_input_dim": bundle["structure_input_dim"],
            "pointwise_input_dim": bundle["structure_input_dim"] + 1,
            "pointwise_output_dim": 1,
            "assembled_spectrum_dim": bundle["output_dim"],
            "hidden_dim": args.hidden_dim,
            "activation": "ReLU",
            "output_activation": "Softplus",
            "loss": "MSE",
            "parameter_count": parameter_count,
        },
        "input_features": {
            "geometry": [
                "r_core",
                "shell1_thickness",
                "shell2_thickness",
                "shell3_thickness",
            ],
            "material": "4-layer one-hot",
            "uses_wavelength_input": True,
            "wavelength_coordinate": "normalized_k0=(2*pi/lambda-k0_min)/(k0_max-k0_min)",
            "uses_fourier_features": False,
            "uses_material_nk": False,
            "uses_physics_features": False,
            "uses_auxiliary_qabs_qext_tasks": False,
            "uses_physical_constraints": False,
        },
        "ablation_role": (
            "Tests explicit pointwise wavelength conditioning without Fourier or physics guidance."
        ),
    }
    save_json(args.output_dir / "hyperparameters.json", hyperparameters)

    np.savez_compressed(
        args.output_dir / "training_history.npz",
        epoch=np.asarray(history["epoch"]),
        train_loss=np.asarray(history["train_loss"]),
        val_loss=np.asarray(history["val_loss"]),
        learning_rate=np.asarray(history["learning_rate"]),
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        sample_ids=bundle["test_indices"],
        wavelengths=bundle["wavelengths"],
        true_q_sca=targets,
        pred_q_sca=predictions,
        absolute_error=np.abs(predictions - targets),
        sample_mae=sample_mae,
    )

    with open(
        args.output_dir / "per_wavelength_metrics.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("wavelength_nm", "MAE", "RMSE"))
        for wavelength, mae, rmse in zip(
            bundle["wavelengths"], per_wavelength_mae, per_wavelength_rmse
        ):
            writer.writerow((float(wavelength), float(mae), float(rmse)))

    plot_training_history(history, args.output_dir)
    plot_wavelength_errors(
        bundle["wavelengths"],
        per_wavelength_mae,
        per_wavelength_rmse,
        args.output_dir,
    )
    plot_sample_predictions(
        bundle["wavelengths"], predictions, targets, sample_mae, args.output_dir
    )
    logger.info("Outputs saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
