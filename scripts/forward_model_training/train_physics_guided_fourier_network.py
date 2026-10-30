"""Train and evaluate the paper-core PGFF forward surrogate.

This script deliberately reuses the exact train/validation/test indices and
geometry normalization of the direct-spectrum network. Test data are never used
for normalization, physics-feature scaling, scheduling, or model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core_shell_optical_design.forward_models.physics_guided_fourier_network import PGFFNet, PHYSICS_FEATURE_NAMES
from core_shell_optical_design.project_paths import (
    DATASET_PATH,
    DIRECT_MODEL_DIR,
    REPRODUCED_RESULT_DIR,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA = DATASET_PATH
DEFAULT_BASELINE_DIR = DIRECT_MODEL_DIR
DEFAULT_OUTPUT = (
    REPRODUCED_RESULT_DIR
    / "forward_models"
    / "physics_guided_fourier_network"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PGFF-Net")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline_dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fourier_seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--fourier_frequencies", type=int, default=32)
    parser.add_argument("--fourier_scale", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda_abs", type=float, default=0.10)
    parser.add_argument("--lambda_ext", type=float, default=0.10)
    parser.add_argument("--lambda_curve", type=float, default=0.01)
    parser.add_argument("--curve_warmup_epochs", type=int, default=20)
    parser.add_argument("--huber_beta", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--scheduler_patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_json(path: Path, value: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
    os.replace(temp, path)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("PGFF")
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


def load_config(data) -> dict:
    if "config" in data:
        return data["config"].item()
    return json.loads(str(data["config_json"].item()))


def cumulative_to_geometry(radii: np.ndarray) -> np.ndarray:
    geometry = np.empty_like(radii, dtype=np.float32)
    geometry[:, 0] = radii[:, 0]
    geometry[:, 1:] = np.diff(radii, axis=1)
    return geometry


def validate_splits(train_idx, val_idx, test_idx, n_samples: int):
    train_set, val_set, test_set = map(set, (train_idx.tolist(), val_idx.tolist(), test_idx.tolist()))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Baseline split files overlap.")
    combined = np.concatenate((train_idx, val_idx, test_idx))
    if len(combined) != n_samples or len(np.unique(combined)) != n_samples:
        raise RuntimeError("Baseline split files do not cover the dataset exactly once.")


def load_data(args, logger):
    required_baseline = (
        "idx_train.npy", "idx_val.npy", "idx_test.npy", "normalization.npz"
    )
    for name in required_baseline:
        if not (args.baseline_dir / name).exists():
            raise FileNotFoundError(f"Missing baseline artifact: {args.baseline_dir / name}")

    with np.load(args.data_path, allow_pickle=True) as data:
        for key in (
            "radii", "material_indices", "wavelengths", "Q_sca", "Q_abs",
            "Q_ext", "material_n", "material_k",
        ):
            if key not in data:
                raise KeyError(f"Dataset is missing {key}")
        radii = data["radii"].astype(np.float32)
        material_indices = data["material_indices"].astype(np.int64)
        wavelengths = data["wavelengths"].astype(np.float32)
        q_sca = data["Q_sca"].astype(np.float32)
        q_abs = np.maximum(data["Q_abs"].astype(np.float32), 0.0)
        q_ext = data["Q_ext"].astype(np.float32)
        material_n = data["material_n"].astype(np.float32)
        material_k = data["material_k"].astype(np.float32)
        config = load_config(data)

    geometry = cumulative_to_geometry(radii)
    n_samples, n_wavelengths = q_sca.shape
    if geometry.shape != (n_samples, 4):
        raise ValueError("Geometry shape mismatch.")
    if material_indices.shape != geometry.shape:
        raise ValueError("Material-index shape mismatch.")
    if q_abs.shape != q_sca.shape or q_ext.shape != q_sca.shape:
        raise ValueError("Target spectra do not have identical shapes.")
    if material_n.shape != material_k.shape or material_n.shape[1] != n_wavelengths:
        raise ValueError("Material optical tables do not match the wavelength grid.")
    if not all(np.isfinite(x).all() for x in (geometry, q_sca, q_abs, q_ext, material_n, material_k)):
        raise ValueError("Non-finite values detected.")

    train_idx = np.load(args.baseline_dir / "idx_train.npy").astype(np.int64)
    val_idx = np.load(args.baseline_dir / "idx_val.npy").astype(np.int64)
    test_idx = np.load(args.baseline_dir / "idx_test.npy").astype(np.int64)
    validate_splits(train_idx, val_idx, test_idx, n_samples)

    with np.load(args.baseline_dir / "normalization.npz", allow_pickle=False) as norm:
        geometry_mean = norm["geometry_mean"].astype(np.float32)
        geometry_std = norm["geometry_std"].astype(np.float32)
        baseline_wavelengths = norm["wavelengths"].astype(np.float32)
    if not np.array_equal(wavelengths, baseline_wavelengths):
        raise RuntimeError("Baseline and PGFF wavelength grids differ.")
    recomputed_mean = geometry[train_idx].mean(axis=0)
    recomputed_std = geometry[train_idx].std(axis=0)
    if not np.allclose(geometry_mean, recomputed_mean, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Stored baseline geometry mean is inconsistent.")
    if not np.allclose(geometry_std, recomputed_std, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Stored baseline geometry std is inconsistent.")

    logger.info(
        "Reused baseline split: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return {
        "geometry": geometry,
        "material_indices": material_indices,
        "wavelengths": wavelengths,
        "q_sca": q_sca,
        "q_abs": q_abs,
        "q_ext": q_ext,
        "material_n": material_n,
        "material_k": material_k,
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "config": config,
    }


def physics_features_numpy(
    geometry: np.ndarray,
    material_indices: np.ndarray,
    wavelengths: np.ndarray,
    material_n: np.ndarray,
    material_k: np.ndarray,
) -> np.ndarray:
    radii = np.cumsum(geometry, axis=1)
    n_layers = material_n[material_indices].transpose(0, 2, 1)
    k_layers = material_k[material_indices].transpose(0, 2, 1)
    k0 = (2.0 * np.pi / wavelengths).reshape(1, -1, 1)
    x = k0 * radii[:, None, :]
    dn = np.concatenate((n_layers[..., :-1] - n_layers[..., 1:], n_layers[..., -1:] - 1.0), axis=-1)
    dk = np.concatenate((k_layers[..., :-1] - k_layers[..., 1:], k_layers[..., -1:]), axis=-1)
    phi = k0 * n_layers * geometry[:, None, :]
    return np.concatenate((x, n_layers, k_layers, dn, dk, phi), axis=-1)


def compute_physics_statistics(bundle, output_dir: Path, logger, chunk_size: int = 512):
    stats_path = output_dir / "physics_normalization.npz"
    feature_count = len(PHYSICS_FEATURE_NAMES)
    total_count = 0
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    feature_square_sum = np.zeros(feature_count, dtype=np.float64)
    indices = bundle["train_idx"]

    for start in range(0, len(indices), chunk_size):
        selected = indices[start:start + chunk_size]
        features = physics_features_numpy(
            bundle["geometry"][selected], bundle["material_indices"][selected],
            bundle["wavelengths"].astype(np.float64),
            bundle["material_n"].astype(np.float64),
            bundle["material_k"].astype(np.float64),
        )
        flattened = features.reshape(-1, feature_count)
        feature_sum += flattened.sum(axis=0)
        feature_square_sum += np.square(flattened).sum(axis=0)
        total_count += flattened.shape[0]

    mean = feature_sum / total_count
    variance = np.maximum(feature_square_sum / total_count - mean**2, 1e-16)
    std = np.sqrt(variance)
    np.savez(
        stats_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_names=np.asarray(PHYSICS_FEATURE_NAMES),
        source="training split only",
    )
    logger.info("Computed 24 physics-feature statistics from the training split only.")
    return mean.astype(np.float32), std.astype(np.float32)


def make_loader(bundle, indices, batch_size, shuffle, num_workers, pin_memory, seed):
    dataset = TensorDataset(
        torch.from_numpy(bundle["geometry"][indices]),
        torch.from_numpy(bundle["material_indices"][indices]),
        torch.from_numpy(bundle["q_sca"][indices]),
        torch.from_numpy(bundle["q_abs"][indices]),
        torch.from_numpy(bundle["q_ext"][indices]),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


class PGFFLoss(nn.Module):
    def __init__(self, lambda_abs, lambda_ext, lambda_curve, huber_beta):
        super().__init__()
        self.lambda_abs = float(lambda_abs)
        self.lambda_ext = float(lambda_ext)
        self.lambda_curve = float(lambda_curve)
        self.mse = nn.MSELoss()
        self.abs_loss = nn.SmoothL1Loss(beta=float(huber_beta))

    def forward(self, prediction, q_sca, q_abs, q_ext, curve_fraction=1.0):
        loss_sca = self.mse(prediction["q_sca"], q_sca)
        loss_abs = self.abs_loss(prediction["q_abs"], q_abs)
        loss_ext = self.mse(prediction["q_ext"], q_ext)
        pred_curve = prediction["q_sca"][:, 2:] - 2.0 * prediction["q_sca"][:, 1:-1] + prediction["q_sca"][:, :-2]
        true_curve = q_sca[:, 2:] - 2.0 * q_sca[:, 1:-1] + q_sca[:, :-2]
        loss_curve = self.mse(pred_curve, true_curve)
        total = (
            loss_sca
            + self.lambda_abs * loss_abs
            + self.lambda_ext * loss_ext
            + self.lambda_curve * float(curve_fraction) * loss_curve
        )
        return total, {
            "total": total.detach(), "sca": loss_sca.detach(),
            "abs": loss_abs.detach(), "ext": loss_ext.detach(),
            "curve": loss_curve.detach(),
        }


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, curve_fraction=1.0):
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "sca", "abs", "ext", "curve")}
    count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for geometry, material, q_sca, q_abs, q_ext in loader:
            geometry = geometry.to(device, non_blocking=True)
            material = material.to(device, non_blocking=True)
            q_sca = q_sca.to(device, non_blocking=True)
            q_abs = q_abs.to(device, non_blocking=True)
            q_ext = q_ext.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(geometry, material)
                loss, components = criterion(
                    prediction, q_sca, q_abs, q_ext, curve_fraction
                )
            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            batch_count = geometry.shape[0]
            for name, value in components.items():
                totals[name] += float(value.item()) * batch_count
            count += batch_count
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outputs = {name: [] for name in ("pred_sca", "pred_abs", "pred_ext", "true_sca", "true_abs", "true_ext")}
    for geometry, material, q_sca, q_abs, q_ext in loader:
        prediction = model(geometry.to(device), material.to(device))
        outputs["pred_sca"].append(prediction["q_sca"].cpu().numpy())
        outputs["pred_abs"].append(prediction["q_abs"].cpu().numpy())
        outputs["pred_ext"].append(prediction["q_ext"].cpu().numpy())
        outputs["true_sca"].append(q_sca.numpy())
        outputs["true_abs"].append(q_abs.numpy())
        outputs["true_ext"].append(q_ext.numpy())
    return {name: np.concatenate(value, axis=0) for name, value in outputs.items()}


def regression_metrics(pred, true):
    error = pred - true
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((true - true.mean())**2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mae,
        "R2": float(1.0 - ss_res / ss_tot),
        "MAPE_percent": float(np.mean(np.abs(error) / np.maximum(np.abs(true), 1e-8)) * 100.0),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def scattering_metrics(pred, true, wavelengths):
    result = regression_metrics(pred, true)
    sample_mae = np.mean(np.abs(pred - true), axis=1)
    true_peak = np.argmax(true, axis=1)
    pred_peak = np.argmax(pred, axis=1)
    short = wavelengths <= 600.5
    result.update(
        {
            "median_sample_MAE": float(np.median(sample_mae)),
            "95th_percentile_sample_MAE": float(np.quantile(sample_mae, 0.95)),
            "peak_wavelength_MAE_nm": float(np.mean(np.abs(wavelengths[pred_peak] - wavelengths[true_peak]))),
            "peak_amplitude_MAE": float(np.mean(np.abs(pred.max(axis=1) - true.max(axis=1)))),
            "shortwave_MAE_400p5_600p5": float(np.mean(np.abs(pred[:, short] - true[:, short]))),
            "longwave_MAE_above_600p5": float(np.mean(np.abs(pred[:, ~short] - true[:, ~short]))),
        }
    )
    return result, sample_mae


def save_plots(history, predictions, wavelengths, output_dir, baseline_dir):
    epochs = np.asarray(history["epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(epochs, history["train_sca"], label="train Q_sca MSE")
    axes[0].semilogy(epochs, history["val_sca"], label="val Q_sca MSE")
    axes[0].set(xlabel="Epoch", ylabel="MSE", title="PGFF primary objective")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].semilogy(epochs, history["val_abs"], label="Q_abs")
    axes[1].semilogy(epochs, history["val_ext"], label="Q_ext")
    axes[1].semilogy(epochs, history["val_curve"], label="curvature")
    axes[1].set(xlabel="Epoch", ylabel="Validation component", title="Auxiliary losses")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / "training_history.png", dpi=200); plt.close(fig)

    error = np.abs(predictions["pred_sca"] - predictions["true_sca"])
    pgff_mae = error.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelengths, pgff_mae, label="PGFF-Net")
    baseline_path = baseline_dir / "test_predictions.npz"
    if baseline_path.exists():
        with np.load(baseline_path) as baseline:
            baseline_mae = np.abs(baseline["pred_q_sca"] - baseline["true_q_sca"]).mean(axis=0)
        ax.plot(wavelengths, baseline_mae, label="Simple MLP")
    ax.set(xlabel="Wavelength (nm)", ylabel="MAE", title="Q_sca error by wavelength")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output_dir / "error_by_wavelength.png", dpi=200); plt.close(fig)

    sample_mae = error.mean(axis=1)
    order = np.argsort(sample_mae)
    selected = [order[0], order[len(order)//2], order[-1], order[len(order)//4], order[3*len(order)//4], order[int(0.95*len(order))]]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for ax, index in zip(axes.ravel(), selected):
        ax.plot(wavelengths, predictions["true_sca"][index], label="PyMieDiff")
        ax.plot(wavelengths, predictions["pred_sca"][index], "--", label="PGFF-Net")
        ax.set_title(f"Test row {index}, MAE={sample_mae[index]:.4f}")
        ax.set_ylabel("Q_sca"); ax.grid(alpha=0.3); ax.legend()
    axes[-1, 0].set_xlabel("Wavelength (nm)"); axes[-1, 1].set_xlabel("Wavelength (nm)")
    fig.tight_layout(); fig.savefig(output_dir / "sample_predictions.png", dpi=200); plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.use_amp and device.type == "cuda")
    logger.info("PGFF-Net | device=%s | AMP=%s", device, use_amp)

    bundle = load_data(args, logger)
    physics_mean, physics_std = compute_physics_statistics(bundle, args.output_dir, logger)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(bundle, bundle["train_idx"], args.batch_size, True, args.num_workers, pin_memory, args.seed)
    val_loader = make_loader(bundle, bundle["val_idx"], args.batch_size, False, args.num_workers, pin_memory, args.seed)
    test_loader = make_loader(bundle, bundle["test_idx"], args.batch_size, False, args.num_workers, pin_memory, args.seed)

    model = PGFFNet(
        geometry_mean=torch.from_numpy(bundle["geometry_mean"]),
        geometry_std=torch.from_numpy(bundle["geometry_std"]),
        wavelengths_nm=torch.from_numpy(bundle["wavelengths"]),
        material_n=torch.from_numpy(bundle["material_n"]),
        material_k=torch.from_numpy(bundle["material_k"]),
        physics_mean=torch.from_numpy(physics_mean),
        physics_std=torch.from_numpy(physics_std),
        hidden_dim=args.hidden_dim,
        n_fourier_frequencies=args.fourier_frequencies,
        fourier_scale=args.fourier_scale,
        fourier_seed=args.fourier_seed,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model parameters: %s", f"{parameter_count:,}")

    criterion = PGFFLoss(args.lambda_abs, args.lambda_ext, args.lambda_curve, args.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_val_sca = float("inf")
    best_epoch = 0
    patience_counter = 0
    component_names = ("total", "sca", "abs", "ext", "curve")
    history = {"epoch": [], "learning_rate": []}
    for split in ("train", "val"):
        for name in component_names:
            history[f"{split}_{name}"] = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        curve_fraction = min(1.0, epoch / max(1, args.curve_warmup_epochs))
        train_values = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp, curve_fraction
        )
        val_values = run_epoch(
            model, val_loader, criterion, device, None, None, use_amp, 1.0
        )
        scheduler.step(val_values["sca"])
        history["epoch"].append(epoch)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        for name in component_names:
            history[f"train_{name}"].append(train_values[name])
            history[f"val_{name}"].append(val_values[name])

        improved = val_values["sca"] < best_val_sca - 1e-10
        if improved:
            best_val_sca = val_values["sca"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_q_sca_mse": best_val_sca,
                    "parameter_count": parameter_count,
                    "model_class": "PGFFNet",
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            logger.info(
                "Epoch %03d | train_sca=%.7f | val_sca=%.7f | val_total=%.7f | lr=%.2e",
                epoch, train_values["sca"], val_values["sca"], val_values["total"], optimizer.param_groups[0]["lr"],
            )
        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d; best epoch=%d", epoch, best_epoch)
            break

    training_seconds = time.time() - start_time
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict(model, test_loader, device)

    metrics_sca, sample_mae = scattering_metrics(
        predictions["pred_sca"], predictions["true_sca"], bundle["wavelengths"]
    )
    metrics_abs = regression_metrics(predictions["pred_abs"], predictions["true_abs"])
    metrics_ext = regression_metrics(predictions["pred_ext"], predictions["true_ext"])
    physical_residual = predictions["pred_ext"] - predictions["pred_sca"] - predictions["pred_abs"]
    metrics = {
        "model_name": "PGFFNet",
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_q_sca_MSE": best_val_sca,
        "training_seconds": training_seconds,
        "device": str(device),
        "Q_sca": metrics_sca,
        "Q_abs": metrics_abs,
        "Q_ext": metrics_ext,
        "max_physical_residual": float(np.max(np.abs(physical_residual))),
        "negative_prediction_count": int(
            np.sum(predictions["pred_sca"] < 0) + np.sum(predictions["pred_abs"] < 0) + np.sum(predictions["pred_ext"] < 0)
        ),
    }

    baseline_metrics_path = args.baseline_dir / "test_metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r", encoding="utf-8") as handle:
            baseline_metrics = json.load(handle)
        metrics["relative_to_baseline"] = {
            "MAE_reduction_percent": 100.0 * (baseline_metrics["MAE"] - metrics_sca["MAE"]) / baseline_metrics["MAE"],
            "RMSE_reduction_percent": 100.0 * (baseline_metrics["RMSE"] - metrics_sca["RMSE"]) / baseline_metrics["RMSE"],
            "peak_wavelength_MAE_reduction_percent": 100.0 * (baseline_metrics["peak_wavelength_MAE_nm"] - metrics_sca["peak_wavelength_MAE_nm"]) / baseline_metrics["peak_wavelength_MAE_nm"],
        }

    save_json(args.output_dir / "test_metrics.json", metrics)
    save_json(
        args.output_dir / "hyperparameters.json",
        {
            **vars(args),
            "data_path": str(args.data_path),
            "baseline_dir": str(args.baseline_dir),
            "output_dir": str(args.output_dir),
            "physics_feature_names": PHYSICS_FEATURE_NAMES,
            "parameter_count": parameter_count,
            "model_selection_metric": "validation Q_sca MSE",
            "hard_constraint": "Q_ext = Q_sca + Q_abs",
        },
    )
    np.savez_compressed(args.output_dir / "training_history.npz", **{key: np.asarray(value) for key, value in history.items()})
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        sample_ids=bundle["test_idx"],
        wavelengths=bundle["wavelengths"],
        sample_mae=sample_mae,
        **predictions,
    )

    error = predictions["pred_sca"] - predictions["true_sca"]
    with open(args.output_dir / "per_wavelength_metrics.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("wavelength_nm", "MAE", "RMSE"))
        for wavelength, column in zip(bundle["wavelengths"], error.T):
            writer.writerow((float(wavelength), float(np.mean(np.abs(column))), float(np.sqrt(np.mean(column**2)))))

    save_plots(history, predictions, bundle["wavelengths"], args.output_dir, args.baseline_dir)
    logger.info("Training complete in %.2f min; best epoch=%d", training_seconds / 60.0, best_epoch)
    logger.info("Q_sca test metrics: %s", metrics_sca)
    logger.info("Physical residual max: %.3e", metrics["max_physical_residual"])
    logger.info("Outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
