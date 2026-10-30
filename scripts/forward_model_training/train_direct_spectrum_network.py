"""Train the fixed-grid direct-spectrum forward network.

四层核壳颗粒散射效率预测——固定波长网格的直接光谱网络。

输入：
    几何：
        r_core, t1, t2, t3，共4维

    材料：
        四层材料one-hot
        4层 × 6种材料 = 24维

    总输入维度：
        4 + 24 = 28

输出：
    固定101波长点的Q_sca光谱

特点：
    - 不输入波长坐标
    - 不使用Fourier特征
    - 不输入折射率n/k
    - 不使用尺寸参数
    - 不使用物理约束
    - 只预测Q_sca
    - 归一化参数仅由训练集计算
    - 按材料序列分层划分数据
"""

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from core_shell_optical_design.project_paths import (
    DATASET_PATH,
    REPRODUCED_RESULT_DIR,
)


# ==================== 默认路径 ====================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = DATASET_PATH
DEFAULT_OUTPUT_DIR = (
    REPRODUCED_RESULT_DIR
    / "forward_models"
    / "direct_spectrum_network"
)


# ==================== 参数 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Simple MLP baseline for four-layer "
            "core-shell scattering spectra"
        )
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="在CUDA设备上启用混合精度",
    )

    return parser.parse_args()


# ==================== 可重复性 ====================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== 日志 ====================

def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "SimpleMLPBaseline"
    )
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        output_dir / "training.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ==================== JSON工具 ====================

def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"{type(value).__name__} "
        "is not JSON serializable"
    )


def save_json(
    path: Path,
    obj: dict,
) -> None:
    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    os.replace(temp_path, path)


# ==================== 数据处理 ====================

def load_config(data) -> dict:
    if "config" in data:
        return data["config"].item()

    if "config_json" in data:
        return json.loads(
            str(data["config_json"].item())
        )

    raise KeyError(
        "数据集中缺少config/config_json。"
    )


def cumulative_radii_to_geometry(
    radii: np.ndarray,
) -> np.ndarray:
    """
    累计外半径：

        r1, r2, r3, r4

    转换为四个独立变量：

        r_core, t1, t2, t3
    """
    geometry = np.empty(
        radii.shape,
        dtype=np.float32,
    )

    geometry[:, 0] = radii[:, 0]
    geometry[:, 1] = (
        radii[:, 1] - radii[:, 0]
    )
    geometry[:, 2] = (
        radii[:, 2] - radii[:, 1]
    )
    geometry[:, 3] = (
        radii[:, 3] - radii[:, 2]
    )

    return geometry


def material_sequence_ids(
    material_indices: np.ndarray,
    n_materials: int,
) -> np.ndarray:
    """
    将四层材料排列映射为一个整数序列ID，
    用于分层划分训练/验证/测试集。
    """
    return np.ravel_multi_index(
        tuple(
            material_indices[:, layer]
            for layer in range(4)
        ),
        dims=(n_materials,) * 4,
    )


def build_one_hot(
    material_indices: np.ndarray,
    n_materials: int,
) -> np.ndarray:
    one_hot = np.eye(
        n_materials,
        dtype=np.float32,
    )[material_indices]

    return one_hot.reshape(
        len(material_indices),
        4 * n_materials,
    )


def load_and_prepare_data(
    data_path: Path,
    output_dir: Path,
    seed: int,
    logger: logging.Logger,
):
    if not data_path.exists():
        raise FileNotFoundError(
            f"数据集不存在：{data_path}"
        )

    logger.info(f"加载数据：{data_path}")

    data = np.load(
        data_path,
        allow_pickle=True,
    )

    required_keys = [
        "radii",
        "material_indices",
        "wavelengths",
        "Q_sca",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"数据集缺少字段：{key}"
            )

    radii = data["radii"].astype(
        np.float32
    )
    material_indices = data[
        "material_indices"
    ].astype(np.int64)
    wavelengths = data[
        "wavelengths"
    ].astype(np.float32)
    q_sca = data["Q_sca"].astype(
        np.float32
    )

    config = load_config(data)
    materials = list(config["materials"])
    n_materials = len(materials)

    n_samples = len(radii)
    n_wavelengths = len(wavelengths)

    if radii.shape != (n_samples, 4):
        raise ValueError(
            f"radii形状错误：{radii.shape}"
        )

    if material_indices.shape != (
        n_samples,
        4,
    ):
        raise ValueError(
            "material_indices形状错误："
            f"{material_indices.shape}"
        )

    if q_sca.shape != (
        n_samples,
        n_wavelengths,
    ):
        raise ValueError(
            f"Q_sca形状错误：{q_sca.shape}"
        )

    if not np.isfinite(radii).all():
        raise ValueError(
            "radii包含NaN或Inf。"
        )

    if not np.isfinite(q_sca).all():
        raise ValueError(
            "Q_sca包含NaN或Inf。"
        )

    if q_sca.min() < -1e-6:
        raise ValueError(
            "Q_sca包含明显负值。"
        )

    geometry = cumulative_radii_to_geometry(
        radii
    )

    sequence_ids = material_sequence_ids(
        material_indices,
        n_materials,
    )

    all_indices = np.arange(
        n_samples,
        dtype=np.int64,
    )

    # 70%训练、15%验证、15%测试。
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=sequence_ids,
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=sequence_ids[temp_indices],
    )

    # 归一化参数只从训练集计算。
    geometry_mean = geometry[
        train_indices
    ].mean(axis=0)

    geometry_std = geometry[
        train_indices
    ].std(axis=0)

    geometry_std = np.maximum(
        geometry_std,
        1e-8,
    )

    geometry_normalized = (
        geometry - geometry_mean
    ) / geometry_std

    material_one_hot = build_one_hot(
        material_indices,
        n_materials,
    )

    features = np.concatenate(
        [
            geometry_normalized,
            material_one_hot,
        ],
        axis=1,
    ).astype(np.float32)

    expected_input_dim = (
        4 + 4 * n_materials
    )

    if features.shape != (
        n_samples,
        expected_input_dim,
    ):
        raise ValueError(
            f"输入特征形状错误：{features.shape}"
        )

    np.save(
        output_dir / "idx_train.npy",
        train_indices,
    )
    np.save(
        output_dir / "idx_val.npy",
        val_indices,
    )
    np.save(
        output_dir / "idx_test.npy",
        test_indices,
    )

    np.savez(
        output_dir / "normalization.npz",
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
        geometry_feature_names=np.asarray(
            [
                "r_core",
                "shell1_thickness",
                "shell2_thickness",
                "shell3_thickness",
            ]
        ),
        material_names=np.asarray(materials),
        wavelengths=wavelengths,
    )

    split_info = {
        "n_total": int(n_samples),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(val_indices)),
        "n_test": int(len(test_indices)),
        "train_fraction": (
            len(train_indices) / n_samples
        ),
        "validation_fraction": (
            len(val_indices) / n_samples
        ),
        "test_fraction": (
            len(test_indices) / n_samples
        ),
        "stratification": (
            "four-layer material sequence"
        ),
        "seed": seed,
    }
    save_json(
        output_dir / "split_info.json",
        split_info,
    )

    logger.info(
        f"样本数：{n_samples}"
    )
    logger.info(
        f"波长点数：{n_wavelengths}"
    )
    logger.info(
        f"输入维度：{features.shape[1]}"
    )
    logger.info(
        "数据划分："
        f"train={len(train_indices)}, "
        f"val={len(val_indices)}, "
        f"test={len(test_indices)}"
    )
    logger.info(
        "几何均值："
        f"{geometry_mean.tolist()}"
    )
    logger.info(
        "几何标准差："
        f"{geometry_std.tolist()}"
    )

    return {
        "features": features,
        "targets": q_sca,
        "wavelengths": wavelengths,
        "materials": materials,
        "config": config,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "input_dim": features.shape[1],
        "output_dim": n_wavelengths,
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
    feature_tensor = torch.from_numpy(
        features[indices]
    )
    target_tensor = torch.from_numpy(
        targets[indices]
    )

    dataset = TensorDataset(
        feature_tensor,
        target_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            num_workers > 0
        ),
        generator=(
            generator if shuffle else None
        ),
    )


# ==================== 基础网络 ====================

class SimpleMLPBaseline(nn.Module):
    """
    普通全连接网络：

        28
        → hidden
        → hidden
        → hidden
        → N_wavelength

    使用Softplus保证Q_sca非负。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

        self.output_activation = nn.Softplus()

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        raw_output = self.network(features)

        return self.output_activation(
            raw_output
        )


# ==================== 训练和验证 ====================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    device,
    scaler,
    use_amp,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            predictions = model(features)
            loss = loss_function(
                predictions,
                targets,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = len(features)
        total_loss += (
            loss.item() * batch_size
        )
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    loss_function,
    device,
    use_amp,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            predictions = model(features)
            loss = loss_function(
                predictions,
                targets,
            )

        batch_size = len(features)
        total_loss += (
            loss.item() * batch_size
        )
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def predict(
    model,
    loader,
    device,
):
    model.eval()

    all_predictions = []
    all_targets = []

    for features, targets in loader:
        features = features.to(
            device,
            non_blocking=True,
        )

        predictions = model(features)

        all_predictions.append(
            predictions.cpu().numpy()
        )
        all_targets.append(
            targets.numpy()
        )

    return (
        np.concatenate(
            all_predictions,
            axis=0,
        ),
        np.concatenate(
            all_targets,
            axis=0,
        ),
    )


# ==================== 指标 ====================

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    wavelengths: np.ndarray,
):
    error = predictions - targets
    absolute_error = np.abs(error)

    mse = float(
        np.mean(error**2)
    )
    rmse = float(np.sqrt(mse))
    mae = float(
        np.mean(absolute_error)
    )

    target_mean = float(
        np.mean(targets)
    )
    target_range = float(
        np.max(targets)
        - np.min(targets)
    )

    nrmse_mean = (
        rmse / target_mean
        if target_mean > 0
        else float("nan")
    )
    nrmse_range = (
        rmse / target_range
        if target_range > 0
        else float("nan")
    )

    denominator = np.maximum(
        np.abs(targets),
        1e-8,
    )
    mape = float(
        np.mean(
            absolute_error / denominator
        )
        * 100.0
    )

    target_flat = targets.ravel()
    prediction_flat = predictions.ravel()

    ss_residual = float(
        np.sum(
            (
                target_flat
                - prediction_flat
            )
            ** 2
        )
    )
    ss_total = float(
        np.sum(
            (
                target_flat
                - np.mean(target_flat)
            )
            ** 2
        )
    )

    r2 = (
        1.0 - ss_residual / ss_total
        if ss_total > 0
        else float("nan")
    )

    sample_mae = np.mean(
        absolute_error,
        axis=1,
    )

    true_peak_index = np.argmax(
        targets,
        axis=1,
    )
    pred_peak_index = np.argmax(
        predictions,
        axis=1,
    )

    true_peak_wavelength = wavelengths[
        true_peak_index
    ]
    pred_peak_wavelength = wavelengths[
        pred_peak_index
    ]

    peak_wavelength_mae = float(
        np.mean(
            np.abs(
                pred_peak_wavelength
                - true_peak_wavelength
            )
        )
    )

    true_peak_value = np.max(
        targets,
        axis=1,
    )
    pred_peak_value = np.max(
        predictions,
        axis=1,
    )

    peak_amplitude_mae = float(
        np.mean(
            np.abs(
                pred_peak_value
                - true_peak_value
            )
        )
    )

    per_wavelength_mae = np.mean(
        absolute_error,
        axis=0,
    )
    per_wavelength_rmse = np.sqrt(
        np.mean(
            error**2,
            axis=0,
        )
    )

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE_percent": mape,
        "NRMSE_by_mean": nrmse_mean,
        "NRMSE_by_range": nrmse_range,
        "maximum_absolute_error": float(
            np.max(absolute_error)
        ),
        "median_sample_MAE": float(
            np.median(sample_mae)
        ),
        "95th_percentile_sample_MAE": float(
            np.quantile(
                sample_mae,
                0.95,
            )
        ),
        "peak_wavelength_MAE_nm": (
            peak_wavelength_mae
        ),
        "peak_amplitude_MAE": (
            peak_amplitude_mae
        ),
    }

    return (
        metrics,
        per_wavelength_mae,
        per_wavelength_rmse,
        sample_mae,
    )


# ==================== 绘图 ====================

def plot_training_history(
    history: dict,
    output_dir: Path,
) -> None:
    epochs = np.asarray(
        history["epoch"]
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.semilogy(
        epochs,
        history["train_loss"],
        label="Training loss",
    )
    ax.semilogy(
        epochs,
        history["val_loss"],
        label="Validation loss",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(
        "Simple MLP training history"
    )
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir
        / "training_history.png",
        dpi=200,
    )
    plt.close(fig)


def plot_wavelength_errors(
    wavelengths: np.ndarray,
    per_wavelength_mae: np.ndarray,
    per_wavelength_rmse: np.ndarray,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        wavelengths,
        per_wavelength_mae,
        label="MAE",
    )
    ax.plot(
        wavelengths,
        per_wavelength_rmse,
        label="RMSE",
    )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Prediction error")
    ax.set_title(
        "Baseline error by wavelength"
    )
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir
        / "error_by_wavelength.png",
        dpi=200,
    )
    plt.close(fig)


def plot_sample_predictions(
    wavelengths: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    sample_mae: np.ndarray,
    output_dir: Path,
) -> None:
    sorted_indices = np.argsort(
        sample_mae
    )

    selected = [
        int(sorted_indices[0]),
        int(
            sorted_indices[
                len(sorted_indices) // 2
            ]
        ),
        int(sorted_indices[-1]),
    ]

    rng = np.random.default_rng(2026)

    while len(selected) < 6:
        index = int(
            rng.integers(
                0,
                len(predictions),
            )
        )

        if index not in selected:
            selected.append(index)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 11),
        sharex=True,
    )

    axes = axes.ravel()

    for ax, index in zip(
        axes,
        selected,
    ):
        ax.plot(
            wavelengths,
            targets[index],
            label="PyMieDiff",
            linewidth=1.8,
        )
        ax.plot(
            wavelengths,
            predictions[index],
            "--",
            label="MLP prediction",
            linewidth=1.5,
        )

        ax.set_title(
            f"Test row {index}, "
            f"MAE={sample_mae[index]:.4f}"
        )
        ax.set_ylabel(
            "Q_sca"
        )
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel(
        "Wavelength (nm)"
    )
    axes[-2].set_xlabel(
        "Wavelength (nm)"
    )

    fig.suptitle(
        "Baseline MLP test predictions"
    )
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "sample_predictions.png",
        dpi=200,
    )
    plt.close(fig)


# ==================== 主程序 ====================

def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logger(
        args.output_dir
    )

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = bool(
        args.use_amp
        and device.type == "cuda"
    )

    logger.info("=" * 70)
    logger.info(
        "Simple MLP baseline for Q_sca"
    )
    logger.info("=" * 70)
    logger.info(f"设备：{device}")
    logger.info(f"AMP：{use_amp}")

    data_bundle = load_and_prepare_data(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        logger=logger,
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = make_loader(
        features=data_bundle["features"],
        targets=data_bundle["targets"],
        indices=data_bundle["train_indices"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )

    val_loader = make_loader(
        features=data_bundle["features"],
        targets=data_bundle["targets"],
        indices=data_bundle["val_indices"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )

    test_loader = make_loader(
        features=data_bundle["features"],
        targets=data_bundle["targets"],
        indices=data_bundle["test_indices"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )

    model = SimpleMLPBaseline(
        input_dim=data_bundle["input_dim"],
        output_dim=data_bundle["output_dim"],
        hidden_dim=args.hidden_dim,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    logger.info(
        f"模型参数量：{parameter_count:,}"
    )
    logger.info(
        f"网络结构："
        f"{data_bundle['input_dim']} "
        f"→ {args.hidden_dim} "
        f"→ {args.hidden_dim} "
        f"→ {args.hidden_dim} "
        f"→ {data_bundle['output_dim']}"
    )

    loss_function = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.scheduler_patience,
            min_lr=1e-6,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    best_model_path = (
        args.output_dir
        / "best_model.pt"
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    training_start = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(
            train_loss
        )
        history["val_loss"].append(
            val_loss
        )
        history["learning_rate"].append(
            current_lr
        )

        improved = (
            val_loss
            < best_val_loss - 1e-10
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "input_dim": (
                        data_bundle["input_dim"]
                    ),
                    "output_dim": (
                        data_bundle["output_dim"]
                    ),
                    "hidden_dim": args.hidden_dim,
                    "best_epoch": best_epoch,
                    "best_val_loss": (
                        best_val_loss
                    ),
                    "parameter_count": (
                        parameter_count
                    ),
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):
            logger.info(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.8f} | "
                f"val={val_loss:.8f} | "
                f"lr={current_lr:.2e} | "
                f"best={best_val_loss:.8f}"
            )

        if (
            patience_counter
            >= args.patience
        ):
            logger.info(
                "Early stopping："
                f"epoch={epoch}，"
                f"best_epoch={best_epoch}"
            )
            break

    training_seconds = (
        time.time() - training_start
    )

    logger.info(
        f"训练完成，耗时："
        f"{training_seconds:.2f}秒"
    )
    logger.info(
        f"最佳epoch：{best_epoch}"
    )
    logger.info(
        f"最佳验证MSE："
        f"{best_val_loss:.8f}"
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    predictions, targets = predict(
        model=model,
        loader=test_loader,
        device=device,
    )

    (
        metrics,
        per_wavelength_mae,
        per_wavelength_rmse,
        sample_mae,
    ) = calculate_metrics(
        predictions=predictions,
        targets=targets,
        wavelengths=data_bundle[
            "wavelengths"
        ],
    )

    logger.info("测试集指标：")

    for name, value in metrics.items():
        logger.info(
            f"  {name}: {value:.8f}"
        )

    metrics.update(
        {
            "model_name": (
                "SimpleMLPBaseline"
            ),
            "parameter_count": (
                parameter_count
            ),
            "best_epoch": best_epoch,
            "best_validation_MSE": (
                best_val_loss
            ),
            "training_seconds": (
                training_seconds
            ),
            "device": str(device),
            "used_amp": use_amp,
        }
    )

    save_json(
        args.output_dir
        / "test_metrics.json",
        metrics,
    )

    hyperparameters = {
        **vars(args),
        "data_path": str(
            args.data_path
        ),
        "output_dir": str(
            args.output_dir
        ),
        "model": {
            "name": (
                "SimpleMLPBaseline"
            ),
            "input_dim": (
                data_bundle["input_dim"]
            ),
            "output_dim": (
                data_bundle["output_dim"]
            ),
            "hidden_dim": (
                args.hidden_dim
            ),
            "activation": "ReLU",
            "output_activation": (
                "Softplus"
            ),
            "loss": "MSE",
            "parameter_count": (
                parameter_count
            ),
        },
        "input_features": {
            "geometry": [
                "r_core",
                "shell1_thickness",
                "shell2_thickness",
                "shell3_thickness",
            ],
            "material": (
                "4-layer one-hot"
            ),
            "uses_wavelength_input": False,
            "uses_fourier_features": False,
            "uses_physics_features": False,
        },
    }

    save_json(
        args.output_dir
        / "hyperparameters.json",
        hyperparameters,
    )

    np.savez_compressed(
        args.output_dir
        / "training_history.npz",
        epoch=np.asarray(
            history["epoch"]
        ),
        train_loss=np.asarray(
            history["train_loss"]
        ),
        val_loss=np.asarray(
            history["val_loss"]
        ),
        learning_rate=np.asarray(
            history["learning_rate"]
        ),
    )

    np.savez_compressed(
        args.output_dir
        / "test_predictions.npz",
        sample_ids=data_bundle[
            "test_indices"
        ],
        wavelengths=data_bundle[
            "wavelengths"
        ],
        true_q_sca=targets,
        pred_q_sca=predictions,
        absolute_error=np.abs(
            predictions - targets
        ),
        sample_mae=sample_mae,
    )

    with open(
        args.output_dir
        / "per_wavelength_metrics.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "wavelength_nm",
                "MAE",
                "RMSE",
            ]
        )

        for wavelength, mae, rmse in zip(
            data_bundle["wavelengths"],
            per_wavelength_mae,
            per_wavelength_rmse,
        ):
            writer.writerow(
                [
                    float(wavelength),
                    float(mae),
                    float(rmse),
                ]
            )

    plot_training_history(
        history,
        args.output_dir,
    )

    plot_wavelength_errors(
        wavelengths=data_bundle[
            "wavelengths"
        ],
        per_wavelength_mae=(
            per_wavelength_mae
        ),
        per_wavelength_rmse=(
            per_wavelength_rmse
        ),
        output_dir=args.output_dir,
    )

    plot_sample_predictions(
        wavelengths=data_bundle[
            "wavelengths"
        ],
        predictions=predictions,
        targets=targets,
        sample_mae=sample_mae,
        output_dir=args.output_dir,
    )

    logger.info(
        "全部结果已保存至："
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
