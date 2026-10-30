# -*- coding: utf-8 -*-
"""Stage 02: train the material-conditioned inverse network through frozen PGFF.

Scientific policy
-----------------
* Training uses only the original forward TRAIN split.
* Model selection uses only the original VALIDATION split.
* The TEST split and all blind targets are never read as training examples.
* PGFF parameters are frozen; gradients pass through PGFF only to the inverse net.
* Native material pairs provide stable cycle training and a short decaying geometry
  anchor. Counterfactual legal-material pairs teach the network to support the
  later exact 400-sequence blind enumerator.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from core_shell_optical_design.inverse_design.inverse_design_data import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_DATASET,
    DEFAULT_PGFF_DIR,
    DEFAULT_RESULT_ROOT,
    artifact,
    load_forward_dataset,
    load_split_bundle,
    resolve_device,
    save_json_atomic,
    set_seed,
    sha256_file,
)
from core_shell_optical_design.inverse_design.forward_model_bridge import load_frozen_pgff, pgff_forward
from core_shell_optical_design.inverse_design.geometry_parameterization import (
    GEOMETRY_CHECK_ATOL_NM,
    MAX_OUTER_RADIUS,
    geometry_bounds_mask,
    material_sequence_mask,
)
from core_shell_optical_design.inverse_design.spectrum_conditioned_inverse_network import SpectralMaterialInverseNet
from core_shell_optical_design.inverse_design.inverse_design_objectives import (
    normalized_geometry_anchor_loss,
    normalized_smooth_l1_loss,
    per_sample_metrics,
)

STAGE = "train_spectrum_conditioned_inverse_network"
VERSION = "tandem-mie-inverse-2026-07-24-step02-v3-boundary-fix"


@dataclass(frozen=True)
class ModelConfig:
    n_wavelengths: int
    n_materials: int
    width: int
    depth: int
    dropout: float


class TandemPairDataset(Dataset):
    """A view of one frozen forward split."""

    def __init__(self, dataset: Mapping[str, np.ndarray], indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.qsca = torch.as_tensor(
            np.asarray(dataset["Q_sca"][self.indices], dtype=np.float32)
        )
        self.materials = torch.as_tensor(
            np.asarray(dataset["material_indices"][self.indices], dtype=np.int64)
        )
        self.geometry = torch.as_tensor(
            np.asarray(dataset["geometry"][self.indices], dtype=np.float32)
        )
        if len(self.qsca) != len(self.materials) or len(self.qsca) != len(self.geometry):
            raise RuntimeError("TandemPairDataset arrays have inconsistent lengths.")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        return (
            self.qsca[item],
            self.materials[item],
            self.geometry[item],
            int(self.indices[item]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a material-conditioned inverse network through frozen PGFF."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--pgff-dir", type=Path, default=DEFAULT_PGFF_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=DEFAULT_RESULT_ROOT / "inverse_training_data" / "inverse_design_training_data.npz",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260724)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.01)
    parser.add_argument("--spectral-scale-floor", type=float, default=1.0e-6)

    parser.add_argument("--geometry-anchor-weight", type=float, default=0.05)
    parser.add_argument("--geometry-anchor-decay-epochs", type=int, default=40)
    parser.add_argument("--counterfactual-per-sample", type=int, default=1)
    parser.add_argument("--counterfactual-weight", type=float, default=0.50)
    parser.add_argument("--counterfactual-start-epoch", type=int, default=5)
    parser.add_argument("--counterfactual-probe-interval", type=int, default=5)
    parser.add_argument("--counterfactual-probe-samples", type=int, default=512)
    parser.add_argument("--counterfactual-probe-candidates", type=int, default=4)

    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a two-epoch interface test in a separate smoke-test directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "epochs", "batch_size", "validation_batch_size", "width", "depth",
        "patience", "counterfactual_probe_interval",
    )
    for name in positive_integer_fields:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.counterfactual_per_sample < 0:
        raise ValueError("--counterfactual-per-sample cannot be negative.")
    if args.counterfactual_probe_samples < 0 or args.counterfactual_probe_candidates < 0:
        raise ValueError("Counterfactual probe sizes cannot be negative.")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0,1).")
    for name in (
        "learning_rate", "minimum_learning_rate", "smooth_l1_beta",
        "spectral_scale_floor", "gradient_clip",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("weight_decay", "geometry_anchor_weight", "counterfactual_weight"):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def load_prepared_manifest(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared Stage 01 data not found: {path}")
    required = (
        "train_indices", "validation_indices", "test_indices", "wavelengths_nm",
        "qsca_mean_train", "qsca_std_train", "material_names",
        "legal_material_sequences",
    )
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Prepared data are missing fields: {missing}")
        output = {key: np.array(data[key]) for key in data.files}
    if np.any(output["qsca_std_train"] <= 0.0):
        raise RuntimeError("Prepared qsca_std_train must be positive.")
    return output


def verify_frozen_inputs(
    args: argparse.Namespace,
    dataset: Mapping[str, np.ndarray],
    split,
    prepared: Mapping[str, np.ndarray],
) -> None:
    preflight_path = args.result_root / "environment_preflight" / "preflight_report.json"
    preparation_path = args.result_root / "inverse_training_data" / "preparation_report.json"
    for path in (preflight_path, preparation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required prior-stage report is missing: {path}")
    preflight = read_json(preflight_path)
    preparation = read_json(preparation_path)
    if preflight.get("status") != "passed":
        raise RuntimeError(f"Stage 00 did not pass: {preflight_path}")
    if preparation.get("status") != "passed":
        raise RuntimeError(f"Stage 01 did not pass: {preparation_path}")

    # Freeze provenance: Stage 02 must use exactly the artifacts certified by
    # Stages 00 and 01, not merely files with matching names.
    expected_dataset_hash = str(preflight.get("dataset", {}).get("sha256", ""))
    actual_dataset_hash = sha256_file(args.dataset)
    if expected_dataset_hash and actual_dataset_hash != expected_dataset_hash:
        raise RuntimeError("Dataset SHA256 differs from the Stage 00 certified artifact.")
    expected_pgff_hash = str(
        preflight.get("pgff", {}).get("checkpoint", {}).get("sha256", "")
    )
    actual_pgff_hash = sha256_file(args.pgff_dir / "best_model.pt")
    if expected_pgff_hash and actual_pgff_hash != expected_pgff_hash:
        raise RuntimeError("PGFF checkpoint SHA256 differs from the Stage 00 certified artifact.")
    expected_prepared_hash = str(
        preparation.get("outputs", {}).get("prepared_data", {}).get("sha256", "")
    )
    actual_prepared_hash = sha256_file(args.prepared_data)
    if expected_prepared_hash and actual_prepared_hash != expected_prepared_hash:
        raise RuntimeError("Prepared-data SHA256 differs from the Stage 01 certified artifact.")

    checks = (
        (prepared["train_indices"].astype(np.int64), split.train, "train indices"),
        (prepared["validation_indices"].astype(np.int64), split.validation, "validation indices"),
        (prepared["test_indices"].astype(np.int64), split.test, "test indices"),
    )
    for left, right, label in checks:
        if not np.array_equal(left, right):
            raise RuntimeError(f"Prepared {label} differ from the frozen baseline split.")
    if not np.array_equal(
        prepared["wavelengths_nm"].astype(np.float64),
        np.asarray(dataset["wavelengths"], dtype=np.float64),
    ):
        raise RuntimeError("Prepared and dataset wavelength grids differ.")
    if not np.array_equal(
        prepared["material_names"].astype(str),
        np.asarray(dataset["material_names"], dtype=str),
    ):
        raise RuntimeError("Prepared and dataset material names differ.")
    legal = prepared["legal_material_sequences"].astype(np.int64)
    if legal.ndim != 2 or legal.shape[1] != 4:
        raise RuntimeError("Prepared legal material registry must have shape (N,4).")
    if not np.all(material_sequence_mask(legal, dataset["material_names"])):
        raise RuntimeError("Prepared material registry contains an illegal sequence.")
    if len(dataset["material_names"]) == 6 and len(legal) != 400:
        raise RuntimeError(f"Expected 400 legal sequences, found {len(legal)}.")


def output_directory(args: argparse.Namespace) -> Path:
    name = "inverse_model_training_smoke" if args.smoke_test else "inverse_model_training"
    return args.result_root / name


def prepare_output_directory(args: argparse.Namespace, output_dir: Path) -> None:
    if output_dir.exists() and args.overwrite and not args.resume:
        import shutil
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.resume and not args.overwrite:
        nonempty = any(output_dir.iterdir())
        if nonempty:
            raise FileExistsError(
                f"Output directory already contains files: {output_dir}. "
                "Use --resume or --overwrite deliberately."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1009 * int(epoch) + (17 if shuffle else 0))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=generator,
        persistent_workers=False,
    )


def sample_counterfactual_sequences(
    native_materials_cpu: torch.Tensor,
    legal_sequences_cpu: torch.Tensor,
    *,
    count_per_sample: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return repeated targets' row indices and legal non-native sequences."""
    count = int(count_per_sample)
    batch = int(native_materials_cpu.shape[0])
    if count <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty((0, 4), dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    row_indices = torch.arange(batch, dtype=torch.long).repeat_interleave(count)
    chosen = torch.randint(
        low=0,
        high=len(legal_sequences_cpu),
        size=(batch * count,),
        generator=generator,
    )
    alternatives = legal_sequences_cpu[chosen].clone()
    native_repeated = native_materials_cpu[row_indices]
    equal = (alternatives == native_repeated).all(dim=1)
    attempts = 0
    while bool(equal.any()):
        attempts += 1
        if attempts > 32:
            raise RuntimeError("Failed to sample non-native material sequences.")
        replacement = torch.randint(
            low=0,
            high=len(legal_sequences_cpu),
            size=(int(equal.sum()),),
            generator=generator,
        )
        alternatives[equal] = legal_sequences_cpu[replacement]
        equal = (alternatives == native_repeated).all(dim=1)
    return row_indices, alternatives


def geometry_anchor_weight(epoch: int, base_weight: float, decay_epochs: int) -> float:
    if base_weight <= 0.0 or decay_epochs <= 0:
        return 0.0
    progress = min(max((epoch - 1) / float(decay_epochs), 0.0), 1.0)
    return float(base_weight) * (1.0 - progress)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))




def load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a checkpoint dictionary in {path}.")
    return value


def atomic_torch_save(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def rng_payload() -> dict:
    payload = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: Mapping[str, object]) -> None:
    if "torch_cpu" in payload:
        torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def finite_inverse_gradients(model: torch.nn.Module) -> Tuple[bool, float]:
    total_sq = 0.0
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not torch.isfinite(parameter.grad).all():
            return False, float("nan")
        total_sq += float(parameter.grad.detach().float().square().sum().cpu())
    return found, math.sqrt(total_sq)


def assert_pgff_has_no_gradients(pgff: torch.nn.Module) -> None:
    bad = [name for name, p in pgff.named_parameters() if p.grad is not None]
    if bad:
        raise RuntimeError(f"Frozen PGFF unexpectedly accumulated gradients: {bad[:5]}")


def train_one_epoch(
    *,
    inverse: SpectralMaterialInverseNet,
    pgff: torch.nn.Module,
    train_data: Dataset,
    legal_sequences_cpu: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    inverse.train()
    pgff.eval()
    loader = make_loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        epoch=epoch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    anchor_weight = geometry_anchor_weight(
        epoch, args.geometry_anchor_weight, args.geometry_anchor_decay_epochs
    )
    counter_enabled = (
        args.counterfactual_per_sample > 0
        and args.counterfactual_weight > 0.0
        and epoch >= args.counterfactual_start_epoch
    )

    sums = {
        "samples": 0,
        "total_loss": 0.0,
        "native_spectral_loss": 0.0,
        "counterfactual_spectral_loss": 0.0,
        "geometry_anchor_loss": 0.0,
        "native_nmae": 0.0,
        "gradient_norm": 0.0,
        "gradient_steps": 0,
    }
    for batch_id, (qsca_cpu, materials_cpu, geometry_cpu, _) in enumerate(loader):
        batch_size = int(len(qsca_cpu))
        qsca = qsca_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        materials = materials_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        true_geometry = geometry_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.amp):
            native_output = inverse(qsca, materials)
            native_pgff = pgff_forward(pgff, native_output["geometry"], materials)
            native_loss = normalized_smooth_l1_loss(
                native_pgff["q_sca"],
                qsca,
                beta=args.smooth_l1_beta,
                scale_floor=args.spectral_scale_floor,
            )
            anchor_loss = normalized_geometry_anchor_loss(
                native_output["geometry"], true_geometry
            )
            counter_loss = torch.zeros((), dtype=native_loss.dtype, device=device)
            if counter_enabled:
                rows, alternative_cpu = sample_counterfactual_sequences(
                    materials_cpu,
                    legal_sequences_cpu,
                    count_per_sample=args.counterfactual_per_sample,
                    seed=args.seed + epoch * 1_000_003 + batch_id * 1009,
                )
                alt_qsca = qsca[rows.to(device)]
                alt_materials = alternative_cpu.to(device=device, dtype=torch.long)
                alt_output = inverse(alt_qsca, alt_materials)
                alt_pgff = pgff_forward(pgff, alt_output["geometry"], alt_materials)
                counter_loss = normalized_smooth_l1_loss(
                    alt_pgff["q_sca"],
                    alt_qsca,
                    beta=args.smooth_l1_beta,
                    scale_floor=args.spectral_scale_floor,
                )
            total_loss = (
                native_loss
                + float(args.counterfactual_weight) * counter_loss
                + float(anchor_weight) * anchor_loss
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite training loss at epoch={epoch}, batch={batch_id}."
            )
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradients_ok, raw_grad_norm = finite_inverse_gradients(inverse)
        if not gradients_ok:
            raise FloatingPointError(
                f"Inverse gradients are absent or non-finite at epoch={epoch}, batch={batch_id}."
            )
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            inverse.parameters(), max_norm=float(args.gradient_clip)
        )
        if not torch.isfinite(torch.as_tensor(clipped_norm)):
            raise FloatingPointError("Gradient clipping produced a non-finite norm.")
        scaler.step(optimizer)
        scaler.update()
        assert_pgff_has_no_gradients(pgff)

        with torch.no_grad():
            scale = qsca.amax(dim=1).clamp_min(args.spectral_scale_floor)
            native_nmae = (
                (native_pgff["q_sca"] - qsca).abs().mean(dim=1) / scale
            ).sum()
        sums["samples"] += batch_size
        sums["total_loss"] += float(total_loss.detach().cpu()) * batch_size
        sums["native_spectral_loss"] += float(native_loss.detach().cpu()) * batch_size
        sums["counterfactual_spectral_loss"] += float(counter_loss.detach().cpu()) * batch_size
        sums["geometry_anchor_loss"] += float(anchor_loss.detach().cpu()) * batch_size
        sums["native_nmae"] += float(native_nmae.detach().cpu())
        sums["gradient_norm"] += float(raw_grad_norm)
        sums["gradient_steps"] += 1

    count = max(int(sums["samples"]), 1)
    steps = max(int(sums["gradient_steps"]), 1)
    return {
        "train_total_loss": sums["total_loss"] / count,
        "train_native_spectral_loss": sums["native_spectral_loss"] / count,
        "train_counterfactual_spectral_loss": sums["counterfactual_spectral_loss"] / count,
        "train_geometry_anchor_loss": sums["geometry_anchor_loss"] / count,
        "train_native_mean_nmae": sums["native_nmae"] / count,
        "train_mean_gradient_norm": sums["gradient_norm"] / steps,
        "geometry_anchor_weight": float(anchor_weight),
        "counterfactual_enabled": bool(counter_enabled),
    }


@torch.no_grad()
def evaluate_native(
    *,
    inverse: SpectralMaterialInverseNet,
    pgff: torch.nn.Module,
    data: Dataset,
    device: torch.device,
    wavelengths_nm: np.ndarray,
    batch_size: int,
    num_workers: int,
    scale_floor: float,
    collect_predictions: bool = False,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    inverse.eval()
    pgff.eval()
    loader = make_loader(
        data,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        epoch=0,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    wavelengths = torch.as_tensor(wavelengths_nm, dtype=torch.float32, device=device)
    nmae_values = []
    nrmse_values = []
    r2_values = []
    peak_values = []
    geometry_mae_values = []
    rows = []

    for qsca_cpu, materials_cpu, geometry_cpu, indices in loader:
        qsca = qsca_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        materials = materials_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        true_geometry = geometry_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        output = inverse(qsca, materials)
        prediction = pgff_forward(pgff, output["geometry"], materials)["q_sca"]
        metrics = per_sample_metrics(
            prediction,
            qsca,
            wavelengths,
            scale_floor=scale_floor,
        )
        geometry_mae = (output["geometry"] - true_geometry).abs().mean(dim=1)

        nmae_values.append(metrics["nmae"].cpu())
        nrmse_values.append(metrics["nrmse"].cpu())
        r2_values.append(metrics["r2"].cpu())
        peak_values.append(metrics["peak_error_nm"].cpu())
        geometry_mae_values.append(geometry_mae.cpu())

        if collect_predictions:
            pred_geometry = output["geometry"].cpu().numpy()
            true_geometry_np = geometry_cpu.numpy()
            material_np = materials_cpu.numpy()
            for local, dataset_index in enumerate(indices.numpy().tolist()):
                rows.append(
                    {
                        "dataset_index": int(dataset_index),
                        "m0": int(material_np[local, 0]),
                        "m1": int(material_np[local, 1]),
                        "m2": int(material_np[local, 2]),
                        "m3": int(material_np[local, 3]),
                        "true_r_core": float(true_geometry_np[local, 0]),
                        "true_t1": float(true_geometry_np[local, 1]),
                        "true_t2": float(true_geometry_np[local, 2]),
                        "true_t3": float(true_geometry_np[local, 3]),
                        "pred_r_core": float(pred_geometry[local, 0]),
                        "pred_t1": float(pred_geometry[local, 1]),
                        "pred_t2": float(pred_geometry[local, 2]),
                        "pred_t3": float(pred_geometry[local, 3]),
                        "PGFF_NMAE_Qsca": float(metrics["nmae"][local].cpu()),
                        "PGFF_NRMSE_Qsca": float(metrics["nrmse"][local].cpu()),
                        "PGFF_R2_Qsca": float(metrics["r2"][local].cpu()),
                        "PGFF_peak_error_nm": float(metrics["peak_error_nm"][local].cpu()),
                        "geometry_MAE_nm": float(geometry_mae[local].cpu()),
                    }
                )

    nmae = torch.cat(nmae_values).numpy()
    nrmse = torch.cat(nrmse_values).numpy()
    r2 = torch.cat(r2_values).numpy()
    peak = torch.cat(peak_values).numpy()
    geometry_mae = torch.cat(geometry_mae_values).numpy()
    summary = {
        "mean_nmae": float(np.mean(nmae)),
        "median_nmae": float(np.median(nmae)),
        "p95_nmae": float(np.quantile(nmae, 0.95)),
        "mean_nrmse": float(np.mean(nrmse)),
        "median_nrmse": float(np.median(nrmse)),
        "mean_r2": float(np.nanmean(r2)),
        "median_r2": float(np.nanmedian(r2)),
        "mean_peak_error_nm": float(np.mean(peak)),
        "median_peak_error_nm": float(np.median(peak)),
        "mean_geometry_mae_nm": float(np.mean(geometry_mae)),
        "sample_count": int(len(nmae)),
    }
    frame = pd.DataFrame(rows) if collect_predictions else None
    return summary, frame


@torch.no_grad()
def evaluate_counterfactual_probe(
    *,
    inverse: SpectralMaterialInverseNet,
    pgff: torch.nn.Module,
    data: TandemPairDataset,
    legal_sequences_cpu: torch.Tensor,
    device: torch.device,
    sample_count: int,
    candidates_per_sample: int,
    batch_size: int,
    seed: int,
    scale_floor: float,
) -> Dict[str, float]:
    """Evaluate best PGFF match among a fixed small set of non-native materials."""
    if sample_count <= 0 or candidates_per_sample <= 0 or len(data) == 0:
        return {
            "probe_sample_count": 0,
            "probe_candidates_per_sample": 0,
            "best_alternative_mean_nmae": float("nan"),
            "best_alternative_median_nmae": float("nan"),
        }
    inverse.eval()
    pgff.eval()
    count = min(int(sample_count), len(data))
    probe_positions = np.linspace(0, len(data) - 1, num=count, dtype=np.int64)
    qsca_cpu = data.qsca[probe_positions]
    native_cpu = data.materials[probe_positions]
    rows, alternatives = sample_counterfactual_sequences(
        native_cpu,
        legal_sequences_cpu,
        count_per_sample=int(candidates_per_sample),
        seed=int(seed),
    )
    targets = qsca_cpu[rows]
    all_nmae = []
    for start in range(0, len(targets), int(batch_size)):
        end = min(start + int(batch_size), len(targets))
        qsca = targets[start:end].to(device=device, dtype=torch.float32)
        materials = alternatives[start:end].to(device=device, dtype=torch.long)
        geometry = inverse(qsca, materials)["geometry"]
        pred = pgff_forward(pgff, geometry, materials)["q_sca"]
        scale = qsca.amax(dim=1).clamp_min(float(scale_floor))
        all_nmae.append(((pred - qsca).abs().mean(dim=1) / scale).cpu())
    matrix = torch.cat(all_nmae).reshape(count, int(candidates_per_sample)).numpy()
    best = matrix.min(axis=1)
    return {
        "probe_sample_count": int(count),
        "probe_candidates_per_sample": int(candidates_per_sample),
        "best_alternative_mean_nmae": float(best.mean()),
        "best_alternative_median_nmae": float(np.median(best)),
        "best_alternative_p95_nmae": float(np.quantile(best, 0.95)),
    }


def checkpoint_payload(
    *,
    epoch: int,
    inverse: SpectralMaterialInverseNet,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    model_config: ModelConfig,
    args: argparse.Namespace,
    metrics: Mapping[str, object],
    best_val_nmae: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> dict:
    return {
        "stage": STAGE,
        "version": VERSION,
        "epoch": int(epoch),
        "model_class": "SpectralMaterialInverseNet",
        "model_config": asdict(model_config),
        "model_state_dict": inverse.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_arguments": vars(args),
        "metrics": dict(metrics),
        "best_val_mean_nmae": float(best_val_nmae),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
        "rng": rng_payload(),
        "data_policy": {
            "train_split": "forward train only",
            "model_selection": "forward validation only",
            "test_split_used": False,
            "blind_targets_used": False,
            "pgff_parameters_frozen": True,
        },
    }


def write_history(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    frame = pd.DataFrame(records)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def plot_history(history: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(history["epoch"], history["train_total_loss"], label="Training total loss")
    ax.plot(history["epoch"], history["train_native_spectral_loss"], label="Native spectral loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "tandem_loss_history.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "tandem_loss_history.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(history["epoch"], history["validation_mean_nmae"], label="Validation mean NMAE")
    ax.plot(history["epoch"], history["validation_median_nmae"], label="Validation median NMAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PGFF-reconstructed NMAE")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_nmae_history.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "validation_nmae_history.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 2
        args.patience = 2
        args.counterfactual_probe_interval = 1
        args.counterfactual_probe_samples = min(args.counterfactual_probe_samples, 64)
    validate_args(args)
    start_time = time.perf_counter()
    set_seed(args.seed)

    output_dir = output_directory(args)
    prepare_output_directory(args, output_dir)
    history_path = output_dir / "training_history.csv"
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_checkpoint.pt"

    dataset = load_forward_dataset(args.dataset)
    split = load_split_bundle(dataset, args.baseline_dir)
    prepared = load_prepared_manifest(args.prepared_data)
    verify_frozen_inputs(args, dataset, split, prepared)

    train_indices = split.train
    validation_indices = split.validation
    formal_run = not args.smoke_test
    if args.smoke_test:
        train_indices = train_indices[: min(len(train_indices), 2048)]
        validation_indices = validation_indices[: min(len(validation_indices), 512)]

    train_data = TandemPairDataset(dataset, train_indices)
    validation_data = TandemPairDataset(dataset, validation_indices)
    legal_sequences_cpu = torch.as_tensor(
        prepared["legal_material_sequences"].astype(np.int64), dtype=torch.long
    )
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    qsca_mean = torch.as_tensor(prepared["qsca_mean_train"], dtype=torch.float32)
    qsca_std = torch.as_tensor(prepared["qsca_std_train"], dtype=torch.float32)
    model_config = ModelConfig(
        n_wavelengths=int(len(dataset["wavelengths"])),
        n_materials=int(len(dataset["material_names"])),
        width=int(args.width),
        depth=int(args.depth),
        dropout=float(args.dropout),
    )
    inverse = SpectralMaterialInverseNet(
        n_wavelengths=model_config.n_wavelengths,
        n_materials=model_config.n_materials,
        qsca_mean=qsca_mean,
        qsca_std=qsca_std,
        width=model_config.width,
        depth=model_config.depth,
        dropout=model_config.dropout,
    ).to(device)
    pgff = load_frozen_pgff(dataset, args.baseline_dir, args.pgff_dir, device)
    if any(parameter.requires_grad for parameter in pgff.parameters()):
        raise RuntimeError("PGFF must be completely frozen before tandem training.")

    optimizer = torch.optim.AdamW(
        inverse.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
        eta_min=float(args.minimum_learning_rate),
    )
    scaler = create_grad_scaler(amp_enabled)

    start_epoch = 1
    best_val_nmae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_records = []

    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint is missing: {last_path}")
        checkpoint = load_torch_checkpoint(last_path, device)
        if checkpoint.get("model_config") != asdict(model_config):
            raise RuntimeError("Resume checkpoint model configuration differs from current arguments.")
        inverse.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng(checkpoint.get("rng", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_nmae = float(checkpoint["best_val_mean_nmae"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        if history_path.is_file():
            history_records = pd.read_csv(history_path).to_dict(orient="records")

    print("Stage 02 frozen-PGFF tandem training")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Training samples: {len(train_data)}; validation samples: {len(validation_data)}")
    print(f"Legal material sequences: {len(legal_sequences_cpu)}")
    print(f"Inverse parameters: {sum(p.numel() for p in inverse.parameters()):,}")
    print(f"Frozen PGFF parameters: {sum(p.numel() for p in pgff.parameters()):,}")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            inverse=inverse,
            pgff=pgff,
            train_data=train_data,
            legal_sequences_cpu=legal_sequences_cpu,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
        )
        validation_metrics, _ = evaluate_native(
            inverse=inverse,
            pgff=pgff,
            data=validation_data,
            device=device,
            wavelengths_nm=np.asarray(dataset["wavelengths"], dtype=np.float32),
            batch_size=args.validation_batch_size,
            num_workers=args.num_workers,
            scale_floor=args.spectral_scale_floor,
            collect_predictions=False,
        )
        probe_metrics = {
            "probe_sample_count": 0,
            "probe_candidates_per_sample": 0,
            "best_alternative_mean_nmae": float("nan"),
            "best_alternative_median_nmae": float("nan"),
            "best_alternative_p95_nmae": float("nan"),
        }
        if epoch % int(args.counterfactual_probe_interval) == 0:
            probe_metrics = evaluate_counterfactual_probe(
                inverse=inverse,
                pgff=pgff,
                data=validation_data,
                legal_sequences_cpu=legal_sequences_cpu,
                device=device,
                sample_count=args.counterfactual_probe_samples,
                candidates_per_sample=args.counterfactual_probe_candidates,
                batch_size=args.validation_batch_size,
                seed=args.seed + 7_000_001,
                scale_floor=args.spectral_scale_floor,
            )

        current_lr = learning_rate(optimizer)
        record = {
            "epoch": int(epoch),
            "learning_rate": current_lr,
            **train_metrics,
            "validation_mean_nmae": validation_metrics["mean_nmae"],
            "validation_median_nmae": validation_metrics["median_nmae"],
            "validation_p95_nmae": validation_metrics["p95_nmae"],
            "validation_mean_nrmse": validation_metrics["mean_nrmse"],
            "validation_mean_r2": validation_metrics["mean_r2"],
            "validation_mean_peak_error_nm": validation_metrics["mean_peak_error_nm"],
            "validation_mean_geometry_mae_nm": validation_metrics["mean_geometry_mae_nm"],
            "counterfactual_probe_mean_nmae": probe_metrics["best_alternative_mean_nmae"],
            "counterfactual_probe_median_nmae": probe_metrics["best_alternative_median_nmae"],
            "epoch_seconds": float(time.perf_counter() - epoch_start),
        }
        history_records.append(record)
        write_history(history_path, history_records)

        improved = (
            validation_metrics["mean_nmae"]
            < best_val_nmae - float(args.minimum_improvement)
        )
        if improved:
            best_val_nmae = float(validation_metrics["mean_nmae"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            best_payload = checkpoint_payload(
                epoch=epoch,
                inverse=inverse,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=model_config,
                args=args,
                metrics={"validation_native": validation_metrics, "counterfactual_probe": probe_metrics},
                best_val_nmae=best_val_nmae,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
            atomic_torch_save(best_path, best_payload)
        else:
            epochs_without_improvement += 1

        scheduler.step()
        last_payload = checkpoint_payload(
            epoch=epoch,
            inverse=inverse,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=model_config,
            args=args,
            metrics={"validation_native": validation_metrics, "counterfactual_probe": probe_metrics},
            best_val_nmae=best_val_nmae,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
        )
        atomic_torch_save(last_path, last_payload)

        probe_text = (
            f", alt-probe={probe_metrics['best_alternative_mean_nmae']:.6f}"
            if np.isfinite(probe_metrics["best_alternative_mean_nmae"])
            else ""
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train={train_metrics['train_total_loss']:.6f}, "
            f"native-NMAE={train_metrics['train_native_mean_nmae']:.6f}, "
            f"val-NMAE={validation_metrics['mean_nmae']:.6f}, "
            f"val-R2={validation_metrics['mean_r2']:.6f}{probe_text}, "
            f"lr={current_lr:.3e}, best={best_val_nmae:.6f}@{best_epoch}"
        )

        if epochs_without_improvement >= int(args.patience):
            print(
                f"Early stopping after {epochs_without_improvement} epochs without "
                f"a validation-NMAE improvement >= {args.minimum_improvement:.2e}."
            )
            break

    if not best_path.is_file():
        raise RuntimeError("Training finished without writing a best-model checkpoint.")
    best_checkpoint = load_torch_checkpoint(best_path, device)
    inverse.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    final_validation, validation_predictions = evaluate_native(
        inverse=inverse,
        pgff=pgff,
        data=validation_data,
        device=device,
        wavelengths_nm=np.asarray(dataset["wavelengths"], dtype=np.float32),
        batch_size=args.validation_batch_size,
        num_workers=args.num_workers,
        scale_floor=args.spectral_scale_floor,
        collect_predictions=True,
    )
    if validation_predictions is None:
        raise RuntimeError("Validation predictions were not collected.")
    predicted_geometry = validation_predictions[
        ["pred_r_core", "pred_t1", "pred_t2", "pred_t3"]
    ].to_numpy(dtype=np.float64)
    predicted_outer = predicted_geometry.sum(axis=1)
    validation_predictions["pred_outer_radius"] = predicted_outer
    legal_mask = geometry_bounds_mask(
        predicted_geometry, atol=GEOMETRY_CHECK_ATOL_NM
    )
    geometry_diagnostics = {
        "check_atol_nm": float(GEOMETRY_CHECK_ATOL_NM),
        "hard_outer_radius_nm": float(MAX_OUTER_RADIUS),
        "sample_count": int(len(predicted_geometry)),
        "illegal_count": int((~legal_mask).sum()),
        "minimum_r_core_nm": float(predicted_geometry[:, 0].min()),
        "maximum_r_core_nm": float(predicted_geometry[:, 0].max()),
        "minimum_shell_thickness_nm": float(predicted_geometry[:, 1:].min()),
        "maximum_shell_thickness_nm": float(predicted_geometry[:, 1:].max()),
        "minimum_outer_radius_nm": float(predicted_outer.min()),
        "maximum_outer_radius_nm": float(predicted_outer.max()),
        "maximum_outer_excess_nm": float(
            np.maximum(predicted_outer - MAX_OUTER_RADIUS, 0.0).max()
        ),
    }
    save_json_atomic(
        output_dir / "validation_geometry_diagnostics.json", geometry_diagnostics
    )
    if not np.all(legal_mask):
        bad = validation_predictions.loc[~legal_mask].copy()
        bad["outer_excess_nm"] = np.maximum(
            bad["pred_outer_radius"].to_numpy(dtype=np.float64)
            - MAX_OUTER_RADIUS,
            0.0,
        )
        bad.to_csv(
            output_dir / "illegal_validation_geometries.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise RuntimeError(
            "Best inverse model produced illegal validation geometry after the "
            f"boundary-safe map: illegal={int((~legal_mask).sum())}/"
            f"{len(legal_mask)}, max_outer={predicted_outer.max():.9f} nm. "
            "See validation_geometry_diagnostics.json and "
            "illegal_validation_geometries.csv."
        )
    validation_predictions.to_csv(
        output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig"
    )

    history = pd.read_csv(history_path)
    plot_history(history, output_dir)
    summary = {
        "stage": STAGE,
        "version": VERSION,
        "status": "passed",
        "formal_run": bool(formal_run),
        "wall_time_seconds": float(time.perf_counter() - start_time),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "epochs_completed": int(history["epoch"].max()),
        "best_validation_mean_nmae": float(best_checkpoint["best_val_mean_nmae"]),
        "final_best_checkpoint_validation": final_validation,
        "validation_geometry_diagnostics": geometry_diagnostics,
        "data_policy": {
            "training_examples": "forward train split only",
            "model_selection": "forward validation split only",
            "test_split_evaluated": False,
            "blind_targets_read": False,
            "legacy_inverse_results_read": False,
            "counterfactual_pairs_source": "train spectra + exact legal material registry",
        },
        "optimization": {
            "primary_loss": "per-target-peak-normalized Smooth-L1 Qsca cycle loss",
            "native_pair_weight": 1.0,
            "counterfactual_pair_weight": float(args.counterfactual_weight),
            "counterfactual_pairs_per_native": int(args.counterfactual_per_sample),
            "geometry_anchor_initial_weight": float(args.geometry_anchor_weight),
            "geometry_anchor_decay_epochs": int(args.geometry_anchor_decay_epochs),
            "checkpoint_selection_metric": "validation native mean NMAE",
            "pgff_parameters_frozen": True,
        },
        "dimensions": {
            "train_samples": int(len(train_data)),
            "validation_samples": int(len(validation_data)),
            "test_samples_untouched": int(len(split.test)),
            "wavelengths": int(model_config.n_wavelengths),
            "materials": int(model_config.n_materials),
            "legal_material_sequences": int(len(legal_sequences_cpu)),
            "inverse_parameter_count": int(sum(p.numel() for p in inverse.parameters())),
            "frozen_pgff_parameter_count": int(sum(p.numel() for p in pgff.parameters())),
        },
        "model_config": asdict(model_config),
        "arguments": vars(args),
        "inputs": {
            "dataset": artifact(args.dataset),
            "prepared_data": artifact(args.prepared_data),
            "pgff_checkpoint": artifact(args.pgff_dir / "best_model.pt"),
        },
        "outputs": {
            "best_model": artifact(best_path),
            "last_checkpoint": artifact(last_path),
            "training_history": artifact(history_path),
            "validation_predictions": artifact(output_dir / "validation_predictions.csv"),
            "validation_geometry_diagnostics": artifact(
                output_dir / "validation_geometry_diagnostics.json"
            ),
        },
    }
    save_json_atomic(output_dir / "training_summary.json", summary)
    save_json_atomic(
        output_dir / "inverse_model_config.json",
        {
            "model_class": "SpectralMaterialInverseNet",
            "model_config": asdict(model_config),
            "prepared_data": str(args.prepared_data.resolve()),
            "best_model": str(best_path.resolve()),
        },
    )
    print("Stage 02 training PASSED")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Best validation mean NMAE: {summary['best_validation_mean_nmae']:.6f}")
    print(f"Best model: {best_path}")
    print("TEST split and blind targets remain untouched.")


if __name__ == "__main__":
    main()
