"""Run and summarize the frozen, parameter-matched PGFF ablation study.

The five primary variants share the same registered modules, hidden width,
optimizer, scheduler, data split, loss weights, epoch budget, and paired
training seeds.  Only the information switches declared in the protocol are
changed.  The optional ``full_pgff_no_curve`` run is a secondary loss ablation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_shell_optical_design.forward_models.controlled_ablation import (  # noqa: E402
    build_ablation_model,
    input_informed_parameter_count,
    parameter_count,
    variants_from_protocol,
)
from train_physics_guided_fourier_network import (  # noqa: E402
    PGFFLoss,
    compute_physics_statistics,
    load_data,
    make_loader,
    predict,
    regression_metrics,
    run_epoch,
    save_json,
    save_plots,
    scattering_metrics,
    set_seed,
    setup_logger,
)


DEFAULT_PROTOCOL = (
    PROJECT_DIR / "configs" / "forward" / "controlled_ablation" / "protocol.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parameter-matched PGFF component ablations."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variants. Defaults to the five primary variants.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated integer seeds. Defaults to the frozen protocol.",
    )
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-run-plots", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. Overrides are marked non-formal.",
    )
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        protocol = json.load(handle)
    required = (
        "protocol_version",
        "dataset_path",
        "split_dir",
        "output_dir",
        "seeds",
        "variants",
        "primary_variants",
        "model",
        "training",
    )
    missing = [key for key in required if key not in protocol]
    if missing:
        raise KeyError(f"Protocol is missing required keys: {missing}")
    return protocol


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def parse_csv_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("An explicitly supplied comma-separated list is empty.")
    return result


def parse_seed_list(value: str | None, defaults: Iterable[int]) -> list[int]:
    if value is None:
        return [int(seed) for seed in defaults]
    return [int(item) for item in parse_csv_names(value) or []]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def protocol_selection(
    protocol: dict[str, Any], args: argparse.Namespace
) -> tuple[list[str], list[int]]:
    variants = parse_csv_names(args.variants)
    if variants is None:
        variants = list(protocol["primary_variants"])
        if args.include_secondary:
            variants.extend(protocol.get("secondary_variants", []))
    unknown = sorted(set(variants) - set(protocol["variants"]))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if len(variants) != len(set(variants)):
        raise ValueError("Variant list contains duplicates.")
    seeds = parse_seed_list(args.seeds, protocol["seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed list contains duplicates.")
    return variants, seeds


def build_model_for_variant(
    variant,
    protocol: dict[str, Any],
    bundle: dict[str, Any],
    physics_mean: np.ndarray,
    physics_std: np.ndarray,
) -> torch.nn.Module:
    return build_ablation_model(
        variant=variant,
        geometry_mean=torch.from_numpy(bundle["geometry_mean"]),
        geometry_std=torch.from_numpy(bundle["geometry_std"]),
        wavelengths_nm=torch.from_numpy(bundle["wavelengths"]),
        material_n=torch.from_numpy(bundle["material_n"]),
        material_k=torch.from_numpy(bundle["material_k"]),
        physics_mean=torch.from_numpy(physics_mean),
        physics_std=torch.from_numpy(physics_std),
        model_config=protocol["model"],
        fourier_seed=int(protocol["fourier_seed"]),
    )


def audit_parameter_matching(
    selected_names: list[str],
    variant_specs: dict,
    protocol: dict[str, Any],
    bundle: dict[str, Any],
    physics_mean: np.ndarray,
    physics_std: np.ndarray,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    informed_counts: dict[str, int] = {}
    for name in selected_names:
        model = build_model_for_variant(
            variant_specs[name], protocol, bundle, physics_mean, physics_std
        )
        counts[name] = parameter_count(model)
        informed_counts[name] = input_informed_parameter_count(
            model, variant_specs[name]
        )
    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) != 1:
        raise RuntimeError(f"Registered parameter counts are not identical: {counts}")
    return {
        "parameter_counts": counts,
        "identical": True,
        "common_parameter_count": unique_counts[0],
        "input_informed_parameter_counts": informed_counts,
        "minimum_input_informed_fraction": min(informed_counts.values())
        / unique_counts[0],
    }


def run_one(
    *,
    variant_name: str,
    seed: int,
    variant_specs: dict,
    protocol: dict[str, Any],
    bundle: dict[str, Any],
    physics_mean: np.ndarray,
    physics_std: np.ndarray,
    study_output: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    save_run_plots: bool,
    formal_protocol: bool,
    provenance: dict[str, Any],
) -> str:
    variant = variant_specs[variant_name]
    output_dir = study_output / variant_name / f"seed_{seed}"
    metrics_path = output_dir / "test_metrics.json"
    if metrics_path.is_file():
        return "skipped_complete"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info(
        "Controlled ablation | variant=%s | seed=%d | device=%s",
        variant_name,
        seed,
        device,
    )
    set_seed(seed)
    training = protocol["training"]
    use_amp = bool(training["use_amp"] and device.type == "cuda")
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        bundle,
        bundle["train_idx"],
        batch_size,
        True,
        int(training["num_workers"]),
        pin_memory,
        seed,
    )
    val_loader = make_loader(
        bundle,
        bundle["val_idx"],
        batch_size,
        False,
        int(training["num_workers"]),
        pin_memory,
        seed,
    )
    test_loader = make_loader(
        bundle,
        bundle["test_idx"],
        batch_size,
        False,
        int(training["num_workers"]),
        pin_memory,
        seed,
    )

    model = build_model_for_variant(
        variant, protocol, bundle, physics_mean, physics_std
    ).to(device)
    n_parameters = parameter_count(model)
    criterion = PGFFLoss(
        training["lambda_abs"],
        training["lambda_ext"],
        variant.lambda_curve,
        training["huber_beta"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(training["scheduler_patience"]),
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_path = output_dir / "best_model.pt"
    best_val_sca = float("inf")
    best_epoch = 0
    patience_counter = 0
    component_names = ("total", "sca", "abs", "ext", "curve")
    history: dict[str, list[float]] = {"epoch": [], "learning_rate": []}
    for split in ("train", "val"):
        for component in component_names:
            history[f"{split}_{component}"] = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        curve_fraction = min(
            1.0, epoch / max(1, int(training["curve_warmup_epochs"]))
        )
        train_values = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            curve_fraction,
        )
        val_values = run_epoch(
            model, val_loader, criterion, device, None, None, use_amp, 1.0
        )
        scheduler.step(val_values["sca"])
        history["epoch"].append(epoch)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        for component in component_names:
            history[f"train_{component}"].append(train_values[component])
            history[f"val_{component}"].append(val_values[component])

        improved = val_values["sca"] < best_val_sca - 1e-10
        if improved:
            best_val_sca = val_values["sca"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": variant_name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_q_sca_mse": best_val_sca,
                    "parameter_count": n_parameters,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            logger.info(
                "Epoch %03d | train_sca=%.7f | val_sca=%.7f | "
                "val_total=%.7f | lr=%.2e",
                epoch,
                train_values["sca"],
                val_values["sca"],
                val_values["total"],
                optimizer.param_groups[0]["lr"],
            )
        if patience_counter >= int(training["patience"]):
            logger.info(
                "Early stopping at epoch %d; best epoch=%d", epoch, best_epoch
            )
            break

    training_seconds = time.time() - start_time
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    inference_start = time.perf_counter()
    predictions = predict(model, test_loader, device)
    inference_seconds = time.perf_counter() - inference_start
    metrics_sca, sample_mae = scattering_metrics(
        predictions["pred_sca"], predictions["true_sca"], bundle["wavelengths"]
    )
    metrics_sca["99th_percentile_sample_MAE"] = float(
        np.quantile(sample_mae, 0.99)
    )
    metrics_abs = regression_metrics(
        predictions["pred_abs"], predictions["true_abs"]
    )
    metrics_ext = regression_metrics(
        predictions["pred_ext"], predictions["true_ext"]
    )
    physical_residual = (
        predictions["pred_ext"]
        - predictions["pred_sca"]
        - predictions["pred_abs"]
    )
    metrics = {
        "study": protocol["study_name"],
        "protocol_version": protocol["protocol_version"],
        "formal_protocol_run": formal_protocol,
        "variant": variant_name,
        "role": variant.role,
        "seed": seed,
        "fourier_seed": int(protocol["fourier_seed"]),
        "parameter_count": n_parameters,
        "best_epoch": best_epoch,
        "epochs_completed": len(history["epoch"]),
        "best_validation_q_sca_MSE": best_val_sca,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "test_samples_per_second": len(bundle["test_idx"]) / inference_seconds,
        "device": str(device),
        "torch_version": torch.__version__,
        "Q_sca": metrics_sca,
        "Q_abs": metrics_abs,
        "Q_ext": metrics_ext,
        "max_physical_residual": float(np.max(np.abs(physical_residual))),
        "negative_prediction_count": int(
            np.sum(predictions["pred_sca"] < 0)
            + np.sum(predictions["pred_abs"] < 0)
            + np.sum(predictions["pred_ext"] < 0)
        ),
        "provenance": provenance,
    }
    save_json(metrics_path, metrics)
    save_json(
        output_dir / "run_config.json",
        {
            "variant": variant.__dict__,
            "seed": seed,
            "fourier_seed": int(protocol["fourier_seed"]),
            "model": protocol["model"],
            "training": {
                **training,
                "epochs": epochs,
                "batch_size": batch_size,
                "lambda_curve": variant.lambda_curve,
            },
            "formal_protocol_run": formal_protocol,
            "parameter_count": n_parameters,
        },
    )
    np.savez_compressed(
        output_dir / "training_history.npz",
        **{key: np.asarray(value) for key, value in history.items()},
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        sample_ids=bundle["test_idx"],
        wavelengths=bundle["wavelengths"],
        sample_mae=sample_mae,
        **predictions,
    )
    error = predictions["pred_sca"] - predictions["true_sca"]
    with (output_dir / "per_wavelength_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("wavelength_nm", "MAE", "RMSE"))
        for wavelength, column in zip(bundle["wavelengths"], error.T):
            writer.writerow(
                (
                    float(wavelength),
                    float(np.mean(np.abs(column))),
                    float(np.sqrt(np.mean(column**2))),
                )
            )
    if save_run_plots:
        save_plots(
            history,
            predictions,
            bundle["wavelengths"],
            output_dir,
            project_path(protocol["split_dir"]),
        )
    logger.info(
        "Completed variant=%s seed=%d in %.2f min | Q_sca MAE=%.8f",
        variant_name,
        seed,
        training_seconds / 60.0,
        metrics_sca["MAE"],
    )
    return "completed"


METRIC_FIELDS = {
    "q_sca_mae": ("Q_sca", "MAE"),
    "q_sca_rmse": ("Q_sca", "RMSE"),
    "q_sca_r2": ("Q_sca", "R2"),
    "q_sca_p95_sample_mae": ("Q_sca", "95th_percentile_sample_MAE"),
    "q_sca_p99_sample_mae": ("Q_sca", "99th_percentile_sample_MAE"),
    "peak_wavelength_mae_nm": ("Q_sca", "peak_wavelength_MAE_nm"),
    "peak_amplitude_mae": ("Q_sca", "peak_amplitude_MAE"),
    "q_abs_mae": ("Q_abs", "MAE"),
    "q_ext_mae": ("Q_ext", "MAE"),
}


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if df in table:
        return table[df]
    lower = max(key for key in table if key < df) if df > 1 else 1
    return table[lower] if df <= 30 else 1.96


def mean_sd_ci(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    sd = float(array.std(ddof=1))
    half = t_critical_95(len(array) - 1) * sd / math.sqrt(len(array))
    return mean, sd, mean - half, mean + half


def summarize_study(
    study_output: Path,
    protocol: dict[str, Any],
    selected_variants: list[str],
    selected_seeds: list[int],
    parameter_audit: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in selected_variants:
        for seed in selected_seeds:
            path = study_output / variant / f"seed_{seed}" / "test_metrics.json"
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8-sig") as handle:
                metrics = json.load(handle)
            row: dict[str, Any] = {
                "variant": variant,
                "seed": seed,
                "parameter_count": metrics["parameter_count"],
                "best_epoch": metrics["best_epoch"],
                "training_seconds": metrics["training_seconds"],
                "formal_protocol_run": metrics["formal_protocol_run"],
            }
            for label, path_keys in METRIC_FIELDS.items():
                row[label] = metrics[path_keys[0]][path_keys[1]]
            rows.append(row)

    summary_dir = study_output / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (summary_dir / "per_run_metrics.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for variant in selected_variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            continue
        summary_row: dict[str, Any] = {
            "variant": variant,
            "n_seeds": len(variant_rows),
            "parameter_count": variant_rows[0]["parameter_count"],
        }
        for label in METRIC_FIELDS:
            mean, sd, lower, upper = mean_sd_ci(
                [float(row[label]) for row in variant_rows]
            )
            summary_row[f"{label}_mean"] = mean
            summary_row[f"{label}_sd"] = sd
            summary_row[f"{label}_ci95_low"] = lower
            summary_row[f"{label}_ci95_high"] = upper
        summary_rows.append(summary_row)
    if summary_rows:
        with (summary_dir / "summary_by_variant.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    paired_rows: list[dict[str, Any]] = []
    full_by_seed = {
        int(row["seed"]): row for row in rows if row["variant"] == "full_pgff"
    }
    for variant in selected_variants:
        if variant == "full_pgff":
            continue
        comparator = {int(row["seed"]): row for row in rows if row["variant"] == variant}
        common = sorted(set(full_by_seed) & set(comparator))
        if not common:
            continue
        result: dict[str, Any] = {
            "variant": variant,
            "reference": "full_pgff",
            "paired_seeds": ";".join(map(str, common)),
            "n_pairs": len(common),
        }
        for label in METRIC_FIELDS:
            differences = [
                float(comparator[seed][label]) - float(full_by_seed[seed][label])
                for seed in common
            ]
            mean, sd, lower, upper = mean_sd_ci(differences)
            result[f"delta_{label}_mean"] = mean
            result[f"delta_{label}_sd"] = sd
            result[f"delta_{label}_ci95_low"] = lower
            result[f"delta_{label}_ci95_high"] = upper
        paired_rows.append(result)
    if paired_rows:
        with (summary_dir / "paired_differences_vs_full.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)

    expected = len(selected_variants) * len(selected_seeds)
    status = {
        "protocol_version": protocol["protocol_version"],
        "selected_variants": selected_variants,
        "selected_seeds": selected_seeds,
        "expected_runs": expected,
        "completed_runs": len(rows),
        "complete": len(rows) == expected,
        "parameter_audit": parameter_audit,
        "warning": (
            "Five paired seeds support uncertainty reporting but not a strong "
            "standalone significance claim. Report effect sizes and confidence "
            "intervals without binary significance language."
        ),
    }
    save_json(summary_dir / "completion_status.json", status)

    if summary_rows:
        by_name = {row["variant"]: row for row in summary_rows}
        lines = [
            "# Controlled forward-model ablation",
            "",
            "All variants have identical registered parameter counts and use paired seeds.",
            "",
            "| Variant | Seeds | Parameters | Q_sca MAE (mean ± SD) | RMSE (mean ± SD) | R² (mean ± SD) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for variant in selected_variants:
            if variant not in by_name:
                continue
            row = by_name[variant]
            lines.append(
                f"| {variant} | {row['n_seeds']} | {row['parameter_count']} | "
                f"{row['q_sca_mae_mean']:.8f} ± {row['q_sca_mae_sd']:.8f} | "
                f"{row['q_sca_rmse_mean']:.8f} ± {row['q_sca_rmse_sd']:.8f} | "
                f"{row['q_sca_r2_mean']:.8f} ± {row['q_sca_r2_sd']:.8f} |"
            )
        (summary_dir / "paper_ablation_table.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return status


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    selected_variants, selected_seeds = protocol_selection(protocol, args)
    variants = variants_from_protocol(protocol)
    data_path = project_path(protocol["dataset_path"])
    split_dir = project_path(protocol["split_dir"])
    study_output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_path(protocol["output_dir"])
    )
    study_output.mkdir(parents=True, exist_ok=True)

    formal_protocol = (
        args.epochs is None
        and args.batch_size is None
        and args.output_dir is None
        and selected_seeds == [int(seed) for seed in protocol["seeds"]]
        and selected_variants == list(protocol["primary_variants"])
    )
    epochs = int(args.epochs or protocol["training"]["epochs"])
    batch_size = int(args.batch_size or protocol["training"]["batch_size"])
    device = choose_device(args.device)

    load_args = argparse.Namespace(data_path=data_path, baseline_dir=split_dir)
    study_logger = setup_logger(study_output)
    bundle = load_data(load_args, study_logger)
    physics_mean, physics_std = compute_physics_statistics(
        bundle, study_output, study_logger
    )
    parameter_audit = audit_parameter_matching(
        selected_variants,
        variants,
        protocol,
        bundle,
        physics_mean,
        physics_std,
    )
    provenance = {
        "protocol_path": protocol_path.relative_to(PROJECT_DIR).as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "dataset_path": data_path.relative_to(PROJECT_DIR).as_posix(),
        "dataset_sha256": sha256_file(data_path),
        "split_sha256": {
            name: sha256_file(split_dir / name)
            for name in ("idx_train.npy", "idx_val.npy", "idx_test.npy")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    save_json(
        study_output / "study_manifest.json",
        {
            "protocol": protocol,
            "selection": {
                "variants": selected_variants,
                "seeds": selected_seeds,
                "epochs": epochs,
                "batch_size": batch_size,
                "device": str(device),
                "formal_protocol": formal_protocol,
            },
            "parameter_audit": parameter_audit,
            "provenance": provenance,
        },
    )
    if args.dry_run:
        print(json.dumps(parameter_audit, indent=2))
        return

    run_count = 0
    # Seed-major ordering yields a complete paired five-variant block before
    # moving to the next replicate, so interrupted long CPU studies remain
    # interpretable and can be resumed by the completed-run markers.
    for seed in selected_seeds:
        for variant_name in selected_variants:
            if args.max_runs is not None and run_count >= args.max_runs:
                break
            status = run_one(
                variant_name=variant_name,
                seed=seed,
                variant_specs=variants,
                protocol=protocol,
                bundle=bundle,
                physics_mean=physics_mean,
                physics_std=physics_std,
                study_output=study_output,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                save_run_plots=args.save_run_plots,
                formal_protocol=formal_protocol,
                provenance=provenance,
            )
            print(f"{variant_name}/seed_{seed}: {status}", flush=True)
            run_count += 1
            summarize_study(
                study_output,
                protocol,
                selected_variants,
                selected_seeds,
                parameter_audit,
            )
        if args.max_runs is not None and run_count >= args.max_runs:
            break

    status = summarize_study(
        study_output,
        protocol,
        selected_variants,
        selected_seeds,
        parameter_audit,
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
