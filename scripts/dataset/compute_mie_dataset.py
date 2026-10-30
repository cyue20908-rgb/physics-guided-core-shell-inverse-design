"""
使用 PyMieDiff 计算四层颗粒的 Mie 效率。

特点：
1. 显式使用 backend='pena'、precision='double'。
2. 不把完整 eps_layers 提前复制到 GPU。
3. 按估计计算复杂度排序分批。
4. 自动检查有限性、非负性、能量关系和 x=m*pi。
5. 不对异常结果进行裁剪；发现异常直接停止。
"""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pymiediff as pmd
import torch

from core_shell_optical_design.project_paths import REPRODUCED_DATASET_BUILD_DIR


STEP1_DIR = REPRODUCED_DATASET_BUILD_DIR / "design_space_sampling"
STEP2_DIR = REPRODUCED_DATASET_BUILD_DIR / "optical_input_preparation"
OUTPUT_DIR = REPRODUCED_DATASET_BUILD_DIR / "mie_dataset_computation"

BACKEND = "pena"
PRECISION = "double"
NEGATIVE_TOL = 1e-7
SINGULAR_TOL = 1e-10

CPU_BATCH_SIZE = 512
GPU_BATCH_SIZE = 1500
OVERWRITE = False


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()) and not OVERWRITE:
        raise FileExistsError(
            f"{path} 已存在且非空，请确认后删除旧目录。"
        )
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "wb") as f:
        np.save(f, array)
    os.replace(temp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def load_and_verify_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for name, expected_hash in manifest["files"].items():
        file_path = directory / name

        if not file_path.exists() or file_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"文件不存在或为空：{file_path}"
            )

        if sha256_file(file_path) != expected_hash:
            raise RuntimeError(f"文件哈希不匹配：{file_path}")

    return manifest


def check_singular_points(
    radii: np.ndarray,
    wavelengths: np.ndarray,
) -> None:
    r_outer = radii[:, -1]

    for wavelength_index, wavelength in enumerate(wavelengths):
        x = 2.0 * np.pi * r_outer / wavelength
        bad = np.where(np.abs(np.sin(x)) <= SINGULAR_TOL)[0]

        if len(bad) > 0:
            raise RuntimeError(
                f"检测到 x=m*pi：波长索引 {wavelength_index}，"
                f"波长 {wavelength:.9f} nm，"
                f"颗粒示例 {bad[:10].tolist()}"
            )


def validate_batch(
    particle_ids: np.ndarray,
    q_ext: np.ndarray,
    q_sca: np.ndarray,
    q_abs: np.ndarray,
) -> dict:
    for name, values in [
        ("Q_ext", q_ext),
        ("Q_sca", q_sca),
        ("Q_abs", q_abs),
    ]:
        if not np.isfinite(values).all():
            index = np.argwhere(~np.isfinite(values))[0]
            particle = int(particle_ids[index[0]])
            raise RuntimeError(
                f"{name} 出现 NaN/Inf，颗粒 {particle}，"
                f"波长索引 {int(index[1])}"
            )

    if q_sca.min() < -NEGATIVE_TOL:
        index = np.unravel_index(np.argmin(q_sca), q_sca.shape)
        raise RuntimeError(
            f"Q_sca 出现负值：{q_sca[index]}，"
            f"颗粒 {int(particle_ids[index[0]])}，"
            f"波长索引 {index[1]}"
        )

    if q_ext.min() < -NEGATIVE_TOL:
        index = np.unravel_index(np.argmin(q_ext), q_ext.shape)
        raise RuntimeError(
            f"Q_ext 出现负值：{q_ext[index]}，"
            f"颗粒 {int(particle_ids[index[0]])}，"
            f"波长索引 {index[1]}"
        )

    if q_abs.min() < -NEGATIVE_TOL:
        index = np.unravel_index(np.argmin(q_abs), q_abs.shape)
        raise RuntimeError(
            f"Q_abs 出现负值：{q_abs[index]}，"
            f"颗粒 {int(particle_ids[index[0]])}，"
            f"波长索引 {index[1]}"
        )

    residual = q_ext - q_sca - q_abs

    return {
        "min_q_ext": float(q_ext.min()),
        "min_q_sca": float(q_sca.min()),
        "min_q_abs": float(q_abs.min()),
        "max_q_ext": float(q_ext.max()),
        "max_q_sca": float(q_sca.max()),
        "max_q_abs": float(q_abs.max()),
        "max_energy_residual": float(
            np.max(np.abs(residual))
        ),
    }


def main() -> None:
    prepare_output_dir(OUTPUT_DIR)

    step1_manifest = load_and_verify_manifest(STEP1_DIR)
    step2_manifest = load_and_verify_manifest(STEP2_DIR)

    if (
        step2_manifest["step1_dataset_id"]
        != step1_manifest["dataset_id"]
    ):
        raise RuntimeError("第一步和第二步数据不属于同一批次。")

    radii = np.load(STEP1_DIR / "radii.npy", mmap_mode="r")
    material_indices = np.load(
        STEP1_DIR / "material_indices.npy", mmap_mode="r"
    )
    wavelengths = np.load(
        STEP2_DIR / "wavelengths_calc.npy"
    )
    eps_layers = np.load(
        STEP2_DIR / "eps_layers.npy", mmap_mode="r"
    )

    n_particles = radii.shape[0]
    n_wavelengths = len(wavelengths)

    expected_eps_shape = (
        n_particles,
        radii.shape[1],
        n_wavelengths,
    )
    if eps_layers.shape != expected_eps_shape:
        raise ValueError(
            f"eps_layers 形状错误：{eps_layers.shape}"
        )

    check_singular_points(radii, wavelengths)

    optics = np.load(
        STEP2_DIR / "material_optics.npz",
        allow_pickle=False,
    )
    n_complex = optics["n"] + 1j * optics["k"]
    material_complexity = np.max(
        np.abs(n_complex), axis=1
    )

    particle_complexity = (
        radii[:, -1]
        * np.max(material_complexity[material_indices], axis=1)
    )
    compute_order = np.argsort(
        particle_complexity, kind="stable"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    batch_size = (
        GPU_BATCH_SIZE
        if device.type == "cuda"
        else CPU_BATCH_SIZE
    )

    print(f"PyMieDiff 版本：{pmd.__version__}")
    print(f"设备：{device}")
    print(f"批次大小：{batch_size}")
    print(f"后端：{BACKEND}，精度：{PRECISION}")

    k0 = 2.0 * np.pi / wavelengths
    k0_tensor = torch.as_tensor(
        k0,
        dtype=torch.float64,
        device=device,
    )

    q_ext_all = np.empty(
        (n_particles, n_wavelengths), dtype=np.float64
    )
    q_sca_all = np.empty_like(q_ext_all)
    q_abs_all = np.empty_like(q_ext_all)

    batch_records = []
    start_time = time.time()

    with torch.no_grad():
        for batch_number, start in enumerate(
            range(0, n_particles, batch_size), start=1
        ):
            stop = min(start + batch_size, n_particles)
            particle_ids = compute_order[start:stop]

            radii_batch = np.asarray(
                radii[particle_ids], dtype=np.float64
            )
            eps_batch = np.asarray(
                eps_layers[particle_ids],
                dtype=np.complex128,
            )

            radii_tensor = torch.as_tensor(
                radii_batch,
                dtype=torch.float64,
                device=device,
            )
            eps_tensor = torch.as_tensor(
                eps_batch,
                dtype=torch.complex128,
                device=device,
            )

            result = pmd.multishell.cross_sections(
                k0=k0_tensor,
                r_layers=radii_tensor,
                eps_layers=eps_tensor,
                eps_env=torch.tensor(
                    1.0 + 0.0j,
                    dtype=torch.complex128,
                    device=device,
                ),
                backend=BACKEND,
                precision=PRECISION,
                n_max=None,
            )

            batch_count = len(particle_ids)

            q_ext = (
                result["q_ext"]
                .detach()
                .cpu()
                .numpy()
                .reshape(batch_count, n_wavelengths)
            )
            q_sca = (
                result["q_sca"]
                .detach()
                .cpu()
                .numpy()
                .reshape(batch_count, n_wavelengths)
            )
            q_abs = (
                result["q_abs"]
                .detach()
                .cpu()
                .numpy()
                .reshape(batch_count, n_wavelengths)
            )

            quality = validate_batch(
                particle_ids, q_ext, q_sca, q_abs
            )

            q_ext_all[particle_ids] = q_ext
            q_sca_all[particle_ids] = q_sca
            q_abs_all[particle_ids] = q_abs

            record = {
                "batch": batch_number,
                "count": batch_count,
                "n_max": int(result["n_max"].item()),
                "complexity_min": float(
                    particle_complexity[particle_ids].min()
                ),
                "complexity_max": float(
                    particle_complexity[particle_ids].max()
                ),
                **quality,
            }
            batch_records.append(record)

            elapsed = time.time() - start_time
            print(
                f"批次 {batch_number}: {stop}/{n_particles}, "
                f"n_max={record['n_max']}, "
                f"耗时={elapsed:.1f}s"
            )

    final_quality = validate_batch(
        np.arange(n_particles),
        q_ext_all,
        q_sca_all,
        q_abs_all,
    )

    atomic_save_npy(OUTPUT_DIR / "Q_ext.npy", q_ext_all)
    atomic_save_npy(OUTPUT_DIR / "Q_sca.npy", q_sca_all)
    atomic_save_npy(OUTPUT_DIR / "Q_abs.npy", q_abs_all)
    atomic_save_npy(
        OUTPUT_DIR / "compute_order.npy", compute_order
    )
    atomic_save_npy(
        OUTPUT_DIR / "batch_nmax.npy",
        np.asarray(
            [record["n_max"] for record in batch_records],
            dtype=np.int32,
        ),
    )

    quality_report = {
        "pymiediff_version": str(pmd.__version__),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "backend": BACKEND,
        "precision": PRECISION,
        "batch_size": batch_size,
        "negative_tolerance": NEGATIVE_TOL,
        "singular_tolerance": SINGULAR_TOL,
        "total_seconds": time.time() - start_time,
        "final_quality": final_quality,
        "batches": batch_records,
    }
    atomic_write_json(
        OUTPUT_DIR / "quality_report.json",
        quality_report,
    )

    files = [
        "Q_ext.npy",
        "Q_sca.npy",
        "Q_abs.npy",
        "compute_order.npy",
        "batch_nmax.npy",
        "quality_report.json",
    ]
    hashes = {
        name: sha256_file(OUTPUT_DIR / name)
        for name in files
    }

    manifest = {
        "stage": 3,
        "step1_dataset_id": step1_manifest["dataset_id"],
        "step2_stage_id": step2_manifest["stage_id"],
        "files": hashes,
    }
    atomic_write_json(OUTPUT_DIR / "manifest.json", manifest)

    print("Mie 计算完成。")
    print(json.dumps(final_quality, indent=2))


if __name__ == "__main__":
    main()
