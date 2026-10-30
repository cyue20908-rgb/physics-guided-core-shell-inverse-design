"""
导出统一数据集。

兼容原训练代码：
    config 仍以对象字典保存。

同时增加：
    config_json
    material_n/material_k/material_epsilon
    sample_id
    quality_report_json
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from core_shell_optical_design.project_paths import REPRODUCED_DATASET_BUILD_DIR


STEP1_DIR = REPRODUCED_DATASET_BUILD_DIR / "design_space_sampling"
STEP2_DIR = REPRODUCED_DATASET_BUILD_DIR / "optical_input_preparation"
STEP3_DIR = REPRODUCED_DATASET_BUILD_DIR / "mie_dataset_computation"
OUTPUT_DIR = REPRODUCED_DATASET_BUILD_DIR / "learning_dataset_export"

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


def verify_manifest(directory: Path) -> dict:
    with open(
        directory / "manifest.json", "r", encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    for name, expected_hash in manifest["files"].items():
        path = directory / name

        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"文件不存在或为空：{path}")

        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"哈希不匹配：{path}")

    return manifest


def atomic_save_npz(path: Path, **arrays) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def atomic_write_text(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(temp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def main() -> None:
    prepare_output_dir(OUTPUT_DIR)

    manifest1 = verify_manifest(STEP1_DIR)
    manifest2 = verify_manifest(STEP2_DIR)
    manifest3 = verify_manifest(STEP3_DIR)

    if manifest2["step1_dataset_id"] != manifest1["dataset_id"]:
        raise RuntimeError("第一步和第二步数据批次不一致。")

    if manifest3["step1_dataset_id"] != manifest1["dataset_id"]:
        raise RuntimeError("第一步和第三步数据批次不一致。")

    if manifest3["step2_stage_id"] != manifest2["stage_id"]:
        raise RuntimeError("第二步和第三步数据批次不一致。")

    radii = np.load(STEP1_DIR / "radii.npy")
    material_indices = np.load(
        STEP1_DIR / "material_indices.npy"
    )
    num_layers = np.load(STEP1_DIR / "num_layers.npy")
    wavelengths = np.load(
        STEP2_DIR / "wavelengths_calc.npy"
    )

    q_ext = np.load(STEP3_DIR / "Q_ext.npy")
    q_sca = np.load(STEP3_DIR / "Q_sca.npy")
    q_abs = np.load(STEP3_DIR / "Q_abs.npy")

    with open(
        STEP2_DIR / "config.json", "r", encoding="utf-8"
    ) as f:
        config = json.load(f)

    with open(
        STEP3_DIR / "quality_report.json",
        "r",
        encoding="utf-8",
    ) as f:
        quality_report = json.load(f)

    optics = np.load(
        STEP2_DIR / "material_optics.npz",
        allow_pickle=False,
    )

    n_particles = len(radii)
    n_wavelengths = len(wavelengths)
    expected_shape = (n_particles, n_wavelengths)

    for name, values in [
        ("Q_ext", q_ext),
        ("Q_sca", q_sca),
        ("Q_abs", q_abs),
    ]:
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} 形状错误：{values.shape}"
            )

        if not np.isfinite(values).all():
            raise ValueError(f"{name} 包含 NaN 或 Inf。")

    if q_sca.min() < -1e-7:
        raise ValueError("Q_sca 存在负值，禁止导出。")

    if q_abs.min() < -1e-7:
        raise ValueError("Q_abs 存在负值，禁止导出。")

    if q_ext.min() < -1e-7:
        raise ValueError("Q_ext 存在负值，禁止导出。")

    residual = q_ext - q_sca - q_abs
    max_residual = float(np.max(np.abs(residual)))

    # `radii.npy` is stored as int16. Cast before squaring; otherwise
    # radii above ~181 nm overflow int16 and corrupt all cross sections.
    r_outer = radii[:, -1].astype(np.float64)
    c_geo = np.pi * r_outer**2
    c_ext = q_ext.astype(np.float64) * c_geo[:, None]
    c_sca = q_sca.astype(np.float64) * c_geo[:, None]
    c_abs = q_abs.astype(np.float64) * c_geo[:, None]

    config["dataset_id"] = manifest1["dataset_id"]
    config["step2_stage_id"] = manifest2["stage_id"]
    config["pymiediff"] = {
        "version": quality_report["pymiediff_version"],
        "backend": quality_report["backend"],
        "precision": quality_report["precision"],
    }
    config["max_energy_residual_float64"] = max_residual

    output_path = OUTPUT_DIR / "multilayer_mie_dataset.npz"

    atomic_save_npz(
        output_path,
        sample_id=np.arange(n_particles, dtype=np.int32),
        radii=radii.astype(np.float32),
        material_indices=material_indices.astype(np.int8),
        num_layers=num_layers.astype(np.int8),
        wavelengths=wavelengths.astype(np.float32),
        Q_ext=q_ext.astype(np.float32),
        Q_sca=q_sca.astype(np.float32),
        Q_abs=q_abs.astype(np.float32),
        C_ext=c_ext.astype(np.float32),
        C_sca=c_sca.astype(np.float32),
        C_abs=c_abs.astype(np.float32),
        material_names=optics["material_names"],
        material_n=optics["n"].astype(np.float32),
        material_k=optics["k"].astype(np.float32),
        material_epsilon=optics["epsilon"].astype(
            np.complex64
        ),
        # 与原训练代码兼容。
        config=np.array(config, dtype=object),
        # 非 Pickle 形式的安全配置副本。
        config_json=np.array(
            json.dumps(config, ensure_ascii=False)
        ),
        quality_report_json=np.array(
            json.dumps(quality_report, ensure_ascii=False)
        ),
    )

    readme = """multilayer_mie_dataset.npz

主要字段：
- sample_id: 原始样本编号
- radii: (N, 4)，各层累计外半径，nm
- material_indices: (N, 4)，材料索引
- num_layers: (N,)，恒为4
- wavelengths: (101,)，波长，nm
- Q_ext/Q_sca/Q_abs: Mie效率
- C_ext/C_sca/C_abs: 截面积，nm^2
- material_names: 材料名称
- material_n/material_k: 每种材料的插值光学常数
- material_epsilon: 每种材料的复介电常数
- config: 与原训练代码兼容的字典
- config_json: JSON格式配置
- quality_report_json: Mie计算质量报告
"""
    atomic_write_text(OUTPUT_DIR / "dataset_description.txt", readme)

    final_hash = sha256_file(output_path)
    export_manifest = {
        "stage": 4,
        "dataset_id": manifest1["dataset_id"],
        "file": output_path.name,
        "sha256": final_hash,
        "size_bytes": output_path.stat().st_size,
    }
    atomic_write_json(
        OUTPUT_DIR / "manifest.json",
        export_manifest,
    )

    print(f"导出完成：{output_path}")
    print(
        f"大小：{output_path.stat().st_size / 1024**2:.2f} MB"
    )
    print(f"SHA256：{final_hash}")
    print(f"最大能量残差：{max_residual:.3e}")


if __name__ == "__main__":
    main()
