"""
读取材料数据库，构建四层复介电常数。

插值规则：
    先分别插值 n 和 k，再计算 epsilon = (n + i*k)^2。
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from core_shell_optical_design.project_paths import (
    MATERIAL_DATA_DIR,
    REPRODUCED_DATASET_BUILD_DIR,
)


INPUT_DIR = REPRODUCED_DATASET_BUILD_DIR / "design_space_sampling"
OUTPUT_DIR = REPRODUCED_DATASET_BUILD_DIR / "optical_input_preparation"

DEFAULT_MATERIAL_DB = MATERIAL_DATA_DIR / "material_data.npz"

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


def atomic_save_npz(path: Path, **arrays) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def load_and_verify_manifest(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少 {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for name, expected_hash in manifest["files"].items():
        path = directory / name

        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"文件不存在或为空：{path}")

        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"文件哈希不匹配：{path}")

    return manifest


def main() -> None:
    prepare_output_dir(OUTPUT_DIR)
    step1_manifest = load_and_verify_manifest(INPUT_DIR)

    with open(INPUT_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    material_indices = np.load(INPUT_DIR / "material_indices.npy")
    wavelengths = np.load(INPUT_DIR / "wavelengths.npy")

    materials = config["materials"]
    n_particles = config["n_particles"]
    n_layers = config["n_layers"]

    db_env = os.environ.get("MATERIAL_DB_PATH")
    material_db_path = Path(db_env) if db_env else DEFAULT_MATERIAL_DB

    if not material_db_path.exists():
        raise FileNotFoundError(
            f"材料数据库不存在：{material_db_path}"
        )

    db_hash = sha256_file(material_db_path)
    database = np.load(material_db_path, allow_pickle=False)

    if "wavelengths" not in database:
        raise KeyError("材料数据库缺少 wavelengths。")

    wl_reference = np.asarray(
        database["wavelengths"], dtype=np.float64
    )

    if not np.isfinite(wl_reference).all():
        raise ValueError("材料数据库波长中存在 NaN 或 Inf。")

    if not np.all(np.diff(wl_reference) > 0):
        raise ValueError("材料数据库波长必须严格递增。")

    n_material = np.empty(
        (len(materials), len(wavelengths)), dtype=np.float64
    )
    k_material = np.empty_like(n_material)
    eps_material = np.empty(
        n_material.shape, dtype=np.complex128
    )

    for material_id, material_name in enumerate(materials):
        n_key = f"{material_name}_n"
        k_key = f"{material_name}_k"

        if n_key not in database or k_key not in database:
            raise KeyError(
                f"材料数据库缺少 {material_name} 的 n/k 数据。"
            )

        n_all = np.asarray(database[n_key], dtype=np.float64)
        k_all = np.asarray(database[k_key], dtype=np.float64)

        if n_all.shape != wl_reference.shape:
            raise ValueError(f"{n_key} 与 wavelengths 形状不一致。")

        if k_all.shape != wl_reference.shape:
            raise ValueError(f"{k_key} 与 wavelengths 形状不一致。")

        finite = np.isfinite(n_all) & np.isfinite(k_all)
        valid_indices = np.flatnonzero(finite)

        if len(valid_indices) < 2:
            raise ValueError(f"{material_name} 有效数据点不足。")

        if np.any(np.diff(valid_indices) != 1):
            raise ValueError(
                f"{material_name} 的有效光学数据存在内部缺口。"
            )

        wl_valid = wl_reference[finite]
        n_valid = n_all[finite]
        k_valid = k_all[finite]

        if wavelengths.min() < wl_valid.min() or (
            wavelengths.max() > wl_valid.max()
        ):
            raise ValueError(
                f"{material_name} 的有效范围为 "
                f"{wl_valid.min():.3f}–{wl_valid.max():.3f} nm，"
                "不能覆盖计算波长。"
            )

        n_interp = np.interp(
            wavelengths,
            wl_valid,
            n_valid,
        )
        k_interp = np.interp(
            wavelengths,
            wl_valid,
            k_valid,
        )

        if not np.isfinite(n_interp).all():
            raise ValueError(
                f"{material_name} 插值 n 非有限。"
            )

        if not np.isfinite(k_interp).all():
            raise ValueError(
                f"{material_name} 插值 k 非有限。"
            )

        if np.any(k_interp < -1e-12):
            negative_index = int(np.argmin(k_interp))

            raise ValueError(
                f"{material_name} 在实际计算波段出现负消光系数："
                f"wavelength={wavelengths[negative_index]:.6f} nm, "
                f"k={k_interp[negative_index]:.6e}"
            )

        n_complex = n_interp + 1j * k_interp
        eps_interp = n_complex ** 2

        n_material[material_id] = n_interp
        k_material[material_id] = k_interp
        eps_material[material_id] = eps_interp

        print(
            f"{material_name}: "
            f"n={n_interp.min():.5f}–{n_interp.max():.5f}, "
            f"k={k_interp.min():.6g}–{k_interp.max():.6g}"
        )

    if np.any(material_indices < 0) or np.any(
        material_indices >= len(materials)
    ):
        raise ValueError("材料索引越界。")

    # 利用索引一次性组装，不再对每个颗粒重复插值。
    eps_layers = eps_material[material_indices]

    expected_shape = (
        n_particles,
        n_layers,
        len(wavelengths),
    )
    if eps_layers.shape != expected_shape:
        raise ValueError(
            f"eps_layers 形状错误：{eps_layers.shape}"
        )

    if not np.isfinite(eps_layers.real).all() or not np.isfinite(
        eps_layers.imag
    ).all():
        raise ValueError("eps_layers 中存在 NaN 或 Inf。")

    if np.any(eps_layers.imag < -1e-12):
        raise ValueError("被动材料介电常数出现明显负虚部。")

    atomic_save_npy(OUTPUT_DIR / "eps_layers.npy", eps_layers)
    atomic_save_npy(
        OUTPUT_DIR / "wavelengths_calc.npy", wavelengths
    )
    atomic_save_npz(
        OUTPUT_DIR / "material_optics.npz",
        material_names=np.asarray(materials),
        wavelengths=wavelengths,
        n=n_material,
        k=k_material,
        epsilon=eps_material,
    )

    config_v2 = dict(config)
    config_v2.update(
        {
            "step1_dataset_id": step1_manifest["dataset_id"],
            "material_database_path": str(
                material_db_path.resolve()
            ),
            "material_database_sha256": db_hash,
            "interpolation_method": (
                "linear interpolation of n and k, "
                "followed by epsilon=(n+i*k)^2"
            ),
        }
    )
    atomic_write_json(OUTPUT_DIR / "config.json", config_v2)

    output_files = [
        "eps_layers.npy",
        "wavelengths_calc.npy",
        "material_optics.npz",
        "config.json",
    ]
    hashes = {
        name: sha256_file(OUTPUT_DIR / name)
        for name in output_files
    }

    stage_id = hashlib.sha256(
        (
            step1_manifest["dataset_id"]
            + "".join(hashes[name] for name in sorted(hashes))
        ).encode("utf-8")
    ).hexdigest()

    manifest = {
        "stage": 2,
        "stage_id": stage_id,
        "step1_dataset_id": step1_manifest["dataset_id"],
        "files": hashes,
    }
    atomic_write_json(OUTPUT_DIR / "manifest.json", manifest)

    print(f"完成，eps_layers 形状：{eps_layers.shape}")
    print(f"stage_id: {stage_id}")


if __name__ == "__main__":
    main()
