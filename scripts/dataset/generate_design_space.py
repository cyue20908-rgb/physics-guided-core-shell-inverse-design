"""
生成四层球形核壳颗粒的整数结构参数。

采样原则：
1. 核半径：120–220 nm，整数，并尽可能等量覆盖。
2. 三层壳厚：20–80 nm，整数。
3. 最外层半径不超过300 nm。
4. 每个核心半径下，从全部合法壳厚组合中不重复抽样。
5. Air只允许位于核心，相邻层材料不同。
6. 400种合法材料序列各出现75次。
7. 波长使用400–1000 nm，并在几何生成阶段过滤 x=m*pi 危险外半径。
"""

import hashlib
import json
import os
from itertools import product
from pathlib import Path

import numpy as np

from core_shell_optical_design.project_paths import REPRODUCED_DATASET_BUILD_DIR


# ==================== 基本配置 ====================

SEED = 42
N_PARTICLES = 30000
N_LAYERS = 4

R_CORE_MIN = 120
R_CORE_MAX = 220

SHELL_THICK_MIN = 20
SHELL_THICK_MAX = 80

R_OUTER_MAX = 300

MATERIALS = [
    "Al2O3",
    "BaSO4",
    "SiO2",
    "TiO2",
    "PDMS",
    "Air",
]
AIR_INDEX = MATERIALS.index("Air")

# 101 points from 400 to 1000 nm with 6 nm spacing.
# Structures whose outer size parameter approaches x = m*pi
# are removed by is_safe_outer_radius().
WL_START = 400.0
WL_END = 1000.0
N_WL = 101

WAVELENGTHS = np.linspace(
    WL_START,
    WL_END,
    N_WL,
    dtype=np.float64,
)

# 检查PyMieDiff外部尺寸参数是否接近m*pi。
SINGULAR_TOL = 1e-10

OUTPUT_DIR = REPRODUCED_DATASET_BUILD_DIR / "design_space_sampling"
OVERWRITE = False


# ==================== 文件工具 ====================

def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()) and not OVERWRITE:
        raise FileExistsError(
            f"{path} 已存在且非空。\n"
            "请先删除旧的输出目录，避免新旧数据混合。"
        )

    path.mkdir(parents=True, exist_ok=True)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "wb") as f:
        np.save(f, array)

    os.replace(temp_path, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temp_path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# ==================== 数值安全检查 ====================

def is_safe_outer_radius(
    outer_radii: np.ndarray,
) -> np.ndarray:
    """
    检查所有波长下：

        abs(sin(x)) > SINGULAR_TOL

    其中：

        x = 2*pi*r_outer/lambda
    """
    outer_radii = np.asarray(
        outer_radii,
        dtype=np.float64,
    )

    safe = np.ones(
        len(outer_radii),
        dtype=bool,
    )

    for wavelength in WAVELENGTHS:
        x = (
            2.0
            * np.pi
            * outer_radii
            / wavelength
        )

        safe &= (
            np.abs(np.sin(x))
            > SINGULAR_TOL
        )

    return safe


# ==================== 几何结构生成 ====================

def build_all_shell_combinations() -> np.ndarray:
    """
    枚举全部整数壳层厚度组合。

    总组合数：

        61^3 = 226981
    """
    thickness_values = np.arange(
        SHELL_THICK_MIN,
        SHELL_THICK_MAX + 1,
        dtype=np.int16,
    )

    t1, t2, t3 = np.meshgrid(
        thickness_values,
        thickness_values,
        thickness_values,
        indexing="ij",
    )

    combinations = np.column_stack(
        [
            t1.ravel(),
            t2.ravel(),
            t3.ravel(),
        ]
    ).astype(np.int16)

    return combinations


def build_core_sample_counts(
    rng: np.random.Generator,
) -> dict:
    """
    将30000个样本尽可能均匀分配给101个整数核心半径。

    每个核心半径会出现297或298次。
    """
    core_values = np.arange(
        R_CORE_MIN,
        R_CORE_MAX + 1,
        dtype=np.int16,
    )

    base_count, remainder = divmod(
        N_PARTICLES,
        len(core_values),
    )

    counts = {
        int(core): base_count
        for core in core_values
    }

    # 剩余样本随机分配，避免总是给最小核心半径。
    extra_cores = rng.choice(
        core_values,
        size=remainder,
        replace=False,
    )

    for core in extra_cores:
        counts[int(core)] += 1

    return counts


def sample_integer_radii(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    对每个整数核心半径，从合法壳厚组合中不重复抽样。

    几何约束：
    1. r_outer <= R_OUTER_MAX；
    2. 外部尺寸参数 x = 2*pi*r_outer/lambda 不接近 m*pi。
    """
    shell_combinations = build_all_shell_combinations()
    shell_sum = shell_combinations.sum(axis=1)

    core_counts = build_core_sample_counts(rng)

    sampled_radii = []
    total_before_safety = 0
    total_after_safety = 0

    for core_radius in range(
        R_CORE_MIN,
        R_CORE_MAX + 1,
    ):
        required_count = core_counts[core_radius]

        # 总半径约束。
        valid = (
            core_radius + shell_sum
            <= R_OUTER_MAX
        )

        valid_combinations = shell_combinations[valid]
        valid_outer = (
            core_radius
            + valid_combinations.sum(axis=1)
        )

        n_before_safety = len(valid_combinations)

        # PyMieDiff numerical safety constraint:
        # remove geometries for which x = 2*pi*r_outer/lambda
        # is close to m*pi at any wavelength.
        safe = is_safe_outer_radius(valid_outer)

        valid_combinations = valid_combinations[safe]
        valid_outer = valid_outer[safe]

        n_after_safety = len(valid_combinations)
        total_before_safety += n_before_safety
        total_after_safety += n_after_safety

        if len(valid_combinations) < required_count:
            raise RuntimeError(
                f"核心半径 {core_radius} nm "
                f"只有 {len(valid_combinations)} 个合法且数值安全的组合，"
                f"但需要 {required_count} 个。"
            )

        selected_indices = rng.choice(
            len(valid_combinations),
            size=required_count,
            replace=False,
        )

        selected_thickness = valid_combinations[
            selected_indices
        ].astype(np.int16)

        radii_for_core = np.empty(
            (required_count, N_LAYERS),
            dtype=np.int16,
        )

        radii_for_core[:, 0] = core_radius
        radii_for_core[:, 1] = (
            core_radius
            + selected_thickness[:, 0]
        )
        radii_for_core[:, 2] = (
            radii_for_core[:, 1]
            + selected_thickness[:, 1]
        )
        radii_for_core[:, 3] = (
            radii_for_core[:, 2]
            + selected_thickness[:, 2]
        )

        sampled_radii.append(radii_for_core)

    print(
        "PyMieDiff size-parameter safety filtering: "
        f"removed {total_before_safety - total_after_safety} "
        f"of {total_before_safety} candidate geometries."
    )

    radii = np.vstack(sampled_radii)

    if len(radii) != N_PARTICLES:
        raise RuntimeError(
            f"生成数量错误：{len(radii)}"
        )

    # 打乱核心半径顺序。
    rng.shuffle(radii)

    return radii

# ==================== 材料序列生成 ====================

def build_all_material_sequences() -> np.ndarray:
    """
    枚举所有合法的四层材料序列。

    约束：
    - 核心允许全部6种材料；
    - 壳层只允许5种非Air材料；
    - 相邻层材料不同。

    合法序列总数应为400。
    """
    non_air_indices = [
        index
        for index in range(len(MATERIALS))
        if index != AIR_INDEX
    ]

    sequences = []

    for core in range(len(MATERIALS)):
        for layer1, layer2, layer3 in product(
            non_air_indices,
            repeat=3,
        ):
            if layer1 == core:
                continue

            if layer2 == layer1:
                continue

            if layer3 == layer2:
                continue

            sequences.append(
                [core, layer1, layer2, layer3]
            )

    sequences = np.asarray(
        sequences,
        dtype=np.int8,
    )

    if sequences.shape != (400, 4):
        raise RuntimeError(
            f"合法材料序列数量错误：{sequences.shape}"
        )

    return sequences


def sample_balanced_materials(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    30000 / 400 = 75。

    因此每种合法材料序列精确出现75次。
    """
    sequences = build_all_material_sequences()

    repetitions, remainder = divmod(
        N_PARTICLES,
        len(sequences),
    )

    material_indices = np.repeat(
        sequences,
        repeats=repetitions,
        axis=0,
    )

    if remainder > 0:
        extra_indices = rng.choice(
            len(sequences),
            size=remainder,
            replace=False,
        )

        material_indices = np.vstack(
            [
                material_indices,
                sequences[extra_indices],
            ]
        )

    rng.shuffle(material_indices)

    return material_indices.astype(np.int8)


# ==================== 完整验证 ====================

def validate_dataset(
    radii: np.ndarray,
    material_indices: np.ndarray,
) -> None:
    if radii.shape != (
        N_PARTICLES,
        N_LAYERS,
    ):
        raise ValueError(
            f"radii形状错误：{radii.shape}"
        )

    if material_indices.shape != (
        N_PARTICLES,
        N_LAYERS,
    ):
        raise ValueError(
            "material_indices形状错误："
            f"{material_indices.shape}"
        )

    # 确保结构参数全部为整数。
    if not np.issubdtype(
        radii.dtype,
        np.integer,
    ):
        raise ValueError(
            f"radii不是整数类型：{radii.dtype}"
        )

    if not np.all(
        np.diff(radii, axis=1) > 0
    ):
        raise ValueError(
            "各层外半径没有严格递增。"
        )

    if radii[:, 0].min() < R_CORE_MIN:
        raise ValueError("核心半径低于下限。")

    if radii[:, 0].max() > R_CORE_MAX:
        raise ValueError("核心半径超过上限。")

    if radii[:, -1].max() > R_OUTER_MAX:
        raise ValueError("总半径超过上限。")

    thickness = np.column_stack(
        [
            radii[:, 0],
            np.diff(radii, axis=1),
        ]
    )

    shell_thickness = thickness[:, 1:]

    if shell_thickness.min() < SHELL_THICK_MIN:
        raise ValueError("存在壳层厚度低于下限。")

    if shell_thickness.max() > SHELL_THICK_MAX:
        raise ValueError("存在壳层厚度超过上限。")

    if np.any(material_indices < 0):
        raise ValueError("材料索引存在负值。")

    if np.any(
        material_indices >= len(MATERIALS)
    ):
        raise ValueError("材料索引越界。")

    if np.any(
        material_indices[:, 1:] == AIR_INDEX
    ):
        raise ValueError("Air出现在壳层。")

    if np.any(
        material_indices[:, :-1]
        == material_indices[:, 1:]
    ):
        raise ValueError("存在相邻层材料相同。")

    if not np.all(
        is_safe_outer_radius(radii[:, -1])
    ):
        raise ValueError(
            "仍存在接近x=m*pi的结构。"
        )

    # 每个几何结构应唯一。
    unique_geometry = np.unique(
        radii,
        axis=0,
    )

    if len(unique_geometry) != N_PARTICLES:
        raise ValueError(
            f"存在重复几何结构："
            f"{N_PARTICLES - len(unique_geometry)} 个"
        )

    unique_sequences, sequence_counts = np.unique(
        material_indices,
        axis=0,
        return_counts=True,
    )

    if len(unique_sequences) != 400:
        raise ValueError(
            f"材料序列覆盖不完整："
            f"{len(unique_sequences)}/400"
        )

    if sequence_counts.max() - sequence_counts.min() > 1:
        raise ValueError(
            "材料序列数量不均衡。"
        )


# ==================== 主程序 ====================

def main() -> None:
    prepare_output_dir(OUTPUT_DIR)

    rng = np.random.default_rng(SEED)

    print("生成整数几何结构……")
    radii = sample_integer_radii(rng)

    print("生成均衡材料序列……")
    material_indices = sample_balanced_materials(rng)

    num_layers = np.full(
        N_PARTICLES,
        N_LAYERS,
        dtype=np.int8,
    )

    print("执行完整质量检查……")
    validate_dataset(
        radii,
        material_indices,
    )

    config = {
        "version": "L4_integer_v2",
        "seed": SEED,
        "n_particles": N_PARTICLES,
        "n_layers": N_LAYERS,
        "materials": MATERIALS,
        "air_index": AIR_INDEX,
        "radius_dtype": "int16",
        "radius_core_min_nm": R_CORE_MIN,
        "radius_core_max_nm": R_CORE_MAX,
        "shell_thickness_min_nm": SHELL_THICK_MIN,
        "shell_thickness_max_nm": SHELL_THICK_MAX,
        "radius_outer_max_nm": R_OUTER_MAX,
        "wavelength_start_nm": WL_START,
        "wavelength_end_nm": WL_END,
        "wavelength_points": N_WL,
        "wavelength_step_nm": float(
            WAVELENGTHS[1] - WAVELENGTHS[0]
        ),
        "geometry_sampling": (
            "Balanced integer core radii; "
            "sampling without replacement from all valid "
            "integer shell-thickness combinations"
        ),
        "material_sampling": (
            "All 400 valid material sequences, "
            "75 samples per sequence"
        ),
        "wavelength_grid_reason": (
            "Standard 400-1000 nm wavelength grid with "
            "geometry-level filtering of x=m*pi singular points"
        ),
        "singularity_tolerance": SINGULAR_TOL,
    }

    atomic_save_npy(
        OUTPUT_DIR / "radii.npy",
        radii,
    )
    atomic_save_npy(
        OUTPUT_DIR / "material_indices.npy",
        material_indices,
    )
    atomic_save_npy(
        OUTPUT_DIR / "num_layers.npy",
        num_layers,
    )
    atomic_save_npy(
        OUTPUT_DIR / "wavelengths.npy",
        WAVELENGTHS,
    )
    atomic_write_json(
        OUTPUT_DIR / "config.json",
        config,
    )

    file_names = [
        "radii.npy",
        "material_indices.npy",
        "num_layers.npy",
        "wavelengths.npy",
        "config.json",
    ]

    hashes = {
        name: sha256_file(OUTPUT_DIR / name)
        for name in file_names
    }

    dataset_id = hashlib.sha256(
        "".join(
            hashes[name]
            for name in sorted(hashes)
        ).encode("utf-8")
    ).hexdigest()

    manifest = {
        "stage": 1,
        "dataset_id": dataset_id,
        "files": hashes,
    }

    atomic_write_json(
        OUTPUT_DIR / "manifest.json",
        manifest,
    )

    shell_thickness = np.diff(
        radii,
        axis=1,
    )

    core_values, core_counts = np.unique(
        radii[:, 0],
        return_counts=True,
    )

    _, material_sequence_counts = np.unique(
        material_indices,
        axis=0,
        return_counts=True,
    )

    print()
    print(f"完成：{N_PARTICLES} 个颗粒")
    print(f"dataset_id: {dataset_id}")
    print(
        f"核心半径：{radii[:, 0].min()}–"
        f"{radii[:, 0].max()} nm"
    )
    print(
        f"总半径：{radii[:, -1].min()}–"
        f"{radii[:, -1].max()} nm"
    )
    print(
        f"壳层1厚度：{shell_thickness[:, 0].min()}–"
        f"{shell_thickness[:, 0].max()} nm"
    )
    print(
        f"壳层2厚度：{shell_thickness[:, 1].min()}–"
        f"{shell_thickness[:, 1].max()} nm"
    )
    print(
        f"壳层3厚度：{shell_thickness[:, 2].min()}–"
        f"{shell_thickness[:, 2].max()} nm"
    )
    print(
        f"每个核心半径样本数："
        f"{core_counts.min()}–{core_counts.max()}"
    )
    print(
        f"每种材料序列样本数："
        f"{material_sequence_counts.min()}–"
        f"{material_sequence_counts.max()}"
    )
    print(
        f"波长：{WAVELENGTHS[0]}–"
        f"{WAVELENGTHS[-1]} nm，"
        f"{len(WAVELENGTHS)} 点"
    )


if __name__ == "__main__":
    main()
