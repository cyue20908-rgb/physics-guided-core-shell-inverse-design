"""
第五步：最终数据质量检查、结果可视化和在线Mie验证数据导出。

输出目录：
    learning_dataset_validation/

输出内容：
    validation_report.json
    geometry_distribution.png / geometry_distribution.svg
    material_distribution.png / material_distribution.svg
    quality_by_wavelength.png / quality_by_wavelength.svg
    sample_spectra.png / sample_spectra.svg

    online_validation/
        validation_summary.csv
        sample_XXXXX_structure.csv
        sample_XXXXX_spectrum.csv
        sample_XXXXX_optical_constants.csv
"""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from core_shell_optical_design.project_paths import (
    DATASET_PATH,
    REPRODUCED_DATASET_BUILD_DIR,
)


# ==================== 路径和参数 ====================

DATA_PATH = DATASET_PATH

OUTPUT_DIR = REPRODUCED_DATASET_BUILD_DIR / "dataset_validation"

ONLINE_VALIDATION_DIR = (
    OUTPUT_DIR / "online_validation"
)

N_VALIDATION_SAMPLES = 8
RANDOM_SEED = 2026

# 小于该量级的负值视为浮点舍入误差。
NEGATIVE_TOL = 1e-6

# float32导出后允许的能量残差。
ENERGY_RESIDUAL_TOL = 1e-5

# Cross sections are exported as float32, so use combined relative and
# absolute tolerances when comparing C = Q * pi * r_outer^2.
CROSS_SECTION_RTOL = 2e-6
CROSS_SECTION_ATOL = 0.2

# 外部尺寸参数奇点检查容差。
SINGULAR_TOL = 1e-8

ENVIRONMENT_REFRACTIVE_INDEX = 1.0


# ==================== 通用函数 ====================

def json_default(value):
    """将NumPy标量转换为Python原生类型。"""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def atomic_write_json(
    path: Path,
    obj: dict,
) -> None:
    """避免程序中断时留下不完整JSON文件。"""
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


def load_config(data) -> dict:
    if "config" in data:
        return data["config"].item()

    if "config_json" in data:
        return json.loads(
            str(data["config_json"].item())
        )

    raise KeyError(
        "数据集中不存在config或config_json。"
    )


def get_layer_thickness(
    cumulative_radii: np.ndarray,
) -> np.ndarray:
    """累计外半径转换为核心半径和三层壳厚。"""
    return np.concatenate(
        [
            cumulative_radii[:1],
            np.diff(cumulative_radii),
        ]
    )


def select_validation_samples(
    radii: np.ndarray,
    q_sca: np.ndarray,
    q_abs: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    选择具有代表性的验证样本：

    1. 最小外半径；
    2. 最大外半径；
    3. 最小核心；
    4. 最大核心；
    5. 最大Q_sca；
    6. 最大Q_abs；
    7. 其余随机补齐。
    """
    candidates = [
        int(np.argmin(radii[:, -1])),
        int(np.argmax(radii[:, -1])),
        int(np.argmin(radii[:, 0])),
        int(np.argmax(radii[:, 0])),
        int(np.argmax(np.max(q_sca, axis=1))),
        int(np.argmax(np.max(q_abs, axis=1))),
    ]

    selected = []

    for sample_id in candidates:
        if sample_id not in selected:
            selected.append(sample_id)

    rng = np.random.default_rng(RANDOM_SEED)

    while len(selected) < min(
        n_samples,
        len(radii),
    ):
        sample_id = int(
            rng.integers(0, len(radii))
        )

        if sample_id not in selected:
            selected.append(sample_id)

    return np.asarray(
        selected,
        dtype=np.int32,
    )


# ==================== 在线验证数据导出 ====================

def export_online_validation_data(
    sample_ids: np.ndarray,
    radii: np.ndarray,
    material_indices: np.ndarray,
    materials: list,
    wavelengths: np.ndarray,
    q_ext: np.ndarray,
    q_sca: np.ndarray,
    q_abs: np.ndarray,
    c_ext: np.ndarray,
    c_sca: np.ndarray,
    c_abs: np.ndarray,
    material_n: np.ndarray | None,
    material_k: np.ndarray | None,
) -> None:
    ONLINE_VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        ONLINE_VALIDATION_DIR
        / "validation_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "sample_id",
                "environment_n",
                "r1_outer_nm",
                "r2_outer_nm",
                "r3_outer_nm",
                "r4_outer_nm",
                "core_radius_nm",
                "shell1_thickness_nm",
                "shell2_thickness_nm",
                "shell3_thickness_nm",
                "material_layer1",
                "material_layer2",
                "material_layer3",
                "material_layer4",
                "max_Q_ext",
                "max_Q_sca",
                "max_Q_abs",
            ]
        )

        for sample_id in sample_ids:
            sample_id = int(sample_id)

            sample_radii = radii[sample_id]
            thickness = get_layer_thickness(
                sample_radii
            )
            sample_materials = [
                materials[index]
                for index in material_indices[
                    sample_id
                ]
            ]

            writer.writerow(
                [
                    sample_id,
                    ENVIRONMENT_REFRACTIVE_INDEX,
                    *sample_radii.tolist(),
                    *thickness.tolist(),
                    *sample_materials,
                    float(np.max(q_ext[sample_id])),
                    float(np.max(q_sca[sample_id])),
                    float(np.max(q_abs[sample_id])),
                ]
            )

            export_single_validation_sample(
                sample_id=sample_id,
                sample_radii=sample_radii,
                sample_material_indices=(
                    material_indices[sample_id]
                ),
                materials=materials,
                wavelengths=wavelengths,
                q_ext=q_ext[sample_id],
                q_sca=q_sca[sample_id],
                q_abs=q_abs[sample_id],
                c_ext=c_ext[sample_id],
                c_sca=c_sca[sample_id],
                c_abs=c_abs[sample_id],
                material_n=material_n,
                material_k=material_k,
            )

    np.save(
        ONLINE_VALIDATION_DIR
        / "validation_sample_ids.npy",
        sample_ids,
    )


def export_single_validation_sample(
    sample_id: int,
    sample_radii: np.ndarray,
    sample_material_indices: np.ndarray,
    materials: list,
    wavelengths: np.ndarray,
    q_ext: np.ndarray,
    q_sca: np.ndarray,
    q_abs: np.ndarray,
    c_ext: np.ndarray,
    c_sca: np.ndarray,
    c_abs: np.ndarray,
    material_n: np.ndarray | None,
    material_k: np.ndarray | None,
) -> None:
    prefix = f"sample_{sample_id:05d}"

    # ---------- 结构文件 ----------
    structure_path = (
        ONLINE_VALIDATION_DIR
        / f"{prefix}_structure.csv"
    )

    thickness = get_layer_thickness(
        sample_radii
    )

    with open(
        structure_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "sample_id",
                "layer",
                "material_index",
                "material_name",
                "outer_radius_nm",
                "layer_thickness_nm",
                "environment_n",
            ]
        )

        for layer in range(4):
            material_index = int(
                sample_material_indices[layer]
            )

            writer.writerow(
                [
                    sample_id,
                    layer + 1,
                    material_index,
                    materials[material_index],
                    float(sample_radii[layer]),
                    float(thickness[layer]),
                    ENVIRONMENT_REFRACTIVE_INDEX,
                ]
            )

    # ---------- PyMieDiff光谱 ----------
    spectrum_path = (
        ONLINE_VALIDATION_DIR
        / f"{prefix}_spectrum.csv"
    )

    x_outer = (
        2.0
        * np.pi
        * sample_radii[-1]
        / wavelengths
    )

    spectrum_data = np.column_stack(
        [
            wavelengths,
            x_outer,
            q_ext,
            q_sca,
            q_abs,
            c_ext,
            c_sca,
            c_abs,
            q_ext - q_sca - q_abs,
        ]
    )

    np.savetxt(
        spectrum_path,
        spectrum_data,
        delimiter=",",
        header=(
            "wavelength_nm,"
            "outer_size_parameter,"
            "Q_ext,"
            "Q_sca,"
            "Q_abs,"
            "C_ext_nm2,"
            "C_sca_nm2,"
            "C_abs_nm2,"
            "energy_residual"
        ),
        comments="",
    )

    # ---------- 每层n和k ----------
    if material_n is None or material_k is None:
        return

    optical_columns = [wavelengths]
    optical_headers = ["wavelength_nm"]

    for layer in range(4):
        material_index = int(
            sample_material_indices[layer]
        )
        material_name = materials[
            material_index
        ]

        optical_columns.append(
            material_n[material_index]
        )
        optical_columns.append(
            material_k[material_index]
        )

        optical_headers.append(
            f"layer{layer + 1}_{material_name}_n"
        )
        optical_headers.append(
            f"layer{layer + 1}_{material_name}_k"
        )

    optical_data = np.column_stack(
        optical_columns
    )

    optical_path = (
        ONLINE_VALIDATION_DIR
        / f"{prefix}_optical_constants.csv"
    )

    np.savetxt(
        optical_path,
        optical_data,
        delimiter=",",
        header=",".join(optical_headers),
        comments="",
    )


# ==================== 绘图 ====================

# 出版级绘图参数。Windows系统通常已安装Times New Roman。
PLOT_DPI = 600
FONT_FAMILY = "Times New Roman"
TITLE_FONT_SIZE = 15
SUBPLOT_TITLE_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 9
LEGEND_FONT_SIZE = 9
AXIS_LINE_WIDTH = 1.2
CURVE_LINE_WIDTH = 2.0
GRID_ALPHA = 0.22


def configure_plot_style() -> None:
    """
    统一设置全部图片的出版级风格：
    Times New Roman、加粗文字、黑色坐标轴和高分辨率输出。
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": SUBPLOT_TITLE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.framealpha": 0.95,
            "mathtext.fontset": "stix",
            "mathtext.default": "bf",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def set_bold_times_font(
    ax,
    *,
    grid: bool = True,
) -> None:
    """将单个坐标轴内的全部文字、刻度和边框统一为论文格式。"""
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(AXIS_LINE_WIDTH)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        width=1.0,
        length=5.0,
        colors="black",
    )

    for label in ax.get_xticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontweight("bold")
        label.set_fontsize(TICK_FONT_SIZE)

    for label in ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontweight("bold")
        label.set_fontsize(TICK_FONT_SIZE)

    ax.xaxis.label.set_fontname(FONT_FAMILY)
    ax.xaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)

    ax.yaxis.label.set_fontname(FONT_FAMILY)
    ax.yaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)

    ax.title.set_fontname(FONT_FAMILY)
    ax.title.set_fontweight("bold")

    ax.xaxis.get_offset_text().set_fontname(FONT_FAMILY)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.xaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)

    ax.yaxis.get_offset_text().set_fontname(FONT_FAMILY)
    ax.yaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)

    if grid:
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.65,
            alpha=GRID_ALPHA,
            color="gray",
        )
    else:
        ax.grid(False)


def style_legend(legend) -> None:
    """统一图例字体、背景和边框。"""
    if legend is None:
        return

    for text_item in legend.get_texts():
        text_item.set_fontname(FONT_FAMILY)
        text_item.set_fontweight("bold")
        text_item.set_fontsize(LEGEND_FONT_SIZE)

    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_linewidth(0.8)
    frame.set_alpha(0.95)


def style_colorbar(
    colorbar,
    label: str,
) -> None:
    """统一颜色条标签和刻度格式。"""
    colorbar.set_label(
        label,
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        fontname=FONT_FAMILY,
    )
    colorbar.ax.tick_params(
        direction="in",
        width=1.0,
        length=4.0,
        labelsize=TICK_FONT_SIZE,
    )

    for label_item in colorbar.ax.get_yticklabels():
        label_item.set_fontname(FONT_FAMILY)
        label_item.set_fontweight("bold")
        label_item.set_fontsize(TICK_FONT_SIZE)

    colorbar.outline.set_edgecolor("black")
    colorbar.outline.set_linewidth(AXIS_LINE_WIDTH)


def add_panel_label(
    ax,
    label: str,
) -> None:
    """在子图左上角添加(a)、(b)等编号。"""
    ax.text(
        0.02,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        fontname=FONT_FAMILY,
        color="black",
        zorder=10,
    )
def add_panel_label_outside(
    ax,
    label: str,
    *,
    x: float = -0.105,
    y: float = 1.035,
) -> None:
    """
    将子图编号放置在坐标轴左上角外侧。

    参数
    ----------
    ax
        Matplotlib 坐标轴。
    label
        子图编号，例如 "(a)"。
    x, y
        相对于坐标轴的归一化位置。
        x < 0 表示位于左边框外侧；
        y > 1 表示位于上边框外侧。
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        fontname=FONT_FAMILY,
        color="black",
        clip_on=False,
        zorder=20,
    )

def material_display_name(name: str) -> str:
    """将常用材料名称转换为带下标的数学格式。"""
    mapping = {
        "Al2O3": r"Al$_2$O$_3$",
        "BaSO4": r"BaSO$_4$",
        "TiO2": r"TiO$_2$",
        "SiO2": r"SiO$_2$",
    }
    return mapping.get(name, name)


def save_figure(
    fig,
    filename: str,
) -> None:
    """以600 dpi保存PNG，并同时保存同名SVG矢量文件。"""
    png_path = OUTPUT_DIR / filename
    svg_path = png_path.with_suffix(".svg")

    fig.savefig(
        png_path,
        format="png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    fig.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    plt.close(fig)


def plot_geometry_distribution(
    radii: np.ndarray,
) -> None:
    """
    绘制整数几何参数分布及 Pearson 相关系数矩阵。

    图形设置：
        2 × 3 布局；
        (a)–(f) 位于各子图左上角外侧；
        Times New Roman 加粗；
        相关矩阵保留数值和颜色条；
        600 dpi 高分辨率输出。
    """
    # ==============================================================
    # 1. 从累计半径计算核心半径和三个壳层厚度
    # ==============================================================
    thickness = np.column_stack(
        [
            radii[:, 0],
            np.diff(radii, axis=1),
        ]
    )

    axis_labels = [
        "Core radius",
        "Shell 1 thickness",
        "Shell 2 thickness",
        "Shell 3 thickness",
    ]

    histogram_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
    ]

    panel_labels = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
    ]

    # 不使用 constrained_layout。
    # 图外编号需要通过 subplots_adjust 精确预留边距。
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(16.2, 8.9),
        constrained_layout=False,
    )

    # ==============================================================
    # 2. 核心半径和三个壳层厚度的直方图
    # ==============================================================
    for index in range(4):
        ax = axes.flat[index]

        data_values = thickness[:, index]

        bins = np.arange(
            np.min(data_values) - 0.5,
            np.max(data_values) + 1.5,
            1.0,
        )

        ax.hist(
            data_values,
            bins=bins,
            color=histogram_colors[index],
            alpha=0.90,
            edgecolor="black",
            linewidth=0.25,
        )

        ax.set_xlabel(
            f"{axis_labels[index]} (nm)",
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
            fontname=FONT_FAMILY,
        )

        ax.set_ylabel(
            "Count",
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight="bold",
            fontname=FONT_FAMILY,
        )

        set_bold_times_font(
            ax,
            grid=True,
        )

        # 子图编号置于左上角外侧
        add_panel_label_outside(
            ax,
            panel_labels[index],
        )

    # ==============================================================
    # 3. 总外半径分布
    # ==============================================================
    outer_ax = axes.flat[4]
    outer_radius = radii[:, -1]

    outer_bins = np.arange(
        np.min(outer_radius) - 0.5,
        np.max(outer_radius) + 1.5,
        1.0,
    )

    outer_ax.hist(
        outer_radius,
        bins=outer_bins,
        color="tab:purple",
        alpha=0.90,
        edgecolor="black",
        linewidth=0.25,
    )

    outer_ax.set_xlabel(
        "Outer radius (nm)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        fontname=FONT_FAMILY,
    )

    outer_ax.set_ylabel(
        "Count",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        fontname=FONT_FAMILY,
    )

    set_bold_times_font(
        outer_ax,
        grid=True,
    )

    add_panel_label_outside(
        outer_ax,
        panel_labels[4],
    )

    # ==============================================================
    # 4. Pearson 相关系数矩阵
    # ==============================================================
    correlation_data = np.column_stack(
        [
            thickness,
            outer_radius,
        ]
    )

    correlation = np.corrcoef(
        correlation_data,
        rowvar=False,
    )

    corr_ax = axes.flat[5]

    image = corr_ax.imshow(
        correlation,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
        interpolation="nearest",
        aspect="equal",
    )

    correlation_labels = [
        r"$r_{\mathrm{core}}$",
        r"$t_1$",
        r"$t_2$",
        r"$t_3$",
        r"$R_{\mathrm{out}}$",
    ]

    corr_ax.set_xticks(
        np.arange(5)
    )

    corr_ax.set_yticks(
        np.arange(5)
    )

    corr_ax.set_xticklabels(
        correlation_labels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    corr_ax.set_yticklabels(
        correlation_labels
    )

    corr_ax.set_title(
        "Pearson correlation matrix",
        fontsize=SUBPLOT_TITLE_FONT_SIZE,
        fontweight="bold",
        fontname=FONT_FAMILY,
        pad=9,
    )

    # --------------------------------------------------------------
    # 在相关矩阵中标注数值
    # --------------------------------------------------------------
    for row in range(correlation.shape[0]):
        for column in range(correlation.shape[1]):
            value = correlation[row, column]

            # 根据背景颜色自动选择文字颜色
            text_color = (
                "white"
                if abs(value) >= 0.55
                else "black"
            )

            corr_ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                fontname=FONT_FAMILY,
                color=text_color,
                zorder=5,
            )

    set_bold_times_font(
        corr_ax,
        grid=False,
    )

    add_panel_label_outside(
        corr_ax,
        panel_labels[5],
    )

    # ==============================================================
    # 5. 相关系数颜色条
    # ==============================================================
    colorbar = fig.colorbar(
        image,
        ax=corr_ax,
        fraction=0.046,
        pad=0.045,
    )

    style_colorbar(
        colorbar,
        "Correlation coefficient",
    )

    # 固定颜色条刻度，使范围表达更加明确
    colorbar.set_ticks(
        np.linspace(-1.0, 1.0, 9)
    )

    colorbar.set_ticklabels(
        [
            "-1.00",
            "-0.75",
            "-0.50",
            "-0.25",
            "0.00",
            "0.25",
            "0.50",
            "0.75",
            "1.00",
        ]
    )

    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontname(FONT_FAMILY)
        tick_label.set_fontweight("bold")
        tick_label.set_fontsize(TICK_FONT_SIZE)

    # ==============================================================
    # 6. 全图标题
    # ==============================================================
    fig.suptitle(
        "Distributions of sampled integer geometries",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        fontname=FONT_FAMILY,
        x=0.50,
        y=0.975,
    )

    # ==============================================================
    # 7. 精确控制 2 × 3 排版
    #
    # left 和 top 需要留出空间，以容纳位于坐标轴外部的编号。
    # ==============================================================
    fig.subplots_adjust(
        left=0.072,
        right=0.965,
        bottom=0.085,
        top=0.905,
        wspace=0.20,
        hspace=0.28,
    )

    # ==============================================================
    # 8. 高分辨率保存
    # ==============================================================
    geometry_png = OUTPUT_DIR / "geometry_distribution.png"
    geometry_svg = OUTPUT_DIR / "geometry_distribution.svg"

    fig.savefig(
        geometry_png,
        format="png",
        dpi=PLOT_DPI,
        bbox_inches="tight",
        pad_inches=0.09,
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(
        geometry_svg,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.09,
        facecolor="white",
        edgecolor="none",
    )

    print(f"Saved: {geometry_png}")
    print(f"Saved: {geometry_svg}")
    plt.close(fig)


def plot_material_distribution(
    material_indices: np.ndarray,
    materials: list,
) -> None:
    counts = np.zeros(
        (4, len(materials)),
        dtype=np.int64,
    )

    for layer in range(4):
        counts[layer] = np.bincount(
            material_indices[:, layer],
            minlength=len(materials),
        )

    fig, ax = plt.subplots(
        figsize=(10.5, 5.8),
        constrained_layout=True,
    )

    image = ax.imshow(
        counts,
        aspect="auto",
        cmap="Blues",
        interpolation="nearest",
    )

    ax.set_xticks(
        range(len(materials))
    )
    ax.set_xticklabels(
        [
            material_display_name(name)
            for name in materials
        ],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(4))
    ax.set_yticklabels(
        [
            "Core",
            "Shell 1",
            "Shell 2",
            "Shell 3",
        ]
    )
    ax.set_xlabel(
        "Material",
        fontweight="bold",
    )
    ax.set_ylabel(
        "Layer",
        fontweight="bold",
    )
    ax.set_title(
        "Material occurrence by layer",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )

    count_min = float(np.min(counts))
    count_max = float(np.max(counts))
    threshold = (
        count_min
        + 0.55 * (count_max - count_min)
    )

    for layer in range(4):
        for material in range(
            len(materials)
        ):
            value = counts[layer, material]
            text_color = (
                "white"
                if value >= threshold
                else "black"
            )
            ax.text(
                material,
                layer,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                fontname=FONT_FAMILY,
                color=text_color,
            )

    set_bold_times_font(
        ax,
        grid=False,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    style_colorbar(
        colorbar,
        "Count",
    )

    save_figure(
        fig,
        "material_distribution.png",
    )


def plot_quality_by_wavelength(
    wavelengths: np.ndarray,
    q_ext: np.ndarray,
    q_sca: np.ndarray,
    q_abs: np.ndarray,
) -> None:
    residual = q_ext - q_sca - q_abs

    min_q_ext = np.min(
        q_ext,
        axis=0,
    )
    min_q_sca = np.min(
        q_sca,
        axis=0,
    )
    min_q_abs = np.min(
        q_abs,
        axis=0,
    )
    max_residual = np.max(
        np.abs(residual),
        axis=0,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.4),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        wavelengths,
        min_q_ext,
        label=r"Minimum $Q_{\mathrm{ext}}$",
        linewidth=CURVE_LINE_WIDTH,
        color="tab:blue",
    )
    axes[0].plot(
        wavelengths,
        min_q_sca,
        label=r"Minimum $Q_{\mathrm{sca}}$",
        linewidth=CURVE_LINE_WIDTH,
        color="tab:orange",
    )
    axes[0].plot(
        wavelengths,
        min_q_abs,
        label=r"Minimum $Q_{\mathrm{abs}}$",
        linewidth=CURVE_LINE_WIDTH,
        color="tab:green",
    )
    axes[0].axhline(
        0,
        color="black",
        linewidth=1.1,
    )
    axes[0].set_ylabel(
        "Minimum efficiency",
        fontweight="bold",
    )
    legend_top = axes[0].legend(
        loc="best",
        ncol=3,
    )
    style_legend(
        legend_top
    )
    set_bold_times_font(
        axes[0],
        grid=True,
    )
    add_panel_label(
        axes[0],
        "(a)",
    )

    axes[1].semilogy(
        wavelengths,
        np.maximum(
            max_residual,
            1e-15,
        ),
        color="tab:red",
        linewidth=CURVE_LINE_WIDTH,
        label=(
            r"$|Q_{\mathrm{ext}}"
            r"-Q_{\mathrm{sca}}"
            r"-Q_{\mathrm{abs}}|$"
        ),
    )
    axes[1].axhline(
        ENERGY_RESIDUAL_TOL,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label="Validation tolerance",
    )
    axes[1].set_xlabel(
        "Wavelength (nm)",
        fontweight="bold",
    )
    axes[1].set_ylabel(
        "Maximum energy residual",
        fontweight="bold",
    )
    legend_bottom = axes[1].legend(
        loc="best",
    )
    style_legend(
        legend_bottom
    )
    set_bold_times_font(
        axes[1],
        grid=True,
    )
    add_panel_label(
        axes[1],
        "(b)",
    )

    fig.suptitle(
        "Numerical quality assessment across wavelength",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        fontname=FONT_FAMILY,
    )

    save_figure(
        fig,
        "quality_by_wavelength.png",
    )


def plot_sample_spectra(
    sample_ids: np.ndarray,
    radii: np.ndarray,
    material_indices: np.ndarray,
    materials: list,
    wavelengths: np.ndarray,
    q_ext: np.ndarray,
    q_sca: np.ndarray,
    q_abs: np.ndarray,
) -> None:
    n_columns = 2
    n_rows = int(
        np.ceil(
            len(sample_ids) / n_columns
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14.5, 4.3 * n_rows),
        sharex=True,
        constrained_layout=True,
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

    for axis_index, sample_id in enumerate(
        sample_ids
    ):
        sample_id = int(sample_id)
        ax = axes[axis_index]

        ax.plot(
            wavelengths,
            q_ext[sample_id],
            label=r"$Q_{\mathrm{ext}}$",
            linewidth=CURVE_LINE_WIDTH,
            color="tab:blue",
        )
        ax.plot(
            wavelengths,
            q_sca[sample_id],
            label=r"$Q_{\mathrm{sca}}$",
            linewidth=CURVE_LINE_WIDTH,
            color="tab:orange",
        )
        ax.plot(
            wavelengths,
            q_abs[sample_id],
            label=r"$Q_{\mathrm{abs}}$",
            linewidth=CURVE_LINE_WIDTH,
            color="tab:green",
        )

        names = [
            material_display_name(
                materials[index]
            )
            for index in material_indices[
                sample_id
            ]
        ]

        radii_text = ", ".join(
            f"{value:.0f}"
            for value in radii[sample_id]
        )

        ax.set_title(
            f"Sample {sample_id} | "
            f"R = [{radii_text}] nm\n"
            f"{' / '.join(names)}",
            fontsize=9.5,
            fontweight="bold",
            pad=7,
        )
        ax.set_ylabel(
            "Efficiency",
            fontweight="bold",
        )
        legend = ax.legend(
            ncol=3,
            loc="best",
            fontsize=8.5,
        )
        style_legend(
            legend
        )
        set_bold_times_font(
            ax,
            grid=True,
        )
        add_panel_label(
            ax,
            f"({chr(97 + axis_index)})",
        )

    for remaining_axis in axes[
        len(sample_ids):
    ]:
        remaining_axis.axis("off")

    for axis_index, ax in enumerate(
        axes[:len(sample_ids)]
    ):
        row_index = axis_index // n_columns
        if row_index == n_rows - 1:
            ax.set_xlabel(
                "Wavelength (nm)",
                fontweight="bold",
            )

    fig.suptitle(
        "Representative PyMieDiff spectra",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        fontname=FONT_FAMILY,
    )

    save_figure(
        fig,
        "sample_spectra.png",
    )


# ==================== 主程序 ====================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{DATA_PATH}"
        )

    if DATA_PATH.stat().st_size == 0:
        raise RuntimeError(
            f"数据文件为空：{DATA_PATH}"
        )

    print(f"加载数据：{DATA_PATH}")

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    )

    required_keys = [
        "radii",
        "material_indices",
        "num_layers",
        "wavelengths",
        "Q_ext",
        "Q_sca",
        "Q_abs",
        "C_ext",
        "C_sca",
        "C_abs",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"数据集中缺少字段：{key}"
            )

    config = load_config(data)
    materials = list(config["materials"])

    radii = data["radii"].astype(
        np.float64
    )
    material_indices = data[
        "material_indices"
    ].astype(np.int64)
    num_layers = data[
        "num_layers"
    ].astype(np.int64)
    wavelengths = data[
        "wavelengths"
    ].astype(np.float64)

    q_ext = data["Q_ext"].astype(
        np.float64
    )
    q_sca = data["Q_sca"].astype(
        np.float64
    )
    q_abs = data["Q_abs"].astype(
        np.float64
    )

    c_ext = data["C_ext"].astype(
        np.float64
    )
    c_sca = data["C_sca"].astype(
        np.float64
    )
    c_abs = data["C_abs"].astype(
        np.float64
    )

    material_n = (
        data["material_n"].astype(
            np.float64
        )
        if "material_n" in data
        else None
    )
    material_k = (
        data["material_k"].astype(
            np.float64
        )
        if "material_k" in data
        else None
    )

    n_particles = radii.shape[0]
    n_wavelengths = len(wavelengths)
    expected_spectrum_shape = (
        n_particles,
        n_wavelengths,
    )

    # ---------- 基本形状 ----------
    correct_radii_shape = bool(
        radii.shape == (n_particles, 4)
    )
    correct_material_shape = bool(
        material_indices.shape
        == (n_particles, 4)
    )
    correct_layer_count = bool(
        np.all(num_layers == 4)
    )

    correct_spectrum_shapes = bool(
        q_ext.shape == expected_spectrum_shape
        and q_sca.shape
        == expected_spectrum_shape
        and q_abs.shape
        == expected_spectrum_shape
        and c_ext.shape
        == expected_spectrum_shape
        and c_sca.shape
        == expected_spectrum_shape
        and c_abs.shape
        == expected_spectrum_shape
    )

    # ---------- 有限性 ----------
    finite = bool(
        np.isfinite(radii).all()
        and np.isfinite(q_ext).all()
        and np.isfinite(q_sca).all()
        and np.isfinite(q_abs).all()
        and np.isfinite(c_ext).all()
        and np.isfinite(c_sca).all()
        and np.isfinite(c_abs).all()
    )

    # ---------- 整数结构 ----------
    integer_radii = bool(
        np.allclose(
            radii,
            np.rint(radii),
            atol=0,
            rtol=0,
        )
    )

    increasing_radii = bool(
        np.all(
            np.diff(radii, axis=1)
            > 0
        )
    )

    # ---------- 材料规则 ----------
    valid_material_indices = bool(
        np.all(material_indices >= 0)
        and np.all(
            material_indices
            < len(materials)
        )
    )

    adjacent_equal_count = int(
        np.sum(
            material_indices[:, :-1]
            == material_indices[:, 1:]
        )
    )

    air_index = materials.index("Air")
    air_in_shell_count = int(
        np.sum(
            material_indices[:, 1:]
            == air_index
        )
    )

    # ---------- Mie效率 ----------
    negative_q_ext_points = int(
        np.sum(
            q_ext < -NEGATIVE_TOL
        )
    )
    negative_q_sca_points = int(
        np.sum(
            q_sca < -NEGATIVE_TOL
        )
    )
    negative_q_abs_points = int(
        np.sum(
            q_abs < -NEGATIVE_TOL
        )
    )

    min_q_ext = float(np.min(q_ext))
    min_q_sca = float(np.min(q_sca))
    min_q_abs = float(np.min(q_abs))

    max_q_ext = float(np.max(q_ext))
    max_q_sca = float(np.max(q_sca))
    max_q_abs = float(np.max(q_abs))

    # ---------- 能量关系 ----------
    energy_residual = (
        q_ext - q_sca - q_abs
    )
    max_energy_residual = float(
        np.max(
            np.abs(energy_residual)
        )
    )
    mean_energy_residual = float(
        np.mean(
            np.abs(energy_residual)
        )
    )

    energy_conservation_passed = bool(
        max_energy_residual
        <= ENERGY_RESIDUAL_TOL
    )

    # ---------- 截面积一致性 ----------
    c_geo = (
        np.pi
        * radii[:, -1] ** 2
    )

    max_c_ext_error = float(
        np.max(
            np.abs(
                c_ext
                - q_ext * c_geo[:, None]
            )
        )
    )
    max_c_sca_error = float(
        np.max(
            np.abs(
                c_sca
                - q_sca * c_geo[:, None]
            )
        )
    )
    max_c_abs_error = float(
        np.max(
            np.abs(
                c_abs
                - q_abs * c_geo[:, None]
            )
        )
    )

    expected_c_ext = q_ext * c_geo[:, None]
    expected_c_sca = q_sca * c_geo[:, None]
    expected_c_abs = q_abs * c_geo[:, None]

    cross_section_consistency_passed = bool(
        np.allclose(
            c_ext,
            expected_c_ext,
            rtol=CROSS_SECTION_RTOL,
            atol=CROSS_SECTION_ATOL,
        )
        and np.allclose(
            c_sca,
            expected_c_sca,
            rtol=CROSS_SECTION_RTOL,
            atol=CROSS_SECTION_ATOL,
        )
        and np.allclose(
            c_abs,
            expected_c_abs,
            rtol=CROSS_SECTION_RTOL,
            atol=CROSS_SECTION_ATOL,
        )
    )

    # ---------- x=m*pi ----------
    x_outer = (
        2.0
        * np.pi
        * radii[:, -1, None]
        / wavelengths[None, :]
    )

    singular_mask = (
        np.abs(np.sin(x_outer))
        <= SINGULAR_TOL
    )
    singular_points = int(
        np.sum(singular_mask)
    )

    min_abs_sin_x = float(
        np.min(
            np.abs(np.sin(x_outer))
        )
    )

    # ---------- 唯一性 ----------
    unique_geometry_count = int(
        len(
            np.unique(
                radii,
                axis=0,
            )
        )
    )

    unique_material_sequences, sequence_counts = np.unique(
        material_indices,
        axis=0,
        return_counts=True,
    )

    unique_material_sequence_count = int(
        len(unique_material_sequences)
    )
    min_material_sequence_count = int(
        sequence_counts.min()
    )
    max_material_sequence_count = int(
        sequence_counts.max()
    )

    # ---------- 综合判断 ----------
    passed = bool(
        correct_radii_shape
        and correct_material_shape
        and correct_layer_count
        and correct_spectrum_shapes
        and finite
        and integer_radii
        and increasing_radii
        and valid_material_indices
        and adjacent_equal_count == 0
        and air_in_shell_count == 0
        and negative_q_ext_points == 0
        and negative_q_sca_points == 0
        and negative_q_abs_points == 0
        and energy_conservation_passed
        and cross_section_consistency_passed
        and singular_points == 0
        and unique_geometry_count
        == n_particles
        and unique_material_sequence_count
        == 400
    )

    report = {
        "passed": passed,
        "dataset_path": str(DATA_PATH),
        "n_particles": int(n_particles),
        "n_wavelengths": int(
            n_wavelengths
        ),
        "wavelength_min_nm": float(
            wavelengths.min()
        ),
        "wavelength_max_nm": float(
            wavelengths.max()
        ),
        "correct_radii_shape": (
            correct_radii_shape
        ),
        "correct_material_shape": (
            correct_material_shape
        ),
        "correct_layer_count": (
            correct_layer_count
        ),
        "correct_spectrum_shapes": (
            correct_spectrum_shapes
        ),
        "finite": finite,
        "integer_radii": integer_radii,
        "increasing_radii": (
            increasing_radii
        ),
        "valid_material_indices": (
            valid_material_indices
        ),
        "adjacent_equal_count": (
            adjacent_equal_count
        ),
        "air_in_shell_count": (
            air_in_shell_count
        ),
        "negative_q_ext_points": (
            negative_q_ext_points
        ),
        "negative_q_sca_points": (
            negative_q_sca_points
        ),
        "negative_q_abs_points": (
            negative_q_abs_points
        ),
        "min_q_ext": min_q_ext,
        "min_q_sca": min_q_sca,
        "min_q_abs": min_q_abs,
        "max_q_ext": max_q_ext,
        "max_q_sca": max_q_sca,
        "max_q_abs": max_q_abs,
        "max_energy_residual": (
            max_energy_residual
        ),
        "mean_energy_residual": (
            mean_energy_residual
        ),
        "energy_residual_tolerance": (
            ENERGY_RESIDUAL_TOL
        ),
        "energy_conservation_passed": (
            energy_conservation_passed
        ),
        "max_c_ext_consistency_error": (
            max_c_ext_error
        ),
        "max_c_sca_consistency_error": (
            max_c_sca_error
        ),
        "max_c_abs_consistency_error": (
            max_c_abs_error
        ),
        "cross_section_relative_tolerance": (
            CROSS_SECTION_RTOL
        ),
        "cross_section_absolute_tolerance": (
            CROSS_SECTION_ATOL
        ),
        "cross_section_consistency_passed": (
            cross_section_consistency_passed
        ),
        "singular_points": singular_points,
        "minimum_abs_sin_x": (
            min_abs_sin_x
        ),
        "unique_geometry_count": (
            unique_geometry_count
        ),
        "unique_material_sequence_count": (
            unique_material_sequence_count
        ),
        "min_material_sequence_count": (
            min_material_sequence_count
        ),
        "max_material_sequence_count": (
            max_material_sequence_count
        ),
    }

    atomic_write_json(
        OUTPUT_DIR
        / "validation_report.json",
        report,
    )

    # ---------- 代表性样本 ----------
    sample_ids = select_validation_samples(
        radii=radii,
        q_sca=q_sca,
        q_abs=q_abs,
        n_samples=N_VALIDATION_SAMPLES,
    )

    export_online_validation_data(
        sample_ids=sample_ids,
        radii=radii,
        material_indices=material_indices,
        materials=materials,
        wavelengths=wavelengths,
        q_ext=q_ext,
        q_sca=q_sca,
        q_abs=q_abs,
        c_ext=c_ext,
        c_sca=c_sca,
        c_abs=c_abs,
        material_n=material_n,
        material_k=material_k,
    )

    # ---------- 绘图 ----------
    # 统一使用Times New Roman加粗和出版级绘图格式。
    # 不调用seaborn样式，避免其覆盖字体、边框和网格设置。
    configure_plot_style()

    plot_geometry_distribution(radii)

    plot_material_distribution(
        material_indices,
        materials,
    )

    plot_quality_by_wavelength(
        wavelengths,
        q_ext,
        q_sca,
        q_abs,
    )

    plot_sample_spectra(
        sample_ids=sample_ids,
        radii=radii,
        material_indices=material_indices,
        materials=materials,
        wavelengths=wavelengths,
        q_ext=q_ext,
        q_sca=q_sca,
        q_abs=q_abs,
    )

    print()
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )

    print()
    print(
        "在线验证样本ID：",
        sample_ids.tolist(),
    )
    print(
        "在线验证文件目录：",
        ONLINE_VALIDATION_DIR,
    )

    if not passed:
        raise RuntimeError(
            "数据质量检查未通过。"
            "请先检查validation_report.json。"
        )

    print()
    print(
        "数据质量检查通过，可以进入训练阶段。"
    )


if __name__ == "__main__":
    main()
