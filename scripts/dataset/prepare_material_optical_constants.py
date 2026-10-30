"""
convert_material_data.py
将 refractiveindex.info 下载的 CSV 材料数据转换为统一的 NPZ 文件。

输入文件:
 'Air', 'Al2O3', 'BaSO4', 'PDMS', 'TiO2', 'SiO2'

输出文件:
    material_data.npz
        包含:
        wavelengths : 公共波长网格 (nm), shape (N_wl,)
        {mat}_n : 折射率实部, shape (N_wl,)
        {mat}_k : 消光系数, shape (N_wl,)

单位转换:
    原始波长单位为微米 (µm) -> 转换为纳米 (nm)
    n, k 无量纲，保持不变
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ================================
# 配置
# ================================
MATERIALS = ['Air', 'Al2O3', 'BaSO4', 'PDMS', 'TiO2', 'SiO2']
CSV_FILES = {mat: f"{mat}.csv" for mat in MATERIALS}

# 列名映射（处理不同 CSV 的列名变体）
COLUMN_MAP = {
    'wl': 'wavelength_um',
    'Wavelength': 'wavelength_um',
    'n': 'n',
    'k': 'k'
}


def read_material_csv(filepath):
    """
    读取单个材料 CSV，返回波长(um), n, k
    自动识别列名
    """
    df = pd.read_csv(filepath)
    # 重命名列
    df.rename(columns=COLUMN_MAP, inplace=True)
    # 提取数据
    wl_um = df['wavelength_um'].values
    n = df['n'].values
    k = df['k'].values
    return wl_um, n, k


# ================================
# 1. 读取所有材料数据，转换波长单位
# ================================
data = {}
all_wavelengths_nm = []  # 收集所有波长点（nm）

for mat in MATERIALS:
    fpath = CSV_FILES[mat]
    if not Path(fpath).exists():
        raise FileNotFoundError(f"找不到文件: {fpath}")

    wl_um, n, k = read_material_csv(fpath)
    wl_nm = wl_um * 1000.0  # 微米 -> 纳米

    data[mat] = {
        'wavelength_nm': wl_nm,
        'n': n,
        'k': k
    }
    all_wavelengths_nm.append(wl_nm)

# ================================
# 2. 构建公共波长网格（所有材料波长的并集，排序去重）
# ================================
common_wl = np.sort(np.unique(np.concatenate(all_wavelengths_nm)))

print(f"公共波长网格点数: {len(common_wl)}")
print(f"波长范围: {common_wl[0]:.2f} - {common_wl[-1]:.2f} nm")

# ================================
# 3. 将每种材料插值到公共网格
# ================================
interp_data = {'wavelengths': common_wl}

for mat in MATERIALS:
    wl_orig = data[mat]['wavelength_nm']
    n_orig = data[mat]['n']
    k_orig = data[mat]['k']

    # 线性插值，超出原始范围的部分用 NaN 填充（后续使用时会检查范围）
    n_interp = np.interp(common_wl, wl_orig, n_orig, left=np.nan, right=np.nan)
    k_interp = np.interp(common_wl, wl_orig, k_orig, left=np.nan, right=np.nan)

    interp_data[f'{mat}_n'] = n_interp
    interp_data[f'{mat}_k'] = k_interp

# ================================
# 4. 检查插值后是否有 NaN（表示公共网格超出某材料的原始范围）
# ================================
for mat in MATERIALS:
    n_arr = interp_data[f'{mat}_n']
    k_arr = interp_data[f'{mat}_k']
    if np.any(np.isnan(n_arr)) or np.any(np.isnan(k_arr)):
        nan_wl = common_wl[np.isnan(n_arr) | np.isnan(k_arr)]
        print(f"警告: 材料 {mat} 在波长 {nan_wl} 处无原始数据，插值结果为 NaN")
        # 可选：用最近邻填充或直接报错，这里仅警告

# ================================
# 5. 保存为 NPZ 文件
# ================================
np.savez('material_data.npz', **interp_data)

print("转换完成！已保存为 material_data.npz")
print("文件内容包含以下键：", list(interp_data.keys()))