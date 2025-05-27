import os

from matplotlib import patches
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime, timedelta
import re
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.ma.core import zeros_like, around
from sklearn.utils.extmath import density

from utils import just_spectrum, depart_ml_lat
from config_vis import adjust_all_font_sizes
from config_vis import setup_colorbar, auto_close_plot, format_tick, setup_colors, set_fonts_for_fig

# 全局设置 DPI 为 300
plt.rcParams['figure.dpi'] = 300


# colors = [colors[1]] * 6


# colors = ['black'] * 6
# from cartopy.util import add_cyclic_point  # 视情况而定

@auto_close_plot
def scatter_plots_combined_verify_turning_points(

        das_low, das_mid, save_path, cmap=None,
        projection=ccrs.Mollweide,
):
    """
    在左侧画散点图（低频 & 中频数据），右侧画两个区域的地理轮廓：
    1) data_frequency > 0.8 (红色 + 黑色边界线)
    2) highlight_mask 区域 (绿色 + 黑色边界线)
    不再对 high_freq_mask 做第二个区域的绘制，仅保留在散点图中（如不需要也可删除相关散点）。
    增加了一个额外的范围筛选参数range_filter，用于在散点图上绘制两个矩形框。
    """

    # 定义配色方案
    color_low = '#1f78b4'  # 深蓝
    color_mid = '#e66101'  # 深橙
    color_high = '#03C853'  # 深绿色
    color_filter_1 = color_low
    color_filter_2 = color_high

    da_low_1, da_low_2 = das_low
    da_mid_1, da_mid_2 = das_mid

    # 1) 检查形状一致性
    if da_low_1.shape != da_low_2.shape or da_mid_1.shape != da_mid_2.shape:
        raise ValueError("Data arrays must have the same shape.")

    # 2) flatten 处理，用于散点图
    flat_low_1 = da_low_1.values.flatten()
    flat_low_2 = da_low_2.values.flatten()
    flat_mid_1 = da_mid_1.values.flatten()
    flat_mid_2 = da_mid_2.values.flatten()

    mask_low = ~np.isnan(flat_low_1) & ~np.isnan(flat_low_2)
    mask_mid = ~np.isnan(flat_mid_1) & ~np.isnan(flat_mid_2)

    filtered_low_1 = flat_low_1[mask_low]
    filtered_low_2 = flat_low_2[mask_low]
    filtered_mid_1 = flat_mid_1[mask_mid]
    filtered_mid_2 = flat_mid_2[mask_mid]

    # 注意：根据你的情况，若 data_frequency 与 da_low_1 等形状对齐，且不再做任何经纬度裁剪
    nlat, nlon = da_low_1.shape


    # 4) 计算相关系数
    def calc_corr(a, b):
        if len(a) > 1 and len(b) > 1:
            return np.corrcoef(a, b)[0, 1]
        else:
            return np.nan

    correlation_coefficient_low = calc_corr(filtered_low_1, filtered_low_2)
    correlation_coefficient_mid = calc_corr(filtered_mid_1, filtered_mid_2)

    # 创建一个大小为 (7.5, 2.5) 的图形，并且创建 1 行 2 列的子图
    fig = plt.figure(figsize=(7.5, 2.5),constrained_layout=True)
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3, figure=fig)

    # -------------------- 左侧：散点图 --------------------
    gs_left = outer_gs[0, 0].subgridspec(1, 1)
    ax_scatter = fig.add_subplot(gs_left[0, 0])

    # (a) 中频散点
    if len(filtered_mid_1) > 1 and len(filtered_mid_2) > 1:
        sns.scatterplot(x=filtered_mid_2, y=filtered_mid_1, ax=ax_scatter,
                        color=color_mid, alpha=0.3, s=1)
        slope_mid, intercept_mid = np.polyfit(filtered_mid_2, filtered_mid_1, 1)
    else:
        slope_mid, intercept_mid = np.nan, np.nan
    # (b) 低频散点
    if len(filtered_low_1) > 1 and len(filtered_low_2) > 1:
        sns.scatterplot(x=filtered_low_2, y=filtered_low_1, ax=ax_scatter,
                        color=color_low, alpha=0.3, s=1)
        # 设置坐标轴与图例
    ax_scatter.set_xlabel('Wet-day frequency')
    ax_scatter.set_ylabel(f'{da_low_1.name}')
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim([0, 8])
    ax_scatter.legend(loc='upper right', markerscale=7, frameon=False)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax_scatter, right=True, top=True)

    # --- 在 ax_scatter 上加上 mean curve & 转折点 --- #

    # 低纬度 mean 拟合
    # 1) 定义更细的频率 bins
    bins_low = np.linspace(filtered_low_2.min(), filtered_low_2.max(), 10)
    # 2) 计算每个点所属 bin
    inds_low = np.digitize(filtered_low_2, bins_low)
    # 3) 计算 bin 中心和对应的 mean(y)
    bin_centers_low = 0.5 * (bins_low[1:] + bins_low[:-1])
    mean_low = np.array([
        filtered_low_1[inds_low == i].mean() if np.any(inds_low == i) else np.nan
        for i in range(1, len(bins_low))
    ])

    # 中纬度 mean 拟合
    bins_mid = np.linspace(filtered_mid_2.min(), filtered_mid_2.max(), 10)
    inds_mid = np.digitize(filtered_mid_2, bins_mid)
    bin_centers_mid = 0.5 * (bins_mid[1:] + bins_mid[:-1])
    mean_mid = np.array([
        filtered_mid_1[inds_mid == i].mean() if np.any(inds_mid == i) else np.nan
        for i in range(1, len(bins_mid))
    ])

    n_boot = 1000

    # 准备存放置信区间的数组
    ci_lower_low = np.full_like(mean_low, np.nan)
    ci_upper_low = np.full_like(mean_low, np.nan)

    for i in range(len(bin_centers_low)):
        # bin_inds 对应第 i 个 bin 的所有样本
        bin_inds = np.where(inds_low == i + 1)[0]
        data_bin = filtered_low_1[bin_inds]
        if data_bin.size > 0:
            # 每次对 data_bin 做一次有放回抽样，计算均值
            boot_means = np.random.choice(data_bin,
                                          size=(n_boot, data_bin.size),
                                          replace=True).mean(axis=1)
            # 取 2.5% 和 97.5% 分位数
            ci_lower_low[i], ci_upper_low[i] = np.percentile(boot_means, [2.5, 97.5])

    # 绘制带误差棒的 mean 曲线
    ax_scatter.errorbar(
        bin_centers_low, mean_low,
        yerr=[mean_low - ci_lower_low, ci_upper_low - mean_low],
        fmt='o-',  # 实线 + 实心圆点
        markersize=2,
        capsize=3,  # "工"形帽宽度
        linewidth=1,
        label='Low-lat Avg'
    )

    # （可选）如果想要填充误差带：
    ax_scatter.fill_between(
        bin_centers_low,
        ci_lower_low,
        ci_upper_low,
        alpha=0.2
    )

    # ——[对中纬度同理，计算并绘制误差棒]——
    ci_lower_mid = np.full_like(mean_mid, np.nan)
    ci_upper_mid = np.full_like(mean_mid, np.nan)

    for i in range(len(bin_centers_mid)):
        bin_inds = np.where(inds_mid == i + 1)[0]
        data_bin = filtered_mid_1[bin_inds]
        if data_bin.size > 0:
            boot_means = np.random.choice(data_bin,
                                          size=(n_boot, data_bin.size),
                                          replace=True).mean(axis=1)
            ci_lower_mid[i], ci_upper_mid[i] = np.percentile(boot_means, [2.5, 97.5])

    ax_scatter.errorbar(
        bin_centers_mid, mean_mid,
        yerr=[mean_mid - ci_lower_mid, ci_upper_mid - mean_mid],
        fmt='o-',
        markersize=2,
        capsize=3,
        linewidth=1,
        label='Mid-lat Avg'
    )
    # 1. 计算频率（x 轴）方向上的核密度估计
    from scipy.stats import gaussian_kde

    # 1. 构造二维点集
    points = np.vstack([filtered_low_2, filtered_low_1])  # x = wet-day freq, y = variable

    # 2. 二维核密度估计
    kde = gaussian_kde(points)
    dens = kde(points)  # 每个散点的密度

    # 3. 获取前 90% 的高密度散点索引（主干区域）
    density_threshold = np.quantile(dens, 0.3)  # 注意：值越小表示密度越低
    main_mask_scatter = dens > density_threshold  # True 表示高密度主干区域

    # 4. 用主干区域的点重新计算 mean 曲线（沿 x 分 bin）
    x_main = filtered_low_2[main_mask_scatter]
    y_main = filtered_low_1[main_mask_scatter]

    bins_main = np.linspace(0.3, x_main.max(), 20)
    inds_main = np.digitize(x_main, bins_main)
    bin_centers_main = 0.5 * (bins_main[1:] + bins_main[:-1])
    mean_main = np.array([
        y_main[inds_main == i].mean() if np.any(inds_main == i) else np.nan
        for i in range(1, len(bins_main))
    ])


    # 5. 求导并找转折点
    deriv_main = np.gradient(mean_main, bin_centers_main)
    # 构造中间区域 mask（排除边界）
    valid_idx = (np.arange(len(bin_centers_main)) > 0) & (np.arange(len(bin_centers_main)) < len(bin_centers_main) - 1)
    idx_main = np.nanargmin(np.abs(deriv_main[valid_idx]))
    x_turn = bin_centers_main[idx_main]

    # 6. 添加竖直虚线
    ax_scatter.axvline(x_turn, linestyle='--', color='grey', linewidth=1.2)



    # 更新图例
    handles, labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(handles=handles, labels=labels, loc='upper right', markerscale=2, frameon=False)

    # -------------------- 右侧：地图 --------------------
    ax_map = fig.add_subplot(outer_gs[0, 1], projection=projection(central_longitude=0))
    ax_map.coastlines()
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.bottom_labels = False
    gl.left_labels = False
    ax_map.set_global()


    # 准备网格坐标
    Lon, Lat = np.meshgrid(da_low_1.longitude, da_low_1.latitude)

    # [处理循环点函数] - 优化以解决白线问题
    def add_cyclic(data, lons):
        """为数据添加循环点"""
        cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
        return cyclic_data, cyclic_lons


    # 定义转折点左右两侧的高密度区域掩膜
    left_mask_main  = main_mask_scatter & (filtered_low_2 <= x_turn)
    right_mask_main = main_mask_scatter & (filtered_low_2 >  x_turn)

    # 在散点图上高亮左右两侧区域
    sns.scatterplot(
        x=filtered_low_2[left_mask_main],
        y=filtered_low_1[left_mask_main],
        ax=ax_scatter,
        color=color_filter_1,
        alpha=0.6,
        s=5,
        label='Dense Left'
    )
    sns.scatterplot(
        x=filtered_low_2[right_mask_main],
        y=filtered_low_1[right_mask_main],
        ax=ax_scatter,
        color=color_filter_2,
        alpha=0.6,
        s=5,
        label='Dense Right'
    )

    # 更新散点图图例
    handles, labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(handles=handles, labels=labels,
                      loc='upper right', markerscale=2, frameon=False)


    # ——— 右侧地图：把左右区域映射到2D mask ———
    mask_left_2d  = np.zeros_like(da_low_1, dtype=bool)
    mask_right_2d = np.zeros_like(da_low_1, dtype=bool)
    valid_idxs = np.where(mask_low)[0]

    for flat_idx in valid_idxs:
        i = flat_idx // nlon
        j = flat_idx % nlon
        pos = np.sum(valid_idxs <= flat_idx) - 1
        if left_mask_main[pos]:
            mask_left_2d[i, j] = True
        if right_mask_main[pos]:
            mask_right_2d[i, j] = True

    # 添加循环点
    cyclic_left,  lon_c  = add_cyclic(mask_left_2d,  da_low_1.longitude.values)
    cyclic_right, _      = add_cyclic(mask_right_2d, da_low_1.longitude.values)
    cyclic_Lon, cyclic_Lat = np.meshgrid(lon_c, da_low_1.latitude.values)

    # 绘制左侧区域（色块 + 边界）
    ax_map.contourf(cyclic_Lon, cyclic_Lat, cyclic_left,
                    levels=[0.5,1], colors=color_filter_1, alpha=1.0,
                    transform=ccrs.PlateCarree())
    ax_map.contour(cyclic_Lon, cyclic_Lat, cyclic_left, levels=[0.5],
                   colors='black', linewidths=1.2,
                   transform=ccrs.PlateCarree())

    # 绘制右侧区域（色块 + 边界）
    ax_map.contourf(cyclic_Lon, cyclic_Lat, cyclic_right,
                    levels=[0.5,1], colors=color_filter_2, alpha=1.0,
                    transform=ccrs.PlateCarree())
    ax_map.contour(cyclic_Lon, cyclic_Lat, cyclic_right, levels=[0.5],
                   colors='black', linewidths=1.2,
                   transform=ccrs.PlateCarree())

    # 更新地图图例
    left_patch  = mpatches.Patch(color=color_filter_1, label='Dense Left')
    right_patch = mpatches.Patch(color=color_filter_2, label='Dense Right')
    ax_map.legend(handles=[left_patch, right_patch],
                  loc='lower center', bbox_to_anchor=(0.5,-0.25),
                  ncol=2, frameon=False, handlelength=2, prop={'size':12})

    # —— [替换结束] ——
    # set_fonts_for_fig(fig,scale_factor=4,legend_scale_factor=0.8)
    for i, ax in enumerate([ax_scatter, ax_map]):
        label = f"{chr(97 + i)}"  # 按列顺序，标签'a', 'b', 'c'...
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # 保存 + 收尾
    plt.savefig(f'{save_path}_combined.svg', bbox_inches='tight')
    plt.show()
    plt.close()

    # 返回回归线信息
    df_slope_intercept_mid = pd.DataFrame({'Slope': [slope_mid], 'Intercept': [intercept_mid]},
                                          index=[f"{da_mid_1.name}_{da_mid_2.name}"])
    return df_slope_intercept_mid
@auto_close_plot
def scatter_plots_combined_final(


        das_low, das_mid, save_path, cmap=None,
        projection=ccrs.Mollweide,
):
    """
    在左侧画散点图（低频 & 中频数据），右侧画两个区域的地理轮廓：
    1) data_frequency > 0.8 (红色 + 黑色边界线)
    2) highlight_mask 区域 (绿色 + 黑色边界线)
    不再对 high_freq_mask 做第二个区域的绘制，仅保留在散点图中（如不需要也可删除相关散点）。
    增加了一个额外的范围筛选参数range_filter，用于在散点图上绘制两个矩形框。
    """
    # 定义配色方案
    color_low = '#1f78b4'  # 深蓝
    color_mid = '#e66101'  # 深橙
    color_high = '#03C853'  # 深绿色
    color_filter_1 = color_low
    color_filter_2 = color_high

    da_low_1, da_low_2 = das_low
    da_mid_1, da_mid_2 = das_mid

    # 1) 检查形状一致性
    if da_low_1.shape != da_low_2.shape or da_mid_1.shape != da_mid_2.shape:
        raise ValueError("Data arrays must have the same shape.")

    # 2) flatten 处理，用于散点图
    flat_low_1 = da_low_1.values.flatten()
    flat_low_2 = da_low_2.values.flatten()
    flat_mid_1 = da_mid_1.values.flatten()
    flat_mid_2 = da_mid_2.values.flatten()

    mask_low = ~np.isnan(flat_low_1) & ~np.isnan(flat_low_2)
    mask_mid = ~np.isnan(flat_mid_1) & ~np.isnan(flat_mid_2)

    filtered_low_1 = flat_low_1[mask_low]
    filtered_low_2 = flat_low_2[mask_low]
    filtered_mid_1 = flat_mid_1[mask_mid]
    filtered_mid_2 = flat_mid_2[mask_mid]

    # 注意：根据你的情况，若 data_frequency 与 da_low_1 等形状对齐，且不再做任何经纬度裁剪
    nlat, nlon = da_low_1.shape


    # 4) 计算相关系数
    def calc_corr(a, b):
        if len(a) > 1 and len(b) > 1:
            return np.corrcoef(a, b)[0, 1]
        else:
            return np.nan

    correlation_coefficient_low = calc_corr(filtered_low_1, filtered_low_2)
    correlation_coefficient_mid = calc_corr(filtered_mid_1, filtered_mid_2)

    # 创建一个大小为 (7.5, 2.5) 的图形，并且创建 1 行 2 列的子图
    fig = plt.figure(figsize=(7.5, 2.5),constrained_layout=True)
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3, figure=fig)

    # -------------------- 左侧：散点图 --------------------
    gs_left = outer_gs[0, 0].subgridspec(1, 1)
    ax_scatter = fig.add_subplot(gs_left[0, 0])

    # (a) 中频散点
    if len(filtered_mid_1) > 1 and len(filtered_mid_2) > 1:
        sns.scatterplot(x=filtered_mid_2, y=filtered_mid_1, ax=ax_scatter,
                        color=color_mid, alpha=0.3, s=1)
        slope_mid, intercept_mid = np.polyfit(filtered_mid_2, filtered_mid_1, 1)
    else:
        slope_mid, intercept_mid = np.nan, np.nan
    # (b) 低频散点
    if len(filtered_low_1) > 1 and len(filtered_low_2) > 1:
        sns.scatterplot(x=filtered_low_2, y=filtered_low_1, ax=ax_scatter,
                        color=color_low, alpha=0.3, s=1)
        # 设置坐标轴与图例
    ax_scatter.set_xlabel('Wet-day frequency')
    ax_scatter.set_ylabel(f'{da_low_1.name}')
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim([0, 8])
    ax_scatter.legend(loc='upper right', markerscale=7, frameon=False)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax_scatter, right=True, top=True)

    # --- 在 ax_scatter 上加上 mean curve & 转折点 --- #

    # 低纬度 mean 拟合
    # 1) 定义更细的频率 bins
    bins_low = np.linspace(filtered_low_2.min(), filtered_low_2.max(), 10)
    # 2) 计算每个点所属 bin
    inds_low = np.digitize(filtered_low_2, bins_low)
    # 3) 计算 bin 中心和对应的 mean(y)
    bin_centers_low = 0.5 * (bins_low[1:] + bins_low[:-1])
    mean_low = np.array([
        filtered_low_1[inds_low == i].mean() if np.any(inds_low == i) else np.nan
        for i in range(1, len(bins_low))
    ])

    # 中纬度 mean 拟合
    bins_mid = np.linspace(filtered_mid_2.min(), filtered_mid_2.max(), 10)
    inds_mid = np.digitize(filtered_mid_2, bins_mid)
    bin_centers_mid = 0.5 * (bins_mid[1:] + bins_mid[:-1])
    mean_mid = np.array([
        filtered_mid_1[inds_mid == i].mean() if np.any(inds_mid == i) else np.nan
        for i in range(1, len(bins_mid))
    ])

    n_boot = 1000

    # 准备存放置信区间的数组
    ci_lower_low = np.full_like(mean_low, np.nan)
    ci_upper_low = np.full_like(mean_low, np.nan)

    for i in range(len(bin_centers_low)):
        # bin_inds 对应第 i 个 bin 的所有样本
        bin_inds = np.where(inds_low == i + 1)[0]
        data_bin = filtered_low_1[bin_inds]
        if data_bin.size > 0:
            # 每次对 data_bin 做一次有放回抽样，计算均值
            boot_means = np.random.choice(data_bin,
                                          size=(n_boot, data_bin.size),
                                          replace=True).mean(axis=1)
            # 取 2.5% 和 97.5% 分位数
            ci_lower_low[i], ci_upper_low[i] = np.percentile(boot_means, [2.5, 97.5])

    # 绘制带误差棒的 mean 曲线
    ax_scatter.errorbar(
        bin_centers_low, mean_low,
        yerr=[mean_low - ci_lower_low, ci_upper_low - mean_low],
        fmt='o-',  # 实线 + 实心圆点
        markersize=2,
        capsize=4,  # "工"形帽宽度
        linewidth=1,
        label='Low-lat Avg'
    )

    # （可选）如果想要填充误差带：
    ax_scatter.fill_between(
        bin_centers_low,
        ci_lower_low,
        ci_upper_low,
        alpha=0.2
    )

    # ——[对中纬度同理，计算并绘制误差棒]——
    ci_lower_mid = np.full_like(mean_mid, np.nan)
    ci_upper_mid = np.full_like(mean_mid, np.nan)

    for i in range(len(bin_centers_mid)):
        bin_inds = np.where(inds_mid == i + 1)[0]
        data_bin = filtered_mid_1[bin_inds]
        if data_bin.size > 0:
            boot_means = np.random.choice(data_bin,
                                          size=(n_boot, data_bin.size),
                                          replace=True).mean(axis=1)
            ci_lower_mid[i], ci_upper_mid[i] = np.percentile(boot_means, [2.5, 97.5])

    ax_scatter.errorbar(
        bin_centers_mid, mean_mid,
        yerr=[mean_mid - ci_lower_mid, ci_upper_mid - mean_mid],
        fmt='o-',
        markersize=2,
        capsize=4,
        linewidth=1,
        label='Mid-lat Avg'
    )
    # 1. 计算频率（x 轴）方向上的核密度估计
    from scipy.stats import gaussian_kde

    # 1. 构造二维点集
    points = np.vstack([filtered_low_2, filtered_low_1])  # x = wet-day freq, y = variable

    # 2. 二维核密度估计
    kde = gaussian_kde(points)
    dens = kde(points)  # 每个散点的密度

    # --- 计算多个密度百分位对应的转折点列表 ---
    density_pcts = np.linspace(0.25, 0.7, 20)  # 30%, 32.5%, …, 50%
    x_turns = []
    for pct in density_pcts:
        thresh = np.quantile(dens, pct)
        mask = dens > thresh
        x_main = filtered_low_2[mask]
        y_main = filtered_low_1[mask]
        if x_main.size < 3:
            x_turns.append(np.nan)
            continue
        bins_tmp = np.linspace(0, x_main.max(), 20)
        inds_tmp = np.digitize(x_main, bins_tmp)
        centers = 0.5 * (bins_tmp[1:] + bins_tmp[:-1])
        mean_tmp = np.array([y_main[inds_tmp == i].mean() if np.any(inds_tmp == i) else np.nan
                             for i in range(1, len(bins_tmp))])
        deriv = np.gradient(mean_tmp, centers)
        valid = (np.arange(len(centers)) > 0) & (np.arange(len(centers)) < len(centers) - 1)
        idx = np.nanargmin(np.abs(deriv[valid]))
        x_turns.append(centers[valid][idx])

    # --- 统计误差棒 ---
    x_arr = np.array(x_turns)
    x_arr = x_arr[~np.isnan(x_arr)]
    mean_turn = x_arr.mean()
    low_ci = np.percentile(x_arr, 2.5)
    high_ci = np.percentile(x_arr, 97.5)

    # --- 在散点图上添加水平误差棒和一条竖线 ---
    # 竖线：位置为 mean_turn
    # ax_scatter.axvline(mean_turn, linestyle='--', color='grey', linewidth=1.2)

    # 误差棒：用 errorbar，y 坐标选在图顶稍下方
    # —— 把转折点 CI 的误差棒放到图底部，并去掉中间的 marker ——
    ymin, ymax = ax_scatter.get_ylim()
    # 距离下边界 5% 的高度
    y_position = ymin + 0.05 * (ymax - ymin)
    # yerr = [[0], [0]]  # vertical error = 0
    xerr = [[mean_turn - low_ci], [high_ci - mean_turn]]
    print(f'xerr = {low_ci,high_ci}')

    # # 只画横向的误差帽，去掉中间的 marker
    # ax_scatter.errorbar(
    #     mean_turn, y_position,
    #     xerr=xerr,   # 你的水平误差
    #     yerr=None,   # 不需要垂直误差
    #     fmt='None',  # 不画连线和 marker
    #     # markersize=1,
    #     capsize=4,   # 误差帽宽度
    #     capthick=1,
    #     color='grey',
    #     label='Turn-point CI'
    # )

    # 可选：把阈值与对应的转折点画成散点，帮助可视化
    # ax_scatter.scatter(x_arr, np.full_like(x_arr, ymax * 0.9),
    #                    marker='|', s=50, color='darkgrey', alpha=0.7)

    # 3. 获取前 90% 的高密度散点索引（主干区域）
    density_threshold = np.quantile(dens, 0.3)  # 注意：值越小表示密度越低
    main_mask_scatter = dens > density_threshold  # True 表示高密度主干区域

    # 4. 用主干区域的点重新计算 mean 曲线（沿 x 分 bin）
    x_main = filtered_low_2[main_mask_scatter]
    y_main = filtered_low_1[main_mask_scatter]

    bins_main = np.linspace(0.1, x_main.max(), 20)
    inds_main = np.digitize(x_main, bins_main)
    bin_centers_main = 0.5 * (bins_main[1:] + bins_main[:-1])
    mean_main = np.array([
        y_main[inds_main == i].mean() if np.any(inds_main == i) else np.nan
        for i in range(1, len(bins_main))
    ])


    # 5. 求导并找转折点
    deriv_main = np.gradient(mean_main, bin_centers_main)
    # 构造中间区域 mask（排除边界）
    valid_idx = (np.arange(len(bin_centers_main)) > 0) & (np.arange(len(bin_centers_main)) < len(bin_centers_main) - 1)
    idx_main = np.nanargmin(np.abs(deriv_main[valid_idx]))
    x_turn = bin_centers_main[idx_main]

    # 6. 添加竖直虚线
    ax_scatter.axvline(x_turn, linestyle='--', color='grey', linewidth=1.2)
    print(f'x_turn = {x_turn}')




    # 更新图例
    handles, labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(handles=handles, labels=labels, loc='upper right', markerscale=1, frameon=False)

    # -------------------- 右侧：地图 --------------------
    ax_map = fig.add_subplot(outer_gs[0, 1], projection=projection(central_longitude=0))
    ax_map.coastlines()
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.bottom_labels = False
    gl.left_labels = False
    ax_map.set_global()


    # 准备网格坐标
    Lon, Lat = np.meshgrid(da_low_1.longitude, da_low_1.latitude)

    # [处理循环点函数] - 优化以解决白线问题
    def add_cyclic(data, lons):
        """为数据添加循环点"""
        cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
        return cyclic_data, cyclic_lons


    # 定义转折点左右两侧的高密度区域掩膜
    left_mask_main  = main_mask_scatter & (filtered_low_2 <= x_turn)
    right_mask_main = main_mask_scatter & (filtered_low_2 >  x_turn)

    # —— 用双色半透明阴影＋明显无重叠轮廓 ——

    # 1) 构造网格
    xbins = np.linspace(ax_scatter.get_xlim()[0], ax_scatter.get_xlim()[1], 150)
    ybins = np.linspace(ax_scatter.get_ylim()[0], ax_scatter.get_ylim()[1], 150)

    # 2) 分别计算左右高密度直方图
    hist_l, xedges, yedges = np.histogram2d(
        filtered_low_2[left_mask_main],
        filtered_low_1[left_mask_main],
        bins=[xbins, ybins]
    )
    hist_r, _, _ = np.histogram2d(
        filtered_low_2[right_mask_main],
        filtered_low_1[right_mask_main],
        bins=[xbins, ybins]
    )

    # 3) 生成掩膜，并消除重叠
    mask_l = hist_l.T > 0
    mask_r = hist_r.T > 0

    # 找到 x_turn 对应的 bin 索引
    bin_idx = np.searchsorted(xedges, x_turn) - 1
    # 左侧只保留 bin_idx 及之前，右侧只保留之后
    mask_l[:, bin_idx + 1:] = False
    mask_r[:, :bin_idx + 1] = False

    # 4) 绘制阴影和加粗轮廓
    from scipy.ndimage import gaussian_filter

    # 假设 mask_l, mask_r 已经按你原来的方式生成，是 bool 或 0/1 数组
    # 先对它们做高斯滤波，得到连续场
    mask_l= gaussian_filter(mask_l.astype(float), sigma=1)
    mask_r= gaussian_filter(mask_r.astype(float), sigma=1)

    X, Y = np.meshgrid(xedges, yedges)


    # 左侧阴影 + 轮廓
    ax_scatter.contourf(
        X[:-1, :-1], Y[:-1, :-1], mask_l,
        levels=[0.5, 1.0],
        colors=[color_filter_2],
        alpha=0.4
    )
    ax_scatter.contour(
        X[:-1, :-1], Y[:-1, :-1], mask_l,
        levels=[0.5],
        colors=[color_filter_2],
        linewidths=1.5,
        linestyles='-',
        alpha=0.9
    )

    # 右侧阴影 + 轮廓
    ax_scatter.contourf(
        X[:-1, :-1], Y[:-1, :-1], mask_r,
        levels=[0.5, 1.0],
        colors=[color_filter_1],
        alpha=0.4
    )
    ax_scatter.contour(
        X[:-1, :-1], Y[:-1, :-1], mask_r,
        levels=[0.5],
        colors=[color_filter_1],
        linewidths=1.5,
        linestyles='-',
        alpha=0.9
    )

    # 5) 更新图例，加入阴影块说明
    from matplotlib.patches import Patch
    left_patch = Patch(facecolor=color_filter_1, alpha=0.4, label='Filter 1')
    right_patch = Patch(facecolor=color_filter_2, alpha=0.4, label='Filter 2')
    handles, labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(handles + [left_patch, right_patch],
                      labels + ['Filter 1', 'Filter 2'],
                      loc='upper right', frameon=False)

    # ——— 右侧地图：把左右区域映射到2D mask ———
    mask_left_2d  = np.zeros_like(da_low_1, dtype=bool)
    mask_right_2d = np.zeros_like(da_low_1, dtype=bool)
    valid_idxs = np.where(mask_low)[0]

    for flat_idx in valid_idxs:
        i = flat_idx // nlon
        j = flat_idx % nlon
        pos = np.sum(valid_idxs <= flat_idx) - 1
        if left_mask_main[pos]:
            mask_left_2d[i, j] = True
        if right_mask_main[pos]:
            mask_right_2d[i, j] = True

    # 添加循环点
    cyclic_left,  lon_c  = add_cyclic(mask_left_2d,  da_low_1.longitude.values)
    cyclic_right, _      = add_cyclic(mask_right_2d, da_low_1.longitude.values)
    cyclic_Lon, cyclic_Lat = np.meshgrid(lon_c, da_low_1.latitude.values)

    # 绘制左侧区域（色块 + 边界）
    ax_map.contourf(cyclic_Lon, cyclic_Lat, cyclic_left,
                    levels=[0.5,1], colors=color_filter_2, alpha=1.0,
                    transform=ccrs.PlateCarree())
    ax_map.contour(cyclic_Lon, cyclic_Lat, cyclic_left, levels=[0.5],
                   colors='black', linewidths=1.2,
                   transform=ccrs.PlateCarree())

    # 绘制右侧区域（色块 + 边界）
    ax_map.contourf(cyclic_Lon, cyclic_Lat, cyclic_right,
                    levels=[0.5,1], colors=color_filter_1, alpha=1.0,
                    transform=ccrs.PlateCarree())
    ax_map.contour(cyclic_Lon, cyclic_Lat, cyclic_right, levels=[0.5],
                   colors='black', linewidths=1.2,
                   transform=ccrs.PlateCarree())

    # 更新地图图例
    left_patch  = mpatches.Patch(color=color_filter_2, label='Filter 2')
    right_patch = mpatches.Patch(color=color_filter_1, label='Filter 1')
    ax_map.legend(handles=[left_patch, right_patch],
                  loc='lower center', bbox_to_anchor=(0.5,-0.25),
                  ncol=2, frameon=False, handlelength=2, prop={'size':12})

    # —— [替换结束] ——
    # set_fonts_for_fig(fig,scale_factor=4,legend_scale_factor=0.8)
    for i, ax in enumerate([ax_scatter, ax_map]):
        label = f"{chr(97 + i)}"  # 按列顺序，标签'a', 'b', 'c'...
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # 保存 + 收尾
    plt.savefig(f'{save_path}_combined.svg', bbox_inches='tight')
    plt.show()
    plt.close()

    # 返回回归线信息
    df_slope_intercept_mid = pd.DataFrame({'Slope': [slope_mid], 'Intercept': [intercept_mid]},
                                          index=[f"{da_mid_1.name}_{da_mid_2.name}"])
    return df_slope_intercept_mid

@auto_close_plot
def scatter_plots_combined_no_rect(

        das_low, das_mid, data_frequency, save_path, cmap=None,
        projection=ccrs.Mollweide, limited_key=None, range_filter=None
):
    """
    在左侧画散点图（低频 & 中频数据），右侧画两个区域的地理轮廓：
    1) data_frequency > 0.8 (红色 + 黑色边界线)
    2) highlight_mask 区域 (绿色 + 黑色边界线)
    不再对 high_freq_mask 做第二个区域的绘制，仅保留在散点图中（如不需要也可删除相关散点）。
    增加了一个额外的范围筛选参数range_filter，用于在散点图上绘制两个矩形框。
    """
    # 定义配色方案
    color_low = '#1f78b4'  # 深蓝
    color_mid = '#e66101'  # 深橙
    color_high = '#03C853'  # 深绿色
    color_filter_1 = color_low
    color_filter_2 = color_high
    # color_filter_1='#303f9f'
    # color_filter_2='#008b48'
    #
    if limited_key is None:
        limited_key = {}

    if range_filter is None:
        range_filter = {}

    da_low_1, da_low_2 = das_low
    da_mid_1, da_mid_2 = das_mid
    df_low, df_mid = depart_ml_lat([data_frequency])

    # 1) 检查形状一致性
    if da_low_1.shape != da_low_2.shape or da_mid_1.shape != da_mid_2.shape:
        raise ValueError("Data arrays must have the same shape.")

    # 2) flatten 处理，用于散点图
    flat_low_1 = da_low_1.values.flatten()
    flat_low_2 = da_low_2.values.flatten()
    flat_mid_1 = da_mid_1.values.flatten()
    flat_mid_2 = da_mid_2.values.flatten()
    freq_1d_low = df_low[0].values.flatten()  # shape = (nlat, nlon)
    freq_1d_mid = df_mid[0].values.flatten()  # shape = (nlat, nlon)

    mask_low = ~np.isnan(flat_low_1) & ~np.isnan(flat_low_2)
    mask_mid = ~np.isnan(flat_mid_1) & ~np.isnan(flat_mid_2)
    # mask_fre=np.concatenate((mask_low, mask_mid))

    filtered_low_1 = flat_low_1[mask_low]
    filtered_low_2 = flat_low_2[mask_low]
    filtered_mid_1 = flat_mid_1[mask_mid]
    filtered_mid_2 = flat_mid_2[mask_mid]

    # 注意：根据你的情况，若 data_frequency 与 da_low_1 等形状对齐，且不再做任何经纬度裁剪
    nlat, nlon = da_low_1.shape
    # freq_1d = freq_2d.flatten()
    # freq_1d_valid = freq_1d
    # # freq_1d_valid = freq_1d[mask_low]

    # 3) （可选）得到高频/低频散点的布尔掩膜
    #    你之前有在散点图上标绿色散点用这个：
    high_freq_mask_low = (freq_1d_low > 0.5)
    high_freq_mask_mid = (freq_1d_low > 0.5)

    # 4) 计算相关系数
    def calc_corr(a, b):
        if len(a) > 1 and len(b) > 1:
            return np.corrcoef(a, b)[0, 1]
        else:
            return np.nan

    correlation_coefficient_low = calc_corr(filtered_low_1, filtered_low_2)
    correlation_coefficient_mid = calc_corr(filtered_mid_1, filtered_mid_2)

    # 创建一个大小为 (7.5, 2.5) 的图形，并且创建 1 行 2 列的子图
    fig = plt.figure(figsize=(7.5, 2.5),constrained_layout=True)
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3, figure=fig)

    # -------------------- 左侧：散点图 --------------------
    gs_left = outer_gs[0, 0].subgridspec(1, 1)
    ax_scatter = fig.add_subplot(gs_left[0, 0])

    # (a) 中频散点
    if len(filtered_mid_1) > 1 and len(filtered_mid_2) > 1:
        sns.scatterplot(x=filtered_mid_2, y=filtered_mid_1, ax=ax_scatter,
                        color=color_mid, alpha=0.3, s=1)
        slope_mid, intercept_mid = np.polyfit(filtered_mid_2, filtered_mid_1, 1)
    else:
        slope_mid, intercept_mid = np.nan, np.nan
    # (b) 低频散点
    if len(filtered_low_1) > 1 and len(filtered_low_2) > 1:
        sns.scatterplot(x=filtered_low_2, y=filtered_low_1, ax=ax_scatter,
                        color=color_low, alpha=0.3, s=1)
        # 设置坐标轴与图例
    ax_scatter.set_xlabel('Wet-day frequency')
    ax_scatter.set_ylabel(f'{da_low_1.name}')
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim([0, 8])
    ax_scatter.legend(loc='upper right', markerscale=7, frameon=False)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax_scatter, right=True, top=True)

    # (c) 对低频数据增加范围筛选并在散点图上高亮（highlight_mask）
    highlight_mask = np.ones_like(filtered_low_1, dtype=bool)
    if da_low_1.name in limited_key:
        y_min, y_max = limited_key[da_low_1.name]
        highlight_mask &= (filtered_low_1 >= y_min) & (filtered_low_1 <= y_max)
    if da_low_2.name in limited_key:
        x_min, x_max = limited_key[da_low_2.name]
        highlight_mask &= (filtered_low_2 >= x_min) & (filtered_low_2 <= x_max)

    # 添加额外的范围筛选
    range_highlight_mask = np.ones_like(filtered_low_1, dtype=bool)
    if da_low_1.name in range_filter:
        y_min, y_max = range_filter[da_low_1.name]
        range_highlight_mask &= (filtered_low_1 >= y_min) & (filtered_low_1 <= y_max)
    if da_low_2.name in range_filter:
        x_min, x_max = range_filter[da_low_2.name]
        range_highlight_mask &= (filtered_low_2 >= x_min) & (filtered_low_2 <= x_max)

    # 绘制两个矩形框，颜色与右侧地图的区域颜色对应
    # 第一个矩形框 (limit_key) - 与蓝色区域对应
    if da_low_2.name in limited_key and da_low_1.name in limited_key:
        x_min, x_max = limited_key[da_low_2.name]
        y_min, y_max = limited_key[da_low_1.name]
        rect1 = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=1.5, edgecolor=color_filter_1, facecolor='none',
                                  linestyle='-', label='Filter 1')
        ax_scatter.add_patch(rect1)

    # 第二个矩形框 (range_filter) - 与绿色区域对应
    if da_low_2.name in range_filter and da_low_1.name in range_filter:
        x_min, x_max = range_filter[da_low_2.name]
        y_min, y_max = range_filter[da_low_1.name]
        rect2 = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=1.5, edgecolor=color_filter_2, facecolor='none',
                                  linestyle='-', label='Filter 2')
        ax_scatter.add_patch(rect2)

    # --- 在 ax_scatter 上加上 mean curve & 转折点 --- #

    # 低纬度 mean 拟合
    # 1) 定义更细的频率 bins
    bins_low = np.linspace(filtered_low_2.min(), filtered_low_2.max(), 10)
    # 2) 计算每个点所属 bin
    inds_low = np.digitize(filtered_low_2, bins_low)
    # 3) 计算 bin 中心和对应的 mean(y)
    bin_centers_low = 0.5 * (bins_low[1:] + bins_low[:-1])
    mean_low = np.array([
        filtered_low_1[inds_low == i].mean() if np.any(inds_low == i) else np.nan
        for i in range(1, len(bins_low))
    ])

    # 中纬度 mean 拟合
    bins_mid = np.linspace(filtered_mid_2.min(), filtered_mid_2.max(), 10)
    inds_mid = np.digitize(filtered_mid_2, bins_mid)
    bin_centers_mid = 0.5 * (bins_mid[1:] + bins_mid[:-1])
    mean_mid = np.array([
        filtered_mid_1[inds_mid == i].mean() if np.any(inds_mid == i) else np.nan
        for i in range(1, len(bins_mid))
    ])

    n_boot = 1000

    # 准备存放置信区间的数组
    ci_lower_low = np.full_like(mean_low, np.nan)
    ci_upper_low = np.full_like(mean_low, np.nan)

    for i in range(len(bin_centers_low)):
        # bin_inds 对应第 i 个 bin 的所有样本
        bin_inds = np.where(inds_low == i + 1)[0]
        data_bin = filtered_low_1[bin_inds]
        if data_bin.size > 0:
            # 每次对 data_bin 做一次有放回抽样，计算均值
            boot_means = np.random.choice(data_bin,
                                          size=(n_boot, data_bin.size),
                                          replace=True).mean(axis=1)
            # 取 2.5% 和 97.5% 分位数
            ci_lower_low[i], ci_upper_low[i] = np.percentile(boot_means, [2.5, 97.5])

    # 绘制带误差棒的 mean 曲线
    ax_scatter.errorbar(
        bin_centers_low, mean_low,
        yerr=[mean_low - ci_lower_low, ci_upper_low - mean_low],
        fmt='o-',  # 实线 + 实心圆点
        markersize=2,
        capsize=3,  # "工"形帽宽度
        linewidth=1,
        label='Low-lat Avg'
    )

    # （可选）如果想要填充误差带：
    ax_scatter.fill_between(
        bin_centers_low,
        ci_lower_low,
        ci_upper_low,
        alpha=0.2
    )

    # ——[对中纬度同理，计算并绘制误差棒]——
    ci_lower_mid = np.full_like(mean_mid, np.nan)
    ci_upper_mid = np.full_like(mean_mid, np.nan)

    for i in range(len(bin_centers_mid)):
        bin_inds = np.where(inds_mid == i + 1)[0]
        data_bin = filtered_mid_1[bin_inds]
        if data_bin.size > 0:
            boot_means = np.random.choice(data_bin,
                                          size=(n_boot, data_bin.size),
                                          replace=True).mean(axis=1)
            ci_lower_mid[i], ci_upper_mid[i] = np.percentile(boot_means, [2.5, 97.5])

    ax_scatter.errorbar(
        bin_centers_mid, mean_mid,
        yerr=[mean_mid - ci_lower_mid, ci_upper_mid - mean_mid],
        fmt='o-',
        markersize=2,
        capsize=3,
        linewidth=1,
        label='Mid-lat Avg'
    )
    # 1. 计算频率（x 轴）方向上的核密度估计
    from scipy.stats import gaussian_kde

    # 1. 构造二维点集
    points = np.vstack([filtered_low_2, filtered_low_1])  # x = wet-day freq, y = variable

    # 2. 二维核密度估计
    kde = gaussian_kde(points)
    dens = kde(points)  # 每个散点的密度

    # 3. 获取前 90% 的高密度散点索引（主干区域）
    density_threshold = np.quantile(dens, 0.4)  # 注意：值越小表示密度越低
    main_mask_scatter = dens > density_threshold  # True 表示高密度主干区域

    # 4. 用主干区域的点重新计算 mean 曲线（沿 x 分 bin）
    x_main = filtered_low_2[main_mask_scatter]
    y_main = filtered_low_1[main_mask_scatter]

    bins_main = np.linspace(x_main.min(), x_main.max(), 20)
    inds_main = np.digitize(x_main, bins_main)
    bin_centers_main = 0.5 * (bins_main[1:] + bins_main[:-1])
    mean_main = np.array([
        y_main[inds_main == i].mean() if np.any(inds_main == i) else np.nan
        for i in range(1, len(bins_main))
    ])


    # 5. 求导并找转折点
    deriv_main = np.gradient(mean_main, bin_centers_main)
    # 构造中间区域 mask（排除边界）
    valid_idx = (np.arange(len(bin_centers_main)) > 0) & (np.arange(len(bin_centers_main)) < len(bin_centers_main) - 1)
    idx_main = np.nanargmin(np.abs(deriv_main[valid_idx]))
    x_turn = bin_centers_main[idx_main]

    # 6. 添加竖直虚线
    ax_scatter.axvline(x_turn, linestyle='--', color='grey', linewidth=1.2)



    # 更新图例
    handles, labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(handles=handles, labels=labels, loc='upper right', markerscale=2, frameon=False)

    # -------------------- 右侧：地图 --------------------
    ax_map = fig.add_subplot(outer_gs[0, 1], projection=projection(central_longitude=0))
    ax_map.coastlines()
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.bottom_labels = False
    gl.left_labels = False
    ax_map.set_global()

    # [A] 构造 data_frequency > 0.5 的 2D mask (红色区域)
    mask_freq_2d = (data_frequency > 0.5)  # shape=(nlat,nlon)

    # [B] 构造 highlight_mask 对应的 2D mask (蓝色区域)
    mask_highlight_2d = np.zeros_like(da_low_1, dtype=bool)
    valid_indices = np.where(mask_low)[0]  # flat 索引数组，对应 filtered_low_1/2
    for flat_idx in valid_indices:
        i = flat_idx // nlon
        j = flat_idx % nlon
        # 计算这个 flat_idx 在 filtered_low_1/2 里的相对位置
        pos_in_filtered = np.sum(valid_indices <= flat_idx) - 1
        if highlight_mask[pos_in_filtered]:
            mask_highlight_2d[i, j] = True

    # [C] 构造 range_highlight_mask 对应的 2D mask (额外的筛选条件)
    mask_range_2d = np.zeros_like(da_low_1, dtype=bool)
    for flat_idx in valid_indices:
        i = flat_idx // nlon
        j = flat_idx % nlon
        pos_in_filtered = np.sum(valid_indices <= flat_idx) - 1
        if range_highlight_mask[pos_in_filtered]:
            mask_range_2d[i, j] = True

    # [D] 构造两个筛选条件的交集 (用于绿色区域)
    # 高频区域与两个矩形的交集
    mask_freq_2d = mask_freq_2d.sel(latitude=slice(30, -30))
    mask_combined = xr.DataArray(mask_range_2d,
                                 coords=mask_freq_2d.coords,  # 继承原 DataArray 的坐标
                                 dims=mask_freq_2d.dims,  # 继承原 DataArray 的维度
                                 attrs=mask_freq_2d.attrs)

    # 准备网格坐标
    Lon, Lat = np.meshgrid(da_low_1.longitude, da_low_1.latitude)

    # [处理循环点函数] - 优化以解决白线问题
    def add_cyclic(data, lons):
        """为数据添加循环点"""
        cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
        return cyclic_data, cyclic_lons

    # 处理所有需要循环点的数据
    # 处理mask_highlight_2d的循环点
    cyclic_highlight, cyclic_lon = add_cyclic(mask_highlight_2d, da_low_1.longitude.values)
    cyclic_Lon, cyclic_Lat = np.meshgrid(cyclic_lon, da_low_1.latitude.values)

    # 处理mask_range_2d的循环点
    cyclic_range, _ = add_cyclic(mask_range_2d, da_low_1.longitude.values)

    # 处理mask_freq_2d的循环点
    # 根据mask_freq_2d的维度获取经纬度值
    freq_lon_values = mask_freq_2d.longitude.values
    freq_lat_values = mask_freq_2d.latitude.values
    cyclic_freq, cyclic_freq_lon = add_cyclic(mask_freq_2d.values, freq_lon_values)

    # 处理mask_combined的循环点
    # 确保mask_combined与mask_freq_2d具有相同的形状
    cyclic_combined, _ = add_cyclic(mask_combined.values, freq_lon_values)

    # 创建循环点的网格
    cyclic_freq_Lon, cyclic_freq_Lat = np.meshgrid(cyclic_freq_lon, freq_lat_values)

    # -- 在地图上画区域 B: highlight_mask 对应 (蓝色+黑色边界)
    # 使用循环点数据来绘制
    # —— [替换开始] ——

    # 定义转折点左右两侧的高密度区域掩膜
    left_mask_main  = main_mask_scatter & (filtered_low_2 <= x_turn)
    right_mask_main = main_mask_scatter & (filtered_low_2 >  x_turn)

    # # 在散点图上高亮左右两侧区域
    # sns.scatterplot(
    #     x=filtered_low_2[left_mask_main],
    #     y=filtered_low_1[left_mask_main],
    #     ax=ax_scatter,
    #     color=color_filter_1,
    #     alpha=0.6,
    #     s=5,
    #     label='Dense Left'
    # )
    # sns.scatterplot(
    #     x=filtered_low_2[right_mask_main],
    #     y=filtered_low_1[right_mask_main],
    #     ax=ax_scatter,
    #     color=color_filter_2,
    #     alpha=0.6,
    #     s=5,
    #     label='Dense Right'
    # )

    # 更新散点图图例
    handles, labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(handles=handles, labels=labels,
                      loc='upper right', markerscale=2, frameon=False)


    # ——— 右侧地图：把左右区域映射到2D mask ———
    mask_left_2d  = np.zeros_like(da_low_1, dtype=bool)
    mask_right_2d = np.zeros_like(da_low_1, dtype=bool)
    valid_idxs = np.where(mask_low)[0]

    for flat_idx in valid_idxs:
        i = flat_idx // nlon
        j = flat_idx % nlon
        pos = np.sum(valid_idxs <= flat_idx) - 1
        if left_mask_main[pos]:
            mask_left_2d[i, j] = True
        if right_mask_main[pos]:
            mask_right_2d[i, j] = True

    # 添加循环点
    cyclic_left,  lon_c  = add_cyclic(mask_left_2d,  da_low_1.longitude.values)
    cyclic_right, _      = add_cyclic(mask_right_2d, da_low_1.longitude.values)
    cyclic_Lon, cyclic_Lat = np.meshgrid(lon_c, da_low_1.latitude.values)

    # 绘制左侧区域（色块 + 边界）
    ax_map.contourf(cyclic_Lon, cyclic_Lat, cyclic_left,
                    levels=[0.5,1], colors=color_filter_1, alpha=1.0,
                    transform=ccrs.PlateCarree())
    ax_map.contour(cyclic_Lon, cyclic_Lat, cyclic_left, levels=[0.5],
                   colors='black', linewidths=1.2,
                   transform=ccrs.PlateCarree())

    # 绘制右侧区域（色块 + 边界）
    ax_map.contourf(cyclic_Lon, cyclic_Lat, cyclic_right,
                    levels=[0.5,1], colors=color_filter_2, alpha=1.0,
                    transform=ccrs.PlateCarree())
    ax_map.contour(cyclic_Lon, cyclic_Lat, cyclic_right, levels=[0.5],
                   colors='black', linewidths=1.2,
                   transform=ccrs.PlateCarree())

    # 更新地图图例
    left_patch  = mpatches.Patch(color=color_filter_1, label='Dense Left')
    right_patch = mpatches.Patch(color=color_filter_2, label='Dense Right')
    ax_map.legend(handles=[left_patch, right_patch],
                  loc='lower center', bbox_to_anchor=(0.5,-0.25),
                  ncol=2, frameon=False, handlelength=2, prop={'size':12})

# —— [替换结束] ——
    set_fonts_for_fig(fig,scale_factor=4,legend_scale_factor=0.8)
    for i, ax in enumerate([ax_scatter, ax_map]):
        label = f"{chr(97 + i)}"  # 按列顺序，标签'a', 'b', 'c'...
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # 保存 + 收尾
    plt.savefig(f'{save_path}_combined.svg', bbox_inches='tight')
    plt.show()
    plt.close()

    # 返回回归线信息
    df_slope_intercept_mid = pd.DataFrame({'Slope': [slope_mid], 'Intercept': [intercept_mid]},
                                          index=[f"{da_mid_1.name}_{da_mid_2.name}"])
    return df_slope_intercept_mid
@auto_close_plot
def scatter_plots_combined_with_map_new(

        das_low, das_mid, data_frequency, save_path, cmap=None,
        projection=ccrs.Mollweide, limited_key=None, range_filter=None
):
    """
    在左侧画散点图（低频 & 中频数据），右侧画两个区域的地理轮廓：
    1) data_frequency > 0.8 (红色 + 黑色边界线)
    2) highlight_mask 区域 (绿色 + 黑色边界线)
    不再对 high_freq_mask 做第二个区域的绘制，仅保留在散点图中（如不需要也可删除相关散点）。
    增加了一个额外的范围筛选参数range_filter，用于在散点图上绘制两个矩形框。
    """
    # 定义配色方案
    color_low = '#1f78b4'  # 深蓝
    color_mid = '#e66101'  # 深橙
    color_high = '#03C853'  # 深绿色
    color_filter_1 = color_low
    color_filter_2 = color_high
    # color_filter_1='#303f9f'
    # color_filter_2='#008b48'
    #
    if limited_key is None:
        limited_key = {}

    if range_filter is None:
        range_filter = {}

    da_low_1, da_low_2 = das_low
    da_mid_1, da_mid_2 = das_mid
    df_low, df_mid = depart_ml_lat([data_frequency])

    # 1) 检查形状一致性
    if da_low_1.shape != da_low_2.shape or da_mid_1.shape != da_mid_2.shape:
        raise ValueError("Data arrays must have the same shape.")

    # 2) flatten 处理，用于散点图
    flat_low_1 = da_low_1.values.flatten()
    flat_low_2 = da_low_2.values.flatten()
    flat_mid_1 = da_mid_1.values.flatten()
    flat_mid_2 = da_mid_2.values.flatten()
    freq_1d_low = df_low[0].values.flatten()  # shape = (nlat, nlon)
    freq_1d_mid = df_mid[0].values.flatten()  # shape = (nlat, nlon)

    mask_low = ~np.isnan(flat_low_1) & ~np.isnan(flat_low_2)
    mask_mid = ~np.isnan(flat_mid_1) & ~np.isnan(flat_mid_2)
    # mask_fre=np.concatenate((mask_low, mask_mid))

    filtered_low_1 = flat_low_1[mask_low]
    filtered_low_2 = flat_low_2[mask_low]
    filtered_mid_1 = flat_mid_1[mask_mid]
    filtered_mid_2 = flat_mid_2[mask_mid]

    # 注意：根据你的情况，若 data_frequency 与 da_low_1 等形状对齐，且不再做任何经纬度裁剪
    nlat, nlon = da_low_1.shape
    # freq_1d = freq_2d.flatten()
    # freq_1d_valid = freq_1d
    # # freq_1d_valid = freq_1d[mask_low]

    # 3) （可选）得到高频/低频散点的布尔掩膜
    #    你之前有在散点图上标绿色散点用这个：
    high_freq_mask_low = (freq_1d_low > 0.5)
    high_freq_mask_mid = (freq_1d_low > 0.5)

    # 4) 计算相关系数
    def calc_corr(a, b):
        if len(a) > 1 and len(b) > 1:
            return np.corrcoef(a, b)[0, 1]
        else:
            return np.nan

    correlation_coefficient_low = calc_corr(filtered_low_1, filtered_low_2)
    correlation_coefficient_mid = calc_corr(filtered_mid_1, filtered_mid_2)

    # 创建一个大小为 (7.5, 2.5) 的图形，并且创建 1 行 2 列的子图
    fig = plt.figure(figsize=(7.5, 2.5),constrained_layout=True)
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3, figure=fig)

    # -------------------- 左侧：散点图 --------------------
    gs_left = outer_gs[0, 0].subgridspec(1, 1)
    ax_scatter = fig.add_subplot(gs_left[0, 0])

    # (a) 中频散点
    if len(filtered_mid_1) > 1 and len(filtered_mid_2) > 1:
        sns.scatterplot(x=filtered_mid_2, y=filtered_mid_1, ax=ax_scatter,
                        color=color_mid, alpha=0.3, s=1)
        slope_mid, intercept_mid = np.polyfit(filtered_mid_2, filtered_mid_1, 1)
    else:
        slope_mid, intercept_mid = np.nan, np.nan
    # (b) 低频散点
    if len(filtered_low_1) > 1 and len(filtered_low_2) > 1:
        sns.scatterplot(x=filtered_low_2, y=filtered_low_1, ax=ax_scatter,
                                      color=color_low, alpha=0.3, s=1)
        # 设置坐标轴与图例
    ax_scatter.set_xlabel('Wet-day frequency')
    ax_scatter.set_ylabel(f'{da_low_1.name}')
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim([0, 8])
    ax_scatter.legend(loc='upper right', markerscale=7, frameon=False)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax_scatter, right=True, top=True)

    # (c) 对低频数据增加范围筛选并在散点图上高亮（highlight_mask）
    highlight_mask = np.ones_like(filtered_low_1, dtype=bool)
    if da_low_1.name in limited_key:
        y_min, y_max = limited_key[da_low_1.name]
        highlight_mask &= (filtered_low_1 >= y_min) & (filtered_low_1 <= y_max)
    if da_low_2.name in limited_key:
        x_min, x_max = limited_key[da_low_2.name]
        highlight_mask &= (filtered_low_2 >= x_min) & (filtered_low_2 <= x_max)

    # 添加额外的范围筛选
    range_highlight_mask = np.ones_like(filtered_low_1, dtype=bool)
    if da_low_1.name in range_filter:
        y_min, y_max = range_filter[da_low_1.name]
        range_highlight_mask &= (filtered_low_1 >= y_min) & (filtered_low_1 <= y_max)
    if da_low_2.name in range_filter:
        x_min, x_max = range_filter[da_low_2.name]
        range_highlight_mask &= (filtered_low_2 >= x_min) & (filtered_low_2 <= x_max)

    # 绘制两个矩形框，颜色与右侧地图的区域颜色对应
    # 第一个矩形框 (limit_key) - 与蓝色区域对应
    if da_low_2.name in limited_key and da_low_1.name in limited_key:
        x_min, x_max = limited_key[da_low_2.name]
        y_min, y_max = limited_key[da_low_1.name]
        rect1 = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=1.5, edgecolor=color_filter_1, facecolor='none',
                                  linestyle='-', label='Filter 1')
        ax_scatter.add_patch(rect1)

    # 第二个矩形框 (range_filter) - 与绿色区域对应
    if da_low_2.name in range_filter and da_low_1.name in range_filter:
        x_min, x_max = range_filter[da_low_2.name]
        y_min, y_max = range_filter[da_low_1.name]
        rect2 = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=1.5, edgecolor=color_filter_2, facecolor='none',
                                  linestyle='-', label='Filter 2')
        ax_scatter.add_patch(rect2)

    # --- 在 ax_scatter 上加上 mean curve & 转折点 --- #

    # 低纬度 mean 拟合
    # 1) 定义更细的频率 bins
    bins_low = np.linspace(filtered_low_2.min(), filtered_low_2.max(), 10)
    # 2) 计算每个点所属 bin
    inds_low = np.digitize(filtered_low_2, bins_low)
    # 3) 计算 bin 中心和对应的 mean(y)
    bin_centers_low = 0.5 * (bins_low[1:] + bins_low[:-1])
    mean_low = np.array([
        filtered_low_1[inds_low == i].mean() if np.any(inds_low == i) else np.nan
        for i in range(1, len(bins_low))
    ])
    # # 4) 画出 mean 曲线
    # ax_scatter.plot(
    #     bin_centers_low, mean_low,
    #     color=color_low,
    #     linewidth=1,
    #     marker='o',         # 实心圆点
    #     markersize=2,
    #     linestyle='-',      # 实线
    #     label='Low-lat Avg'
    # )

    # 中纬度 mean 拟合
    bins_mid = np.linspace(filtered_mid_2.min(), filtered_mid_2.max(), 10)
    inds_mid = np.digitize(filtered_mid_2, bins_mid)
    bin_centers_mid = 0.5 * (bins_mid[1:] + bins_mid[:-1])
    mean_mid = np.array([
        filtered_mid_1[inds_mid == i].mean() if np.any(inds_mid == i) else np.nan
        for i in range(1, len(bins_mid))
    ])
    # ax_scatter.plot(bin_centers_mid, mean_mid,
    #                 color=color_mid, linewidth=1,marker='o',markersize=2, label='Mid-lat Avg')
    # deriv_mid = np.gradient(mean_mid, bin_centers_mid)
    # idx_tp_mid = np.nanargmin(np.abs(deriv_mid))

    # ——[新增] 计算密度主干区域的转折点并绘制竖线 —— #
    # ——[在计算 mean_low 之后，绘制 mean 曲线前]——
    # Bootstrap 参数
    n_boot = 1000

    # 准备存放置信区间的数组
    ci_lower_low = np.full_like(mean_low, np.nan)
    ci_upper_low = np.full_like(mean_low, np.nan)

    for i in range(len(bin_centers_low)):
        # bin_inds 对应第 i 个 bin 的所有样本
        bin_inds = np.where(inds_low == i + 1)[0]
        data_bin = filtered_low_1[bin_inds]
        if data_bin.size > 0:
            # 每次对 data_bin 做一次有放回抽样，计算均值
            boot_means = np.random.choice(data_bin,
                                          size=(n_boot, data_bin.size),
                                          replace=True).mean(axis=1)
            # 取 2.5% 和 97.5% 分位数
            ci_lower_low[i], ci_upper_low[i] = np.percentile(boot_means, [2.5, 97.5])

    # 绘制带误差棒的 mean 曲线
    ax_scatter.errorbar(
        bin_centers_low, mean_low,
        yerr=[mean_low - ci_lower_low, ci_upper_low - mean_low],
        fmt='o-',  # 实线 + 实心圆点
        markersize=2,
        capsize=3,  # "工"形帽宽度
        linewidth=1,
        label='Low-lat Avg'
    )

    # （可选）如果想要填充误差带：
    ax_scatter.fill_between(
        bin_centers_low,
        ci_lower_low,
        ci_upper_low,
        alpha=0.2
    )

    # ——[对中纬度同理，计算并绘制误差棒]——
    ci_lower_mid = np.full_like(mean_mid, np.nan)
    ci_upper_mid = np.full_like(mean_mid, np.nan)

    for i in range(len(bin_centers_mid)):
        bin_inds = np.where(inds_mid == i + 1)[0]
        data_bin = filtered_mid_1[bin_inds]
        if data_bin.size > 0:
            boot_means = np.random.choice(data_bin,
                                          size=(n_boot, data_bin.size),
                                          replace=True).mean(axis=1)
            ci_lower_mid[i], ci_upper_mid[i] = np.percentile(boot_means, [2.5, 97.5])

    ax_scatter.errorbar(
        bin_centers_mid, mean_mid,
        yerr=[mean_mid - ci_lower_mid, ci_upper_mid - mean_mid],
        fmt='o-',
        markersize=2,
        capsize=3,
        linewidth=1,
        label='Mid-lat Avg'
    )
    # 1. 计算频率（x 轴）方向上的核密度估计
    from scipy.stats import gaussian_kde

    # 1. 构造二维点集
    points = np.vstack([filtered_low_2, filtered_low_1])  # x = wet-day freq, y = variable

    # 2. 二维核密度估计
    kde = gaussian_kde(points)
    dens = kde(points)  # 每个散点的密度

    # 3. 获取前 90% 的高密度散点索引（主干区域）
    density_threshold = np.quantile(dens, 0.50)  # 注意：值越小表示密度越低
    main_mask_scatter = dens > density_threshold  # True 表示高密度主干区域

    # 4. 用主干区域的点重新计算 mean 曲线（沿 x 分 bin）
    x_main = filtered_low_2[main_mask_scatter]
    y_main = filtered_low_1[main_mask_scatter]

    bins_main = np.linspace(x_main.min(), x_main.max(), 10)
    inds_main = np.digitize(x_main, bins_main)
    bin_centers_main = 0.5 * (bins_main[1:] + bins_main[:-1])
    mean_main = np.array([
        y_main[inds_main == i].mean() if np.any(inds_main == i) else np.nan
        for i in range(1, len(bins_main))
    ])


    # 5. 求导并找转折点
    deriv_main = np.gradient(mean_main, bin_centers_main)
    # 构造中间区域 mask（排除边界）
    valid_idx = (np.arange(len(bin_centers_main)) > 0) & (np.arange(len(bin_centers_main)) < len(bin_centers_main) - 1)
    idx_main = np.nanargmin(np.abs(deriv_main[valid_idx]))
    x_turn = bin_centers_main[idx_main]

    # 6. 添加竖直虚线
    ax_scatter.axvline(x_turn, linestyle='--', color='grey', linewidth=1.2)

    # # 7. 临时：高亮主干区域的散点（紫色）
    # sns.scatterplot(x=x_main, y=y_main, ax=ax_scatter,
    #                 color='#155A8A', alpha=0.4, s=4, label='Main dense region')
    #


    # 更新图例
    handles, labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(handles=handles, labels=labels, loc='upper right', markerscale=2, frameon=False)

    # -------------------- 右侧：地图 --------------------
    ax_map = fig.add_subplot(outer_gs[0, 1], projection=projection(central_longitude=0))
    ax_map.coastlines()
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    gl = ax_map.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.bottom_labels = False
    gl.left_labels = False
    ax_map.set_global()

    # [A] 构造 data_frequency > 0.5 的 2D mask (红色区域)
    mask_freq_2d = (data_frequency > 0.5)  # shape=(nlat,nlon)

    # [B] 构造 highlight_mask 对应的 2D mask (蓝色区域)
    mask_highlight_2d = np.zeros_like(da_low_1, dtype=bool)
    valid_indices = np.where(mask_low)[0]  # flat 索引数组，对应 filtered_low_1/2
    for flat_idx in valid_indices:
        i = flat_idx // nlon
        j = flat_idx % nlon
        # 计算这个 flat_idx 在 filtered_low_1/2 里的相对位置
        pos_in_filtered = np.sum(valid_indices <= flat_idx) - 1
        if highlight_mask[pos_in_filtered]:
            mask_highlight_2d[i, j] = True

    # [C] 构造 range_highlight_mask 对应的 2D mask (额外的筛选条件)
    mask_range_2d = np.zeros_like(da_low_1, dtype=bool)
    for flat_idx in valid_indices:
        i = flat_idx // nlon
        j = flat_idx % nlon
        pos_in_filtered = np.sum(valid_indices <= flat_idx) - 1
        if range_highlight_mask[pos_in_filtered]:
            mask_range_2d[i, j] = True

    # [D] 构造两个筛选条件的交集 (用于绿色区域)
    # 高频区域与两个矩形的交集
    mask_freq_2d = mask_freq_2d.sel(latitude=slice(30, -30))
    mask_combined = xr.DataArray(mask_range_2d,
                                 coords=mask_freq_2d.coords,  # 继承原 DataArray 的坐标
                                 dims=mask_freq_2d.dims,  # 继承原 DataArray 的维度
                                 attrs=mask_freq_2d.attrs)
    # mask_combined = mask_freq_2d & mask_range_2d

    # 准备网格坐标
    Lon, Lat = np.meshgrid(da_low_1.longitude, da_low_1.latitude)

    # [处理循环点函数] - 优化以解决白线问题
    def add_cyclic(data, lons):
        """为数据添加循环点"""
        cyclic_data, cyclic_lons = add_cyclic_point(data, coord=lons)
        return cyclic_data, cyclic_lons

    # 处理所有需要循环点的数据
    # 处理mask_highlight_2d的循环点
    cyclic_highlight, cyclic_lon = add_cyclic(mask_highlight_2d, da_low_1.longitude.values)
    cyclic_Lon, cyclic_Lat = np.meshgrid(cyclic_lon, da_low_1.latitude.values)

    # 处理mask_range_2d的循环点
    cyclic_range, _ = add_cyclic(mask_range_2d, da_low_1.longitude.values)

    # 处理mask_freq_2d的循环点
    # 根据mask_freq_2d的维度获取经纬度值
    freq_lon_values = mask_freq_2d.longitude.values
    freq_lat_values = mask_freq_2d.latitude.values
    cyclic_freq, cyclic_freq_lon = add_cyclic(mask_freq_2d.values, freq_lon_values)

    # 处理mask_combined的循环点
    # 确保mask_combined与mask_freq_2d具有相同的形状
    cyclic_combined, _ = add_cyclic(mask_combined.values, freq_lon_values)

    # 创建循环点的网格
    cyclic_freq_Lon, cyclic_freq_Lat = np.meshgrid(cyclic_freq_lon, freq_lat_values)

    # -- 在地图上画区域 B: highlight_mask 对应 (蓝色+黑色边界)
    # 使用循环点数据来绘制
    ax_map.contourf(cyclic_Lon, cyclic_Lat, cyclic_highlight,
                    levels=[0.5, 1],
                    colors=color_filter_1, alpha=1.0,
                    transform=ccrs.PlateCarree())
    ax_map.contour(cyclic_Lon, cyclic_Lat, cyclic_highlight, levels=[0.5],
                   colors='black', linewidths=1.2,
                   transform=ccrs.PlateCarree())

    # -- 在地图上画区域 A: data_frequency > 0.5 与两个筛选条件的交集 (绿色+黑色边界)
    # 使用循环点数据来绘制
    ax_map.contourf(cyclic_freq_Lon, cyclic_freq_Lat, cyclic_combined,
                    levels=[0.5, 1],
                    colors=color_filter_2, alpha=1.0,
                    transform=ccrs.PlateCarree())
    ax_map.contour(cyclic_freq_Lon, cyclic_freq_Lat, cyclic_combined, levels=[0.5],
                   colors='black', linewidths=1.2,
                   transform=ccrs.PlateCarree())

    # 图例（不需要 colorbar）dpt
    blue_patch = mpatches.Patch(color=color_filter_1, alpha=1.0, label='Filter 1')
    green_patch = mpatches.Patch(color=color_filter_2, alpha=1.0, label='Filter 2')
    ax_map.legend(
        handles=[blue_patch, green_patch],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),  # 往下偏移的距离，根据需要自行调整
        ncol=2,
        frameon=False,
        handlelength=2,  # 调整标记的长度
        handleheight=2,  # 调整标记的高度
        prop={'size': 12}  # 调整文字的大小
    )
    set_fonts_for_fig(fig,scale_factor=4,legend_scale_factor=0.8)
    for i, ax in enumerate([ax_scatter, ax_map]):
        label = f"{chr(97 + i)}"  # 按列顺序，标签'a', 'b', 'c'...
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    # 保存 + 收尾
    plt.savefig(f'{save_path}_combined.svg', bbox_inches='tight')
    plt.show()
    plt.close()

    # 返回回归线信息
    df_slope_intercept_mid = pd.DataFrame({'Slope': [slope_mid], 'Intercept': [intercept_mid]},
                                          index=[f"{da_mid_1.name}_{da_mid_2.name}"])
    return df_slope_intercept_mid


def scatter_plots_combined_with_map(das_low, das_mid, data_frequency, save_path, cmap=None,
                                    projection=ccrs.Mollweide, limited_key=None):
    if limited_key is None:
        limited_key = {}

    da_low_1, da_low_2 = das_low
    da_mid_1, da_mid_2 = das_mid

    # 检查数据形状一致性
    if da_low_1.shape != da_low_2.shape or da_mid_1.shape != da_mid_2.shape:
        raise ValueError("Data arrays must have the same shape.")

    # 将数据flatten
    flat_low_1 = da_low_1.values.flatten()
    print(da_low_1.name)
    print(da_low_2.name)
    flat_low_2 = da_low_2.values.flatten()
    flat_mid_1 = da_mid_1.values.flatten()
    flat_mid_2 = da_mid_2.values.flatten()

    mask_low = ~np.isnan(flat_low_1) & ~np.isnan(flat_low_2)
    mask_mid = ~np.isnan(flat_mid_1) & ~np.isnan(flat_mid_2)
    # mask_blue = (flat_low_1 > 2) & (flat_low_1 < 3) & (flat_low_2 > 0.4) & (flat_low_2 < 0.6)
    # mask_blue=mask_blue[mask_low]

    filtered_low_1 = flat_low_1[mask_low]
    filtered_low_2 = flat_low_2[mask_low]
    filtered_mid_1 = flat_mid_1[mask_mid]
    filtered_mid_2 = flat_mid_2[mask_mid]

    # 新增：过滤对应的 data_frequency
    filtered_data_frequency = data_frequency.sel(latitude=slice(30, -30)).values.flatten()[mask_low]
    high_freq_mask = filtered_data_frequency > 0.8
    low_freq_mask = filtered_data_frequency < 0.2

    # 计算相关系数
    def calc_corr(a, b):
        if len(a) > 1 and len(b) > 1:
            return np.corrcoef(a, b)[0, 1]
        else:
            return np.nan

    correlation_coefficient_low = calc_corr(filtered_low_1, filtered_low_2)
    correlation_coefficient_mid = calc_corr(filtered_mid_1, filtered_mid_2)

    # 设置Nature风格
    sns.set(style="ticks", context="talk", palette="colorblind")
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 16,
    })

    # 创建画布：左侧散点图，右侧地理图
    fig = plt.figure(figsize=(14, 7), dpi=100)
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[0.6, 1], wspace=0.3, figure=fig)

    # 左侧散点图
    gs_left = outer_gs[0, 0].subgridspec(1, 1)
    ax_scatter = fig.add_subplot(gs_left[0, 0])

    # 绘制中频散点及回归线
    if len(filtered_mid_1) > 1 and len(filtered_mid_2) > 1:
        sns.scatterplot(x=filtered_mid_2, y=filtered_mid_1, ax=ax_scatter,
                        color='red', alpha=0.4, s=2, label='Mid data')
        slope_mid, intercept_mid = np.polyfit(filtered_mid_2, filtered_mid_1, 1)
        fit_line_mid = np.polyval([slope_mid, intercept_mid], filtered_mid_2)
        ax_scatter.axhline(y=2, color='grey', linestyle='--', label='Horizontal line y=2')
    else:
        slope_mid, intercept_mid = np.nan, np.nan

    # 绘制低频散点（不再绘制回归线）
    if len(filtered_low_1) > 1 and len(filtered_low_2) > 1:
        # 绘制所有低频数据的散点，颜色为蓝色
        sns.scatterplot(x=filtered_low_2[low_freq_mask], y=filtered_low_1[low_freq_mask], ax=ax_scatter,
                        color='blue', alpha=0.4, s=2, label='Low data')
        sns.scatterplot(x=filtered_low_2[high_freq_mask], y=filtered_low_1[high_freq_mask], ax=ax_scatter,
                        color='green', alpha=1.0, s=2, label='Low data')

    # 修改轴标题和范围
    ax_scatter.set_xlabel('Wet-day Frequency', fontsize=16)
    ax_scatter.set_ylabel(f'{da_low_1.name} (day)', fontsize=16)
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim([0, 8])
    ax_scatter.legend(loc='upper right', markerscale=2, frameon=False)

    # 添加网格线
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    # 去掉右边和上边的轴
    sns.despine(ax=ax_scatter, right=True, top=True)

    # 对低频数据增加范围筛选并在散点图上高亮
    highlight_mask = np.ones_like(filtered_low_1, dtype=bool)
    if da_low_1.name in limited_key:
        y_min, y_max = limited_key[da_low_1.name]
        highlight_mask &= (filtered_low_1 >= y_min) & (filtered_low_1 <= y_max)
    if da_low_2.name in limited_key:
        x_min, x_max = limited_key[da_low_2.name]
        highlight_mask &= (filtered_low_2 >= x_min) & (filtered_low_2 <= x_max)

    if highlight_mask.any():
        sns.scatterplot(x=filtered_low_2[highlight_mask], y=filtered_low_1[highlight_mask],
                        ax=ax_scatter, color='green', alpha=1.0, s=2, label='Selected Range')
        ax_scatter.legend(loc='upper right', markerscale=2, frameon=False)

    # 右侧地理图
    ax_map = fig.add_subplot(outer_gs[0, 1], projection=projection(central_longitude=0))
    ax_map.coastlines()
    ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax_map.top_labels = False
    ax_map.right_labels = False
    ax_map.bottom_labels = False
    ax_map.left_labels = True
    ax_map.set_global()
    ax_map.set_xlabel('Longitude', fontsize=14, labelpad=10)
    ax_map.set_ylabel('Latitude', fontsize=14, labelpad=10)

    all_area_num = 20
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, all_area_num)

    infile = data_frequency.squeeze().sel(latitude=slice(60, -60))
    infile = infile.assign_coords(longitude=infile.longitude - 180)
    infile = infile.sortby('longitude')
    tp, longitude = add_cyclic_point(infile, infile.longitude)

    # 采用自然风格的配色
    cmap = sns.color_palette("viridis", as_cmap=True) if cmap is None else cmap
    masked_data = np.isnan(tp)
    cmap_masked = ListedColormap(['none', 'lightgray'])

    contour = ax_map.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap,
                              vmin=vbins[0], vmax=vbins[-1], transform=ccrs.PlateCarree())

    # 添加色条
    cbar = plt.colorbar(contour, ax=ax_map, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(f'{data_frequency.name}', fontsize=14)

    lat_vals = da_low_1.latitude.values
    lon_vals = da_low_1.longitude.values
    Lon, Lat = np.meshgrid(lon_vals, lat_vals)
    flattened_Lat = Lat.flatten()[mask_low]
    flattened_Lon = Lon.flatten()[mask_low]

    if highlight_mask.any():
        highlight_lats = flattened_Lat[highlight_mask]
        highlight_lons = flattened_Lon[highlight_mask]
        ax_map.scatter(highlight_lons, highlight_lats, s=2, c='green',
                       transform=ccrs.PlateCarree(), label='Selected Range Locations')
        ax_map.legend(loc='lower left', frameon=False, markerscale=2)

    plt.savefig(f'{save_path}_combined.png', bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()

    df_slope_intercept_mid = pd.DataFrame({'Slope': [slope_mid], 'Intercept': [intercept_mid]},
                                          index=[f"{da_mid_1.name}_{da_mid_2.name}"])
    return df_slope_intercept_mid


@auto_close_plot
def multi_scatter_plot(wet_data_list_low, duration_list_low,
                       wet_data_list_mid, duration_list_mid,
                       nrows, ncols, titles, save_path=None):
    """绘制多子图版本的散点图"""
    # 保持风格
    # 保持风格时设置全局字体
    sns.set_theme(
        style="whitegrid",
        font="sans-serif",
        rc={
            "font.size": 18,  # 基础字体大小
            "axes.titlesize": 18,  # 子图标题
            "axes.labelsize": 18,  # 轴标签
            "xtick.labelsize": 15,  # X轴刻度
            "ytick.labelsize": 15,  # Y轴刻度
            "legend.fontsize": 12,  # 图例字体
        }
    )

    # 创建大图
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 15))
    axes = axes.flatten(order='F')  # 将二维轴数组转换为一维

    # 循环绘制每个子图
    for i, ax in enumerate(axes):
        if i >= len(wet_data_list_low):  # 处理多余的子图
            ax.axis('off')
            continue

        # --- 提取当前阈值的数据 ---
        da_low_x = wet_data_list_low[i]
        da_low_y = duration_list_low[i]
        da_mid_x = wet_data_list_mid[i]
        da_mid_y = duration_list_mid[i]

        # 展平并去除缺失值
        flat_low_x = da_low_x.values.flatten()
        flat_low_y = da_low_y.values.flatten()
        flat_mid_x = da_mid_x.values.flatten()
        flat_mid_y = da_mid_y.values.flatten()

        mask_low = ~np.isnan(flat_low_x) & ~np.isnan(flat_low_y)
        mask_mid = ~np.isnan(flat_mid_x) & ~np.isnan(flat_mid_y)

        filtered_low_x = flat_low_x[mask_low]
        filtered_low_y = flat_low_y[mask_low]
        filtered_mid_x = flat_mid_x[mask_mid]
        filtered_mid_y = flat_mid_y[mask_mid]

        # --- 数据预处理（与原始函数相同）---
        # ... [此处插入你原有的数据预处理代码] ...

        # --- 绘制散点 ---
        sns.scatterplot(x=filtered_low_x, y=filtered_low_y, ax=ax,
                        color='#1f78b4', alpha=0.4, s=1, marker='o')
        sns.scatterplot(x=filtered_mid_x, y=filtered_mid_y, ax=ax,
                        color='#e66101', alpha=0.4, s=1, marker='o')
        # 低纬度 mean 拟合
        # 1) 定义更细的频率 bins
        bins_low = np.linspace(filtered_low_x.min(), filtered_low_x.max(), 10)
        # 2) 计算每个点所属 bin
        inds_low = np.digitize(filtered_low_x, bins_low)
        # 3) 计算 bin 中心和对应的 mean(y)
        bin_centers_low = 0.5 * (bins_low[1:] + bins_low[:-1])
        mean_low = np.array([
            filtered_low_y[inds_low == i].mean() if np.any(inds_low == i) else np.nan
            for i in range(1, len(bins_low))
        ])

        # 中纬度 mean 拟合
        bins_mid = np.linspace(filtered_mid_x.min(), filtered_mid_x.max(), 10)
        inds_mid = np.digitize(filtered_mid_x, bins_mid)
        bin_centers_mid = 0.5 * (bins_mid[1:] + bins_mid[:-1])
        mean_mid = np.array([
            filtered_mid_y[inds_mid == i].mean() if np.any(inds_mid == i) else np.nan
            for i in range(1, len(bins_mid))
        ])

        n_boot = 1000

        # 准备存放置信区间的数组
        ci_lower_low = np.full_like(mean_low, np.nan)
        ci_upper_low = np.full_like(mean_low, np.nan)

        for k in range(len(bin_centers_low)):
            # bin_inds 对应第 i 个 bin 的所有样本
            bin_inds = np.where(inds_low == k + 1)[0]
            data_bin = filtered_low_y[bin_inds]
            if data_bin.size > 0:
                # 每次对 data_bin 做一次有放回抽样，计算均值
                boot_means = np.random.choice(data_bin,
                                              size=(n_boot, data_bin.size),
                                              replace=True).mean(axis=1)
                # 取 2.5% 和 97.5% 分位数
                ci_lower_low[k], ci_upper_low[k] = np.percentile(boot_means, [2.5, 97.5])

        # 绘制带误差棒的 mean 曲线
        ax.errorbar(
            bin_centers_low, mean_low,
            yerr=[mean_low - ci_lower_low, ci_upper_low - mean_low],
            fmt='o-',  # 实线 + 实心圆点
            markersize=2,
            capsize=4,  # "工"形帽宽度
            linewidth=1,
            label='Low-lat Avg'
        )

        # （可选）如果想要填充误差带：
        ax.fill_between(
            bin_centers_low,
            ci_lower_low,
            ci_upper_low,
            alpha=0.2
        )

        # ——[对中纬度同理，计算并绘制误差棒]——
        ci_lower_mid = np.full_like(mean_mid, np.nan)
        ci_upper_mid = np.full_like(mean_mid, np.nan)

        for k in range(len(bin_centers_mid)):
            bin_inds = np.where(inds_mid == k + 1)[0]
            data_bin = filtered_mid_y[bin_inds]
            if data_bin.size > 0:
                boot_means = np.random.choice(data_bin,
                                              size=(n_boot, data_bin.size),
                                              replace=True).mean(axis=1)
                ci_lower_mid[k], ci_upper_mid[k] = np.percentile(boot_means, [2.5, 97.5])

        ax.errorbar(
            bin_centers_mid, mean_mid,
            yerr=[mean_mid - ci_lower_mid, ci_upper_mid - mean_mid],
            fmt='o-',
            markersize=2,
            capsize=4,
            linewidth=1,
            label='Mid-lat Avg'
        )
        # 1. 计算频率（x 轴）方向上的核密度估计
        from scipy.stats import gaussian_kde

        # 1. 构造二维点集
        points = np.vstack([filtered_low_x, filtered_low_y])  # x = wet-day freq, y = variable

        # 2. 二维核密度估计
        kde = gaussian_kde(points)
        dens = kde(points)  # 每个散点的密度

        # --- 计算多个密度百分位对应的转折点列表 ---
        density_pcts = np.linspace(0.3, 0.7, 20)  # 30%, 32.5%, …, 50%
        x_turns = []
        for pct in density_pcts:
            thresh = np.quantile(dens, pct)
            mask = dens > thresh
            x_main = filtered_low_x[mask]
            y_main = filtered_low_y[mask]
            if x_main.size < 3:
                x_turns.append(np.nan)
                continue
            bins_tmp = np.linspace(0.35, x_main.max(), 20)
            inds_tmp = np.digitize(x_main, bins_tmp)
            centers = 0.5 * (bins_tmp[1:] + bins_tmp[:-1])
            mean_tmp = np.array([y_main[inds_tmp == i].mean() if np.any(inds_tmp == i) else np.nan
                                 for i in range(1, len(bins_tmp))])
            deriv = np.gradient(mean_tmp, centers)
            valid = (np.arange(len(centers)) > 0) & (np.arange(len(centers)) < len(centers) - 1)
            idx = np.nanargmin(np.abs(deriv[valid]))
            x_turns.append(centers[valid][idx])

        # --- 统计误差棒 ---
        x_arr = np.array(x_turns)
        x_arr = x_arr[~np.isnan(x_arr)]
        mean_turn = x_arr.mean()
        low_ci = np.percentile(x_arr, 2.5)
        high_ci = np.percentile(x_arr, 97.5)

        # --- 在散点图上添加水平误差棒和一条竖线 ---
        # 竖线：位置为 mean_turn
        ax.axvline(mean_turn, linestyle='--', color='grey', linewidth=1.2)


        # --- 坐标轴设置 ---
        ax.set_title(titles[i], pad=10)  # 子图标题
        ax.set_xlim(0, None)
        # ax.set_ylim(0, None)
        ax.legend(
            markerscale=2,  # 放大标记符号
            loc='upper right',  # 指定图例位置
        )

        # 为每个子图添加标记（a.、b.、c.……）
        label = f"{chr(97 + i)}"  # 97为'a'的ASCII码
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top')

        # 将y轴刻度标签格式化为两位小数
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # 仅最下方子图显示x轴标签：检查子图所在的行是否为最后一行
        # 仅最下方子图显示 x 轴标签：检查行号是否为 nrows-1
        if i % nrows != (nrows - 1):
            ax.set_xlabel('')
        else:
            ax.set_xlabel(da_low_x.name if da_low_x.name else "X")

        # 仅最左侧子图显示 y 轴标签：检查列号是否为 0
        if i // nrows != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel("Duration (day)")

    # --- 全局设置 ---
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # 为总标题留空间

    # --- 保存或展示 ---
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


@auto_close_plot
def single_scatter_plot(da_low_x, da_low_y, da_mid_x, da_mid_y, save_path=None):
    # --- 0. 前期准备 ---
    # 保持之前的风格
    sns.set_theme(style="whitegrid", font="sans-serif")

    color_low = '#1f78b4'  # 深蓝
    color_mid = '#e66101'  # 深橙

    # 展平并去除缺失值
    flat_low_x = da_low_x.values.flatten()
    flat_low_y = da_low_y.values.flatten()
    flat_mid_x = da_mid_x.values.flatten()
    flat_mid_y = da_mid_y.values.flatten()

    mask_low = ~np.isnan(flat_low_x) & ~np.isnan(flat_low_y)
    mask_mid = ~np.isnan(flat_mid_x) & ~np.isnan(flat_mid_y)

    filtered_low_x = flat_low_x[mask_low]
    filtered_low_y = flat_low_y[mask_low]
    filtered_mid_x = flat_mid_x[mask_mid]
    filtered_mid_y = flat_mid_y[mask_mid]

    fig, ax = plt.subplots(figsize=(6, 6))

    # --- 1. 绘制散点 ---
    sns.scatterplot(x=filtered_low_x, y=filtered_low_y, ax=ax,
                    color=color_low, alpha=0.6, s=1, marker='o',
                    label='Low lat')
    sns.scatterplot(x=filtered_mid_x, y=filtered_mid_y, ax=ax,
                    color=color_mid, alpha=0.6, s=1, marker='o',
                    label='Mid lat')

    # --- 2. 根据需要，是否绘制回归线 ---
    # 这里演示：如果 x 或 y 的变量名有 "power"，则只对低纬度做线性拟合；否则对中纬度
    var_x_name = da_low_x.name.lower() if da_low_x.name else ""
    var_y_name = da_low_y.name.lower() if da_low_y.name else ""
    if "power" in var_x_name or "power" in var_y_name:
        # 对低纬度做线性拟合
        if len(filtered_low_x) > 1 and len(filtered_low_y) > 1:
            slope_low, intercept_low = np.polyfit(filtered_low_x, filtered_low_y, 1)
            fit_low = np.polyval([slope_low, intercept_low], filtered_low_x)
            ax.plot(filtered_low_x, fit_low, color=color_low, linestyle='-', linewidth=2)
    else:
        # 对中纬度做线性拟合
        if len(filtered_mid_x) > 1 and len(filtered_mid_y) > 1:
            slope_mid, intercept_mid = np.polyfit(filtered_mid_x, filtered_mid_y, 1)
            fit_mid = np.polyval([slope_mid, intercept_mid], filtered_mid_x)
            ax.plot(filtered_mid_x, fit_mid, color=color_mid, linestyle='-', linewidth=2)

    # --- 3. 坐标轴和标签 ---
    ax.set_xlabel(da_low_x.name if da_low_x.name else "X", fontsize=12)
    ax.set_ylabel("Mean event duration", fontsize=12)  # 强制设置Y轴标签
    ax.legend(loc='best', frameon=False, markerscale=10, )

    # --- 新增：设置标题 ---
    def extract_threshold(da):
        """从DataArray名称中提取top+数字格式的阈值"""
        if da.name:
            match = re.search(r'top(\d+)', da.name, re.IGNORECASE)
            if match:
                return f"Top {match.group(1)}% Threshold"
        return None

    # 优先使用X轴的名称提取阈值
    title = extract_threshold(da_low_x) or extract_threshold(da_low_y)
    if title:
        plt.title(f"{title}", fontsize=14, pad=20)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    set_fonts_for_fig(fig)

    plt.tight_layout()

    # --- 4. 保存或展示 ---
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


@auto_close_plot
def multi_threshold_scatter_plot(list_low_x, list_low_y,
                                 list_mid_x, list_mid_y,
                                 threshold_list,
                                 save_path=None):
    """
    在同一个图中，绘制多组阈值 (threshold_list) 对应的
    [低纬度: (x, y), 中纬度: (x, y)] 的散点图并可选线性拟合。

    参数
    ----
    list_low_x, list_low_y : list of xarray.DataArray
        低纬度数据的 x / y 列表，每个元素对应一个阈值。
    list_mid_x, list_mid_y : list of xarray.DataArray
        中纬度数据的 x / y 列表，每个元素对应一个阈值。
    threshold_list : list
        阈值列表，如 [10, 20, 30]。
    save_path : str or None
        若提供，则保存图像；否则 plt.show()。
    """

    # --- 0. 前期准备 ---
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 为了示例，低纬度固定一种颜色，中纬度固定另一种颜色
    color_low = '#1f78b4'  # 深蓝
    color_mid = '#e66101'  # 深橙

    # 假设大家的变量名都一致，这里仅演示取第一个来设置坐标名字
    # 如果每个阈值下可能对应不同的变量名，可自行再做处理
    var_x_name_low = list_low_x[0].name if list_low_x[0].name else "X"
    var_y_name_low = list_low_y[0].name if list_low_y[0].name else "Y"

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    # --- 1. 循环绘制多组阈值 ---
    for i, thr in enumerate(threshold_list):
        # 1.1 获取当前阈值下的数据并展平
        da_low_x = list_low_x[i].values.flatten()
        da_low_y = list_low_y[i].values.flatten()
        da_mid_x = list_mid_x[i].values.flatten()
        da_mid_y = list_mid_y[i].values.flatten()

        # 1.2 去缺失值
        mask_low = ~np.isnan(da_low_x) & ~np.isnan(da_low_y)
        mask_mid = ~np.isnan(da_mid_x) & ~np.isnan(da_mid_y)

        x_low_filtered = da_low_x[mask_low]
        y_low_filtered = da_low_y[mask_low]
        x_mid_filtered = da_mid_x[mask_mid]
        y_mid_filtered = da_mid_y[mask_mid]

        # 1.3 散点图
        sns.scatterplot(
            x=x_low_filtered,
            y=y_low_filtered,
            ax=ax,
            color=color_low,
            alpha=0.6,
            s=2,
            marker='o',
            label=f'Low lat - {thr}%'
        )
        sns.scatterplot(
            x=x_mid_filtered,
            y=y_mid_filtered,
            ax=ax,
            color=color_mid,
            alpha=0.6,
            s=2,
            marker='o',
            label=f'Mid lat - {thr}%'
        )

        # 1.4 可选线性拟合逻辑（与 single_scatter_plot 类似）
        # 这里演示：如果变量名里有 "power"，则对低纬度做线性拟合，否则对中纬度。
        # 也可直接写死：每一组都做拟合，或都不做，灵活调整。
        var_x_low_name = list_low_x[i].name.lower() if list_low_x[i].name else ""
        var_y_low_name = list_low_y[i].name.lower() if list_low_y[i].name else ""
        if "power" in var_x_low_name or "power" in var_y_low_name:
            if len(x_low_filtered) > 1 and len(y_low_filtered) > 1:
                slope_low, intercept_low = np.polyfit(x_low_filtered, y_low_filtered, 1)
                fit_low = np.polyval([slope_low, intercept_low], x_low_filtered)
                ax.plot(x_low_filtered, fit_low, color=color_low, linestyle='-')
        else:
            if len(x_mid_filtered) > 1 and len(y_mid_filtered) > 1:
                slope_mid, intercept_mid = np.polyfit(x_mid_filtered, y_mid_filtered, 1)
                fit_mid = np.polyval([slope_mid, intercept_mid], x_mid_filtered)
                ax.plot(x_mid_filtered, fit_mid, color=color_mid, linestyle='-')

    # --- 2. 坐标轴和标签 ---
    ax.set_xlabel(var_x_name_low, fontsize=12)
    ax.set_ylabel(var_y_name_low, fontsize=12)
    ax.legend(loc='best', frameon=False)

    # 如果变量名里包含 'power'，可以用科学计数法
    # 这里以第一个 low_x 名字简单判断，也可自己按需写
    if "power" in var_x_name_low.lower() or "power" in var_y_name_low.lower():
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    # 若需要让坐标从0开始，可手动设置
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # --- 3. 保存或展示 ---
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    else:
        plt.show()


@auto_close_plot
def scatter_plots(data_arrays, save_path):
    if len(data_arrays) == 0:
        raise ValueError("Data array list cannot be empty.")

    fig_size = len(data_arrays) * 6
    fig, axes = plt.subplots(len(data_arrays), len(data_arrays), figsize=(fig_size, fig_size))
    slopes_intercepts = np.empty((len(data_arrays), len(data_arrays)), dtype=object)

    for i in range(len(data_arrays)):
        for j in range(len(data_arrays)):
            ax = axes[i, j]

            da_1 = data_arrays[i]
            da_2 = data_arrays[j]

            if da_1.shape != da_2.shape:
                raise ValueError("Data arrays must have the same shape.")

            flat_array_1 = da_1.values.flatten()
            flat_array_2 = da_2.values.flatten()

            mask = ~np.isnan(flat_array_1) & ~np.isnan(flat_array_2)
            filtered_array_1 = flat_array_1[mask]
            filtered_array_2 = flat_array_2[mask]

            correlation_coefficient = np.corrcoef(filtered_array_1, filtered_array_2)[0, 1]

            if len(filtered_array_1) > 1 and len(filtered_array_2) > 1:
                slope, intercept = np.polyfit(filtered_array_2, filtered_array_1, 1)
                fit_line = np.polyval([slope, intercept], filtered_array_2)
                ax.plot(filtered_array_2, fit_line, color='blue', linestyle='--')

                slopes_intercepts[i, j] = (slope, intercept)

                sns.scatterplot(x=filtered_array_2, y=filtered_array_1, ax=ax, color='blue', alpha=0.5, s=2, label=f'k:{slope:.2f}')
            else:
                slopes_intercepts[i, j] = (np.nan, np.nan)

            if i == len(data_arrays) - 1:
                ax.set_xlabel(da_2.name, fontsize=32)
            if j == 0:
                ax.set_ylabel(da_1.name, fontsize=32)

            ax.set_title(f'r={correlation_coefficient:.2f}', fontsize=24)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', markerscale=10)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    adjust_all_font_sizes(fig, scale_factor=1.2, legend_scale_factor=0.5, label_scale_factor=1, title_scale_factor=1)
    plt.savefig(f'{save_path}_correlation')
    plt.show()

    return slopes_intercepts


@auto_close_plot
def scatter_plots_depart(das_low, das_mid, save_path, limited_key=None):
    if limited_key is None:
        limited_key = {}
    if len(das_low) != len(das_mid):
        raise ValueError("The number of data arrays and variable names must be the same.")

    def calculate_lat_weights(lat):
        """计算纬度权重"""
        return np.cos(np.deg2rad(lat))

    color_low = '#1f78b4'  # 深蓝
    color_mid = '#e66101'  # 深橙

    sns.set_theme(style="whitegrid", font="sans-serif")

    n = len(das_low)
    fig_size = n * 4  # 适当调整图尺寸
    fig, axes = plt.subplots(n, n, figsize=(fig_size, fig_size), sharex=False, sharey=False, dpi=100)
    slope_intercept_mid = {}

    # === 1. 预先计算每个变量的全局最小值、最大值（低纬和中纬合并后） ===
    var_minmax = {}
    for da_low, da_mid in zip(das_low, das_mid):
        var_name = da_low.name  # 假设 da_low.name 与 da_mid.name 一致
        # 合并 low & mid
        combined = np.concatenate([
            da_low.values.flatten(),
            da_mid.values.flatten()
        ])
        combined = combined[~np.isnan(combined)]  # 去除 NaN

        # 得到全局最小/最大
        if len(combined) > 0:
            min_val, max_val = np.min(combined), np.max(combined)
        else:
            min_val, max_val = 0, 1  # 若全 NaN，可自行设置默认范围

        # 例如扩展 5%
        margin = (max_val - min_val) * 0.05
        min_val -= margin
        max_val += margin

        var_minmax[var_name] = (min_val, max_val)

    # 去除多余边框
    for ax_row in axes:
        for ax in ax_row:
            sns.despine(ax=ax)

    # 定义图例标记，仅在对角线子图中添加
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Low-lat', markerfacecolor=color_low, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Mid-lat', markerfacecolor=color_mid, markersize=10)
    ]

    # 定义自定义的刻度格式化函数：科学计数法和常规计数法，均保留1位小数
    from matplotlib.ticker import FuncFormatter
    plain_formatter = FuncFormatter(lambda x, pos: f'{x:.1f}')
    plain_formatter_p = FuncFormatter(lambda x, pos: f'{x:.2f}')

    label_index = 0
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            da_low_x = das_low[j]
            da_low_y = das_low[i]
            da_mid_x = das_mid[j]
            da_mid_y = das_mid[i]

            if da_low_x.shape != da_low_y.shape or da_mid_x.shape != da_mid_y.shape:
                raise ValueError("Data arrays must have the same shape.")

            # 计算纬度权重
            lat_weights_low = calculate_lat_weights(da_low_x.latitude.values)
            lat_weights_mid = calculate_lat_weights(da_mid_x.latitude.values)

            # 展平并去除缺失值
            flat_low_x = da_low_x.values.flatten()
            flat_low_y = da_low_y.values.flatten()
            flat_mid_x = da_mid_x.values.flatten()
            flat_mid_y = da_mid_y.values.flatten()

            # 展平权重
            flat_weights_low = np.repeat(lat_weights_low, len(da_low_x.longitude))
            flat_weights_mid = np.repeat(lat_weights_mid, len(da_mid_x.longitude))

            mask_low = ~np.isnan(flat_low_x) & ~np.isnan(flat_low_y)
            mask_mid = ~np.isnan(flat_mid_x) & ~np.isnan(flat_mid_y)

            filtered_low_x = flat_low_x[mask_low]
            filtered_low_y = flat_low_y[mask_low]
            filtered_mid_x = flat_mid_x[mask_mid]
            filtered_mid_y = flat_mid_y[mask_mid]
            filtered_weights_low = flat_weights_low[mask_low]
            filtered_weights_mid = flat_weights_mid[mask_mid]

            if i == j:
                # 对角线: 显示分布图
                combined_data = np.concatenate([filtered_low_x, filtered_mid_x])
                bins = np.linspace(np.min(combined_data), np.max(combined_data), 40)  # 增加 bins 数量

                # 使用普通的直方图，不添加权重
                sns.histplot(filtered_low_x, ax=ax, kde=True, color=color_low, alpha=0.5,
                             stat="probability", label='Low lat', bins=bins)
                sns.histplot(filtered_mid_x, ax=ax, kde=True, color=color_mid, alpha=0.5,
                             stat="probability", label='Mid lat', bins=bins)

                ax.set_xlabel(da_low_x.name, fontsize=12)
                ax.set_ylabel("Probability density", fontsize=12)

                ax.yaxis.set_major_formatter(plain_formatter_p)

                if da_low_x.name == 'Yearly spectrum':
                    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
                    offset_text1 = ax.xaxis.get_offset_text()
                    offset_text1.set_x(1.1)
                    offset_text1.set_fontsize(14)

                if j == 0:
                    ax.legend(handles=handles, loc='upper right', frameon=False, prop={'size': 15})

                ax.set_ylim(bottom=0)

                var_name_diag = da_low_x.name
                if var_name_diag in var_minmax:
                    min_val, max_val = var_minmax[var_name_diag]
                    ax.set_xlim(min_val, max_val)

            elif i < j:
                ax.axis('off')

            else:
                # 绘制散点图时不使用权重
                sns.scatterplot(x=filtered_low_x, y=filtered_low_y, ax=ax,
                                color=color_low, alpha=0.6, s=5, marker='o',
                                label='Low lat', legend=False)
                sns.scatterplot(x=filtered_mid_x, y=filtered_mid_y, ax=ax,
                                color=color_mid, alpha=0.6, s=5, marker='o',
                                label='Mid lat', legend=False)

                # 计算相关系数
                if len(filtered_low_x) > 1 and len(filtered_low_y) > 1:
                    corr_low = np.corrcoef(filtered_low_x, filtered_low_y)[0, 1]
                    corr_mid = np.corrcoef(filtered_mid_x, filtered_mid_y)[0, 1]
                    # 在右上角显示相关系数
                    ax.text(0.95, 0.95, f'r_low = {corr_low:.2f}\nr_mid = {corr_mid:.2f}',
                           transform=ax.transAxes, ha='right', va='top',
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

                if da_low_x.name == 'Yearly spectrum' or da_low_y.name == 'Yearly spectrum':
                    if len(filtered_low_x) > 1 and len(filtered_low_y) > 1:
                        # 使用加权最小二乘法计算相关性
                        slope_low, intercept_low = np.polyfit(filtered_low_x, filtered_low_y, 1, w=filtered_weights_low)
                        fit_low = np.polyval([slope_low, intercept_low], filtered_low_x)
                        ax.plot(filtered_low_x, fit_low, color=color_low, linestyle='-', linewidth=2)
                else:
                    if len(filtered_mid_x) > 1 and len(filtered_mid_y) > 1:
                        # 使用加权最小二乘法计算相关性
                        slope_mid, intercept_mid = np.polyfit(filtered_mid_x, filtered_mid_y, 1, w=filtered_weights_mid)
                        fit_mid = np.polyval([slope_mid, intercept_mid], filtered_mid_x)
                        ax.plot(filtered_mid_x, fit_mid, color=color_mid, linestyle='-', linewidth=2)
                        slope_intercept_mid[(da_mid_x.name, da_mid_y.name)] = (slope_mid, intercept_mid)

                ax.set_xlabel(da_low_x.name, fontsize=18)
                ax.set_ylabel(da_low_y.name, fontsize=18)

                if da_low_y.name == 'Yearly spectrum':
                    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
                    offset_text = ax.yaxis.get_offset_text()
                    offset_text.set_fontsize(14)
                else:
                    ax.yaxis.set_major_formatter(plain_formatter)

                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

                var_name_x = da_low_x.name
                var_name_y = da_low_y.name

                if var_name_x in var_minmax:
                    x_min_val, x_max_val = var_minmax[var_name_x]
                    ax.set_xlim([x_min_val if x_min_val > 0 else 0, x_max_val])
                if var_name_y in var_minmax:
                    y_min_val, y_max_val = var_minmax[var_name_y]
                    ax.set_ylim([y_min_val if y_min_val > 0 else 0, y_max_val])

            if i >= j:
                # 添加列标签
                label = f"{chr(97 + label_index)}"  # 按列顺序，标签'a', 'b', 'c'...
                ax.text(0.05, 0.95, label, transform=ax.transAxes,
                        fontsize=18, fontweight='bold', va='top')
                label_index += 1

    plt.tight_layout(pad=2.0)
    adjust_all_font_sizes(fig, scale_factor=1.3, legend_scale_factor=1)

    plt.savefig(f'{save_path}_correlation.png', bbox_inches='tight',dpi=300)
    plt.show()

    df_slope_intercept_mid = pd.DataFrame.from_dict(
        {(f"{key[0]}_{key[1]}"): value for key, value in slope_intercept_mid.items()},
        orient='index',
        columns=['Slope', 'Intercept']
    )
    return df_slope_intercept_mid


@auto_close_plot
def plot_interactive_contour(dataarray, bins):
    # 提取纬度、经度和值
    lat = dataarray['latitude'].values
    lon = dataarray['longitude'].values
    values = dataarray.values.squeeze()

    # 转换颜色格式为 'rgba(255, 255, 255, 1.0)' 并与位置配对
    custom_colorscale = [
        [pos, f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, {c[3]})'] for pos, c in zip(bins, colors)
    ]
    # 创建等高线图
    contour = go.Contour(
        z=values,
        x=lon,
        y=lat,
        colorscale=custom_colorscale,
        contours=dict(
            start=bins[0],
            end=bins[-1],
            size=(bins[-1] - bins[0]) / (len(bins) - 1)
        ),
        hoverinfo='x+y+z'  # 在鼠标悬停时显示坐标和值
    )

    # 创建图表布局
    layout = go.Layout(
        title='Interactive Contour Plot',
        xaxis=dict(title='Longitude'),
        yaxis=dict(title='Latitude')
    )

    # 创建图表对象
    fig = go.Figure(data=[contour], layout=layout)

    # 显示图表
    fig.show()


# @auto_close_plot
def pt(dr, th_list, save_path, start_date='2011-01-01', end_date='2012-01-01'):
    # Check the type of dr and slice data if necessary
    if isinstance(dr, xr.DataArray):
        if 'time' not in dr.dims:
            raise ValueError("The DataArray must contain a 'time' dimension.")
        if start_date and end_date:
            dr = dr.sel(time=slice(start_date, end_date))
    elif isinstance(dr, np.ndarray):
        if start_date and end_date:
            start_idx = (np.datetime64(start_date) - np.datetime64('2011-01-01')).astype(int)
            end_idx = (np.datetime64(end_date) - np.datetime64('2011-01-01')).astype(int) + 1
            dr = dr[start_idx:end_idx]
        else:
            dr = dr[0:365]
    else:
        raise TypeError("dr must be an xarray.DataArray or numpy.ndarray type")

    # Plot the time series
    plt.figure(figsize=(20, 4))
    if isinstance(dr, xr.DataArray):
        dr.plot(label='Precipitation')
    else:
        plt.plot(dr, label='Precipitation')

    # Set y-axis to logarithmic scale and define limits
    plt.yscale('log')
    plt.ylim(0.0001, 100)

    # Plot threshold lines
    for th in th_list:
        plt.axhline(y=th, color='r', linestyle='--', label=f'Threshold {th}')

    # Add labels and legend
    # Find the position of the first backslash and the dot
    start_index = save_path.find('/')
    end_index = save_path.find('.')

    # Extract the substring
    extracted_substring = save_path[start_index:end_index]
    plt.title(f'Precipitation at {extracted_substring}')
    plt.xlabel('Time')
    plt.ylabel('Precipitation')
    plt.legend()

    # Save the plot
    # 获取当前的 Figure 对象

    # 设置 DPI
    # fig.set_dpi(150)
    # plt.savefig(save_path)
    plt.show()
    print('函数画图成功')
    # plt.close(fig)


@auto_close_plot
def pt_single(th, dr, sp, label='Precipitation'):
    """
    Plot a single time series with rectangular shaded areas above a threshold, using a modern scientific style.

    Parameters:
        th (float or xarray.DataArray): Threshold value.
        dr (xarray.DataArray or numpy.ndarray): Data series.
        sp (str): File path to save the plot.
        label (str, optional): Label for the data series. Defaults to 'Precipitation'.
    """
    # Apply a clean, modern scientific style without background grids
    plt.style.use('seaborn-white')  # Clean white background without grids
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,  # Significantly increased base font size
        'axes.labelsize': 12,  # Significantly increased axes labels
        'axes.titlesize': 12,  # Significantly increased title size
        'legend.fontsize': 12,  # Significantly increased legend font size
        'lines.linewidth': 2,  # Increased line width
        'lines.markersize': 4,  # Slightly increased marker size
        'figure.figsize': (18, 4.5),
        'savefig.dpi': 150,
        'xtick.labelsize': 6,  # Significantly increased x-tick labels
        'ytick.labelsize': 6,  # Significantly increased y-tick labels
    })

    fig, ax = plt.subplots(constrained_layout=True)

    # Extract time and data based on the type of dr
    if isinstance(dr, xr.DataArray):
        df_2011 = dr.sel(time=slice('2011-01-01', '2011-12-31'))
        times = pd.to_datetime(df_2011['time'].values)
        data = df_2011.values
    elif isinstance(dr, np.ndarray):
        start_date = datetime(2011, 1, 1)
        times = [start_date + timedelta(days=i) for i in range(len(dr))]
        df_2011 = dr[:365]  # Adjust slicing as needed
        times = times[:365]
        data = df_2011
    else:
        raise TypeError("dr must be an xarray.DataArray or numpy.ndarray")

    # Plot the time series with updated color and increased line width
    ax.plot(times, data, label=label, color='#1f77b4')  # Blue color

    # Plot the threshold line with updated color and increased line width
    th_value = th.values if isinstance(th, xr.DataArray) else th
    ax.axhline(y=th_value, linestyle='--', color='#7f7f7f', linewidth=4, label=f'Threshold {th_value}')  # Gray color

    # Identify and shade regions where data is above the threshold using rectangles
    above_th = data > th_value
    start = None
    for i in range(len(above_th)):
        if above_th[i] and start is None:
            start = times[i]
        elif not above_th[i] and start is not None:
            end = times[i]
            ax.axvspan(start, end, color='#a6cee3', alpha=0.4)  # Light blue shading
            start = None
    # Handle case where the series ends while still above threshold
    if start is not None:
        ax.axvspan(start, times[-1], color='#a6cee3', alpha=0.4)  # Light blue shading

    # Set y-axis to logarithmic scale
    ax.set_yscale('log')

    # Set title and labels
    ax.set_title('Precipitation Time Series', fontsize=36)
    ax.set_xlabel('Time', fontsize=32)
    ax.set_ylabel(label, fontsize=32)

    # Set axis limits with padding to avoid clipping
    if len(times) > 0:
        padding = pd.DateOffset(days=15)
        start = times[0] - padding
        end = times[-1] + padding
        ax.set_xlim(start, end)

    ax.set_ylim(0.001, 100)

    # Configure x-axis to show only year and month
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Keep x-axis labels horizontal
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=24)

    # Add legend with increased font size
    ax.legend(loc='upper right', fontsize=28)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=24)

    # Increase the thickness of the plot borders
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # Increased spine thickness

    # Ensure layout does not cut off labels
    plt.tight_layout()

    # Save the plot
    plt.savefig(sp, bbox_inches='tight')
    plt.show()


@auto_close_plot
def pt_6_combine(onat_list, th_list, dr_list, sp):
    fig, axs = plt.subplots(6, figsize=(15, 5), constrained_layout=True)  # 创建6个共享X轴的子图

    cmap, colors = setup_colors()

    for idx, (dr, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')
        if isinstance(dr, xr.DataArray):
            # 如果dr是xarray.DataArray，使用.sel()方法
            df_2011 = dr.sel(time=slice('2011-01-01', '2014-12-31'))
        elif isinstance(dr, np.ndarray):
            # 如果dr是numpy.ndarray，使用数组切片
            df_2011 = dr[0:365]
        else:
            raise TypeError("dr 必须是 xarray.DataArray 或 numpy.ndarray 类型")

        # 在每个子图上绘制
        axs[idx].plot(df_2011, color=colors[idx * 16 + 10])
        axs[idx].axhline(y=th, color=colors[idx], linestyle='--')  # 绘制阈值线
        axs[idx].set_yscale('log')
        # 设置标题和标签
        if idx == 5:
            axs[idx].set_xlabel('Time', fontsize=28)
        else:
            axs[idx].set_xticklabels([])
        axs[idx].set_ylabel('Precipitation', fontsize=24)
        axs[idx].set_xlim(0, 365)
        axs[idx].set_ylim(0.0001, 100)
        axs[idx].legend(loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=28)
        axs[idx].tick_params(axis='both', labelsize=18)

        # 优化子图边框，去除不必要的脊线
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        axs[idx].spines['left'].set_visible(idx == 0)  # 仅第一个子图显示左脊线
        axs[idx].yaxis.set_ticks_position('left')

    # 统一X轴标签格式
    # _, _, colors = setup_colorbar(fig=fig, vbins=bins,cmap_name=cmap)
    plt.xlabel('Time', fontsize=28)
    plt.savefig(sp, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # 确保图表关闭以释放内存


@auto_close_plot
def pt_6(onat_list, th_list, dr_list, bins, sp):
    fig, axs = plt.subplots(6, figsize=(40, 30), constrained_layout=True)  # 创建6个子图
    _, _, colors = setup_colorbar(fig=fig, vbins=bins)

    for idx, (dr, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')
        if isinstance(dr, xr.DataArray):
            # 如果dr是xarray.DataArray，使用.sel()方法
            df_2011 = dr.sel(time=slice('2011-01-01', '2014-12-31'))
        elif isinstance(dr, np.ndarray):
            # 如果dr是numpy.ndarray，使用数组切片
            df_2011 = dr[0:365]
        else:
            raise TypeError("dr 必须是 xarray.DataArray 或 numpy.ndarray 类型")

        # 现在 df_2011 包含了2011年的数据

        # 在每个子图上绘制
        axs[idx].plot(df_2011, label=f'LON/LAT {onat}', color=colors[idx])
        axs[idx].axhline(y=th, color=colors[idx], linestyle='--')  # 绘制阈值线
        axs[idx].set_yscale('log')
        # 设置标题和标签
        axs[idx].set_title(f'Area:{idx + 1} Thresholds: {th:.2f}', fontsize=24)
        if idx == 5:
            axs[idx].set_xlabel('Time', fontsize=28)
        else:
            axs[idx].set_xticklabels([])
        axs[idx].set_ylabel('Precipitation', fontsize=24)
        axs[idx].set_xlim(0, 365)
        axs[idx].set_ylim(0.0001, 100)
        axs[idx].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=28)
        axs[idx].tick_params(axis='both', labelsize=18)
        # 保存图表
    plt.savefig(sp, bbox_inches='tight')


@auto_close_plot
def show_all_spectrum(dr_list, bins, sp):
    fig, axs = plt.subplots(6, figsize=(20, 30), constrained_layout=True)  # 创建6个子图
    _, _, colors = setup_colorbar(fig=fig, vbins=bins)

    for idx, drs in enumerate(dr_list):
        period, inverse_X = just_spectrum(drs)
        print(f'power{np.sum(inverse_X[0:14])}')
        axs[idx].plot(period, inverse_X, color=colors[idx])
        if idx == 5:
            axs[idx].set_xlabel('Period (year)', fontsize=28)
        axs[idx].set_ylabel('Intensity', fontsize=24)
        axs[idx].tick_params(axis='y', labelsize=10)
        axs[idx].set_xlim(0, 4)
        axs[idx].set_ylim(0, 0.004)
        axs[idx].tick_params(axis='both', labelsize=18)

    # 保存并显示图表
    plt.savefig(sp, bbox_inches='tight')


@auto_close_plot
def era5_draw_area_dataArray(dataArray, sp):
    print(f'max:{dataArray.max().values}')
    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes(projection=ccrs.PlateCarree())
    # 创建一个LogNorm实例
    norm = mcolors.LogNorm(0.01, vmax=dataArray.max().values)

    # 创建一个等高线图，使用对数刻度
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    c = ax.contourf(dataArray.longitude, dataArray.latitude, dataArray,
                    transform=ccrs.PlateCarree(), levels=20, cmap='rainbow', norm=norm)

    # 添加颜色条
    plt.colorbar(c, ax=ax, orientation='vertical', label='Log Scaled Values')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    plt.savefig(sp)


@auto_close_plot
def draw_all_era5_area(dataarray, sp):
    # 接着，筛选出2001年的数据
    dataarray_2014 = dataarray.sel(time=dataarray['time'].dt.year == 2014)
    # 现在，遍历每一天，并调用era5_draw_area_dataArray函数
    for day in dataarray_2014['time']:
        # 提取当天的数据
        one_day_data = dataarray_2014.sel(time=day)

        # 构造文件名，例如 "data_20010101"
        name = one_day_data['time'].dt.strftime('data_%Y%m%d').item()
        print(name)

        # 调用函数绘制并保存图像
        era5_draw_area_dataArray(one_day_data, f'{sp}all_area_amsr2/{name}')


@auto_close_plot
def draw_hist_data_collapse(durations, title, vbins, fig_name):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20), constrained_layout=True)
    # 将二维数组转换为一维数组
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=24)

    _, _, colors = setup_colorbar(fig=fig, vbins=vbins)

    for area, p_dur in enumerate(durations.transpose((1, 0))):

        for p, dur in enumerate(p_dur):
            bin_centers, hist = dur
            dur_mean = np.sum(bin_centers * hist)

            print(f'mean:{dur_mean}')
            x = bin_centers
            y = hist
            y[y == 0] = np.nan

            mask = (x > 5) & (x < 40) & ~np.isnan(y)
            x_lim = x[mask]
            y_lim = y[mask]

            scale = dur_mean
            xlim_scale = x_lim / scale
            # 绘制线图表示概率分布
            # axs[area].loglog(bin_centers / scale, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3))
            axs[area].loglog(bin_centers / scale, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3), label=['top40%', 'top30%', 'top20%', 'top10%'][p])
        # 设置标题和标签
        axs[area].set_title(f'Area:{area + 1}', fontsize=24)
        axs[area].set_ylabel('Probability', fontsize=24)
        axs[area].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=24)
        axs[area].tick_params(axis='both', labelsize=18)

    plt.savefig(fig_name)


@auto_close_plot
def draw_hist_dq_dataarray(durations, title, fig_name):
    # 创建目录
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20), constrained_layout=True)
    # 将二维数组转换为一维数组
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=24)

    _, _, colors = setup_colorbar(fig=fig, vbins=4)

    for area_ind, area in enumerate(durations.coords['area'].values):
        for season_ind, season in enumerate(durations.coords['season'].values):
            bin_centers, hist = durations.sel(season=season, area=slice(area, area)).values[0]
            dur_mean = np.sum(bin_centers * hist)
            # 计算二阶矩 (方差)
            variance = np.sum(hist * (bin_centers - dur_mean) ** 2)

            # 计算三阶矩 (偏度)
            skewness = np.sum(hist * (bin_centers - dur_mean) ** 3)
            print(f'mean:{dur_mean}')
            x = bin_centers
            y = hist
            y[y == 0] = np.nan

            mask = (x > 5) & (x < 40) & ~np.isnan(y)
            x_lim = x[mask]
            y_lim = y[mask]

            scale = dur_mean
            xlim_scale = x_lim / scale
            axs[area].loglog(bin_centers, hist, '*', color=colors[season_ind], label=durations.coords['season'].values[season_ind])
        axs[area].set_title(f'Area:{area + 1}', fontsize=24)
        axs[area].set_xlim(1, 1000)
        axs[area].set_ylabel('Probability', fontsize=24)
        axs[area].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=24)
        axs[area].tick_params(axis='both', labelsize=18)

    plt.savefig(fig_name)


@auto_close_plot
def draw_hist_dq(durations, title, vbins, fig_name):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20), constrained_layout=True)
    # 将二维数组转换为一维数组
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=24)
    cmap, colors = setup_colors()
    _, _, colors = setup_colorbar(fig=fig, vbins=vbins, cmap_name=cmap)
    for area, p_dur in enumerate(durations.transpose((1, 0))):

        for p, dur in enumerate(p_dur):
            bin_centers, hist = dur
            dur_mean = np.sum(bin_centers * hist)
            # 计算二阶矩 (方差)
            variance = np.sum(hist * (bin_centers - dur_mean) ** 2)

            # 计算三阶矩 (偏度)
            skewness = np.sum(hist * (bin_centers - dur_mean) ** 3)
            print(f'mean:{dur_mean}')
            x = bin_centers
            y = hist
            y[y == 0] = np.nan

            mask = (x > 5) & (x < 40) & ~np.isnan(y)
            x_lim = x[mask]
            y_lim = y[mask]

            scale = dur_mean
            xlim_scale = x_lim / scale
            # 绘制线图表示概率分布
            # axs[area].loglog(bin_centers / scale, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3))
            axs[area].loglog(bin_centers, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3), label=['top40%', 'top30%', 'top20%', 'top10%'][p])
        axs[area].set_title(f'Area:{area + 1}', fontsize=24)
        axs[area].set_ylabel('Probability', fontsize=24)
        axs[area].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=24)
        axs[area].tick_params(axis='both', labelsize=18)

    plt.savefig(fig_name)


@auto_close_plot
def plt_duration_hist(durations, title, vbins, fig_name):
    # 设置 Seaborn 风格
    sns.set(style="whitegrid")

    fig = plt.figure(figsize=(25, 10), constrained_layout=True)
    # fig.tight_layout()  # `constrained_layout=True` 已经处理了布局

    cmap, colors = setup_colors()

    # 初始化统计数据
    avg = np.zeros(len(vbins))
    std = np.zeros(len(vbins))
    med = np.zeros(len(vbins))
    covs = np.zeros(len(vbins))

    # 第一个子图：Log-Log Histogram with Fitted Lines
    ax1 = plt.subplot(1, 2, 1)
    for ind, dur in enumerate(durations):
        bin_centers, hist = dur

        avg[ind] = np.mean(hist)
        std[ind] = np.std(hist)

        x = bin_centers
        y = hist
        y[y == 0] = np.nan  # 避免对数为零的情况
        ax1.loglog(x, y, '*', markersize=10, color=colors[ind * 16 + 9], label=f'Dataset {ind + 1}')

        # 拟合仅在 5 < x < 50 范围内
        mask = (x > 5) & (x < 50) & ~np.isnan(y)
        x_fit = x[mask]
        y_fit = y[mask]

        log_x = np.log(x_fit)
        log_y = np.log(y_fit)

        # 进行直线拟合
        coefficients = np.polyfit(log_x, log_y, 1)
        covs[ind] = coefficients[0]

        log_y_fit = np.polyval(coefficients, log_x)
        y_fit_line = np.exp(log_y_fit)
        ax1.loglog(x_fit, y_fit_line, color=colors[ind * 16], label=f'k={coefficients[0]:.2f}')

    # 设置坐标轴标签
    ax1.set_xlabel(f'{title} (day)', fontsize=24)
    ax1.set_ylabel('Probability', fontsize=24)

    # 设置刻度字体大小
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)

    # 设置坐标轴范围（使用原始数据范围）
    ax1.set_xlim(1, 500)
    ax1.set_ylim(1e-9, 1)

    # 去除右侧和上侧的轴
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    # 添加 inset 图：拟合斜率 k 的变化
    # 增大 inset 的尺寸并调整位置
    ax_inset = inset_axes(ax1, width="60%", height="60%", loc='lower left',
                          bbox_to_anchor=(0.1, 0.1, 0.5, 0.5),
                          bbox_transform=ax1.transAxes)

    # 使用条形图显示拟合斜率 k，并保持颜色一致
    # 修改点：添加 width 参数来调整柱子的宽度，使其稍微细一点
    sns.barplot(x=np.arange(1, len(covs) + 1), y=covs,
                palette=[colors[ind * 16 + 10] for ind in range(len(covs))],
                ax=ax_inset, width=0.5)  # 设置 width 为 0.5

    ax_inset.set_title('Variation of k', fontsize=16)
    ax_inset.set_xlabel('Wet-day frequency', fontsize=14)
    ax_inset.set_ylabel('k', fontsize=14)
    ax_inset.tick_params(axis='x', labelsize=12)
    ax_inset.tick_params(axis='y', labelsize=12)
    ax_inset.spines['right'].set_visible(False)
    ax_inset.spines['top'].set_visible(False)

    # 第二个子图：统计信息 - 使用箱线散点图
    ax2 = plt.subplot(1, 2, 2)

    # 假设每个 vbin 对应多个数据集，这里需要重构数据以适应箱线散点图
    # 创建一个 DataFrame 格式的数据
    data = []
    for ind, dur in enumerate(durations):
        bin_centers, hist = dur
        for vbin, value in zip(vbins, hist):
            data.append({'Wet-day Frequency': vbin, 'Value': value})

    df = pd.DataFrame(data)

    # 使用 Seaborn 绘制箱线图
    sns.boxplot(x='Wet-day Frequency', y='Value', data=df, palette='muted', ax=ax2)

    # 在箱线图上叠加散点图
    # sns.stripplot(x='Wet-day Frequency', y='Value', data=df,
    #               color='black', alpha=0.5, size=3, jitter=True, ax=ax2)

    # 设置坐标轴标签和格式
    ax2.set_xlabel('Wet-day Frequency', fontsize=24)
    ax2.set_ylabel('Value', fontsize=24)
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_xticklabels([f"{x:.2f}" for x in vbins], fontsize=18)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # 添加网格线（如果需要更明显的网格线，可以调整参数）
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 去除右侧和上侧的轴
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # 优化整体布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(fig_name)
    plt.show()


def plt_duration(durations, title, vbins, fig_name):
    sns.set_style('ticks')
    sns.set_context("paper", font_scale=1.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10), constrained_layout=True)

    # 根据自己的配色函数来
    _, colors = setup_colors()

    # -- 这里的 durations 应该是 [(bin_centers, hist, mean_val, std_val, ci_mean, ci_std), ...]
    num_bins = len(vbins)

    # 用于右侧子图的统计量数组
    mean_arr = np.zeros(num_bins)
    std_arr = np.zeros(num_bins)
    # 均值的置信区间上下限
    mean_ci_lower = np.zeros(num_bins)
    mean_ci_upper = np.zeros(num_bins)
    # 标准差的置信区间上下限
    std_ci_lower = np.zeros(num_bins)
    std_ci_upper = np.zeros(num_bins)
    # k斜率（你原本计算的）
    covs = np.zeros(num_bins)

    # =============== 左侧子图：绘制分布及拟合 ===============
    for ind, d in enumerate(durations):
        # 解包
        bin_centers, hist, mean_val, std_val, ci_mean, ci_std = d
        # ci_mean, ci_std 分别是 (下限, 上限)
        mean_ci_lower[ind], mean_ci_upper[ind] = ci_mean
        std_ci_lower[ind], std_ci_upper[ind] = ci_std

        # 记录到数组里
        mean_arr[ind] = mean_val
        std_arr[ind] = std_val

        # ----- 下面是你原先的绘制直方图点 & 拟合曲线的逻辑 ------
        x = bin_centers
        y = hist
        y[y == 0] = np.nan

        # 绘制概率分布点
        ax1.loglog(
            x, y, marker='o', markersize=7, linestyle='None',
            color=colors[ind % len(colors)],
            label=f'WDF: {vbins[ind]:.2f}'
        )

        # 进行一次对数线性拟合(原本的多项式拟合)
        mask_fit = (x > 5) & (x < 50) & ~np.isnan(y)
        x_fit = x[mask_fit]
        y_fit = y[mask_fit]
        log_x_fit = np.log(x_fit)
        log_y_fit = np.log(y_fit)
        coefficients = np.polyfit(log_x_fit, log_y_fit, 1)  # slope, intercept
        covs[ind] = coefficients[0]  # 只存斜率

        # 拟合曲线
        x_line = np.linspace(2, 100, 500)
        log_y_line = np.polyval(coefficients, np.log(x_line))
        y_line = np.exp(log_y_line)

        ax1.loglog(x_line, y_line,
                   color=colors[ind % len(colors)],
                   linestyle='-', linewidth=2)

    # 左子图修饰
    ax1.set_xlabel(f'{title} (day)', fontsize=18)
    ax1.set_ylabel('Probability', fontsize=18)
    ax1.legend(loc='upper right', frameon=False)
    ax1.grid(axis='y', alpha=0.75, linestyle='--')
    ax1.set_xlim(1, 500)
    ax1.set_ylim(1e-9, 1)
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='major', length=12, width=2)
    ax1.tick_params(axis='both', which='minor', length=6, width=1)

    # =============== 右侧子图：绘制均值、标准差、斜率等 ===============
    ax2_twin = ax2.twinx()

    # ---- 绘制均值及其置信区间 ----
    mean_color = '#2c7bb6'

    # 曲线本身
    line1, = ax2.plot(
        vbins, mean_arr, marker='o', mfc='none', mec=mean_color,
        markersize=10, color=mean_color, linestyle='--',
        linewidth=2, label='Mean'
    )

    # 阴影表示误差区间：Mean ± 95%CI
    ax2.fill_between(
        vbins,
        mean_ci_lower,
        mean_ci_upper,
        color=mean_color, alpha=0.2
    )

    # ---- 绘制标准差及其置信区间 ----
    std_color = '#fdae61'

    line2, = ax2.plot(
        vbins, std_arr, marker='^', mfc='none', mec=std_color,
        markersize=10, color=std_color, linestyle='--',
        linewidth=2, label='Std Dev'
    )

    ax2.fill_between(
        vbins,
        std_ci_lower,
        std_ci_upper,
        color=std_color, alpha=0.2
    )

    # ---- 绘制对数拟合斜率 k ----
    cov_color = '#d7191c'
    line3, = ax2_twin.plot(
        vbins, -covs, marker='D', markersize=10, mfc='none', mec=cov_color,
        color=cov_color, linestyle='-', linewidth=2, label=r'$\alpha$'
    )

    # 右侧子图轴标签
    ax2.set_xlabel('Wet-day Frequency', fontsize=18)
    ax2.set_ylabel('Mean & Std Dev (days)', fontsize=18)
    ax2_twin.set_ylabel(r'$\alpha$', fontsize=18)

    # 格式 & 外观微调
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    for spine in ax2_twin.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    # 创建联合图例
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='best', fontsize=14)

    # 调整刻度与可视化
    ax2.tick_params(axis='both', which='major', length=12, width=2)
    ax2_twin.tick_params(axis='both', which='major', length=12, width=2)

    plt.savefig(fig_name, bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


@auto_close_plot
def plt_duration_combined(durations_list, title, vbins, fig_name):
    """
    durations_list: list of lists, 每个内层列表中每个元素是 (bin_centers, hist, mean, std, ci_mean, ci_std)
    title: 字符串，用于X轴标签
    vbins: array-like, Wet-day frequency 分箱
    fig_name: 输出文件名
    """
    plt.style.use('default')

    n_rows = len(durations_list)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(7.5, 1.5 * n_rows),
                             constrained_layout=True)

    _, colors = setup_colors()

    for row, durations in enumerate(durations_list):
        if n_rows == 1:
            ax1, ax2 = axes
        else:
            ax1, ax2 = axes[row]

        num_bins = len(vbins)
        # 初始化统计量
        avg      = np.zeros(num_bins)
        std      = np.zeros(num_bins)
        med      = np.zeros(num_bins)      # 如果需要，可以填充
        covs     = np.zeros(num_bins)
        covs_se  = np.zeros(num_bins)      # 新增：斜率标准误

        # 如果要同时绘制均值/标准差的置信区间，可用 mean_lower/mean_upper 等
        mean_lower = np.zeros(num_bins)
        mean_upper = np.zeros(num_bins)
        std_lower  = np.zeros(num_bins)
        std_upper  = np.zeros(num_bins)

        # ---- 第1列：概率分布和拟合线 ----
        for ind, dur in enumerate(durations):
            bin_centers, hist, mean_val, std_val, ci_mean, ci_std = dur

            avg[ind]        = mean_val
            std[ind]        = std_val
            mean_lower[ind] = ci_mean[0]
            mean_upper[ind] = ci_mean[1]
            std_lower[ind]  = ci_std[0]
            std_upper[ind]  = ci_std[1]

            x = bin_centers
            y = hist.copy()
            y[y == 0] = np.nan

            mask_fit = (x > 5) & (x < 50) & ~np.isnan(y)
            x_fit = x[mask_fit]
            y_fit = y[mask_fit]

            # 点状分布
            ax1.loglog(x, y,
                       marker='o', markersize=3,
                       linestyle='None',
                       color=colors[ind * 16 + 10],
                       label=f'WDF: {vbins[ind]:.2f}~'
                             f'{0.99 if ind == num_bins-1 else vbins[ind + 1]:.2f}')

            # 拟合 & 误差
            log_x_fit = np.log(x_fit)
            log_y_fit = np.log(y_fit)
            (slope, intercept), cov_mat = np.polyfit(
                log_x_fit, log_y_fit, 1, cov=True
            )
            covs[ind]    = slope
            covs_se[ind] = np.sqrt(cov_mat[0, 0])

            # 拟合曲线
            x_line    = np.linspace(2, 100, 1000)
            log_x_line = np.log(x_line)
            log_y_line = np.polyval((slope, intercept), log_x_line)
            y_line    = np.exp(log_y_line)
            ax1.loglog(x_line, y_line,
                       color=colors[ind * 16 + 10],
                       linestyle='-', linewidth=1)

        ax1.legend(loc='upper right',
                   labelspacing=0.02,
                   handletextpad=0.1,
                   handlelength=0.3,
                   markerscale=0.4)
        if row == n_rows - 1:
            ax1.set_xlabel(f'{title} (day)', fontsize=18)
        ax1.set_ylabel('Probability', fontsize=18)
        ax1.grid(axis='y', alpha=0.75, linestyle='--')
        ax1.set_xlim(1, 500)
        ax1.set_ylim(1e-9, 1)
        ax1.minorticks_on()

        # ---- 第2列：Mean、Std 和斜率 α（附带误差棒） ----
        color_low = '#1f78b4'
        color_mid = '#e66101'
        mean_color = color_low
        std_color  = color_low
        cov_color  = color_mid

        line1, = ax2.plot(vbins, avg,
                          linestyle='--', linewidth=2,
                          marker='o', markersize=5,
                          mfc='none', mec=mean_color,
                          color=mean_color,
                          label='Mean')
        line2, = ax2.plot(vbins, std,
                          linestyle='--', linewidth=2,
                          marker='^', markersize=5,
                          mfc='none', mec=std_color,
                          color=std_color,
                          label='Std')

        if row == n_rows - 1:
            ax2.set_xlabel('Wet-day frequency', fontsize=18)
        ax2.set_ylabel('Mean & Std (mm)', fontsize=18)

        ax2_twin = ax2.twinx()
        line3 = ax2_twin.errorbar(
            vbins, -covs, yerr=covs_se,
            fmt='D',                  # 只画 marker
            markersize=5,
            linestyle='-', linewidth=1.5,
            color=cov_color,
            markerfacecolor='none',
            markeredgecolor=cov_color,
            ecolor=cov_color,         # 误差棒颜色
            elinewidth=1.5,           # 误差棒宽度
            capsize=6,                # 帽线半宽度
            capthick=1.5,             # 帽线宽度
            label=r'$\alpha$'
        )
        ax2_twin.set_ylabel(r'$\alpha$',
                            rotation=0,
                            color=cov_color,
                            fontsize=18)
        ax2_twin.tick_params(axis='y', colors=cov_color)

        # 统一图例
        lines  = [line1, line2, line3]
        labels = [L.get_label() for L in lines]
        ax2.legend(lines, labels,
                   loc='upper right',
                   labelspacing=0.02,
                   handletextpad=0.1,
                   handlelength=0.8,
                   markerscale=0.4)

    # 全局字体设置
    set_fonts_for_fig(fig,
                      scale_factor=1.2,
                      tick_scale_factor=0.8,
                      legend_scale_factor=0.5)

    # 添加 (a), (b), ... 标签
    for i, ax in enumerate(fig.axes[:n_rows*2]):
        label = chr(97 + i)
        ax.annotate(label,
                    xy=(0, 1), xycoords='axes fraction',
                    xytext=(-30, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    ha='left', va='bottom')

    plt.savefig(fig_name, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    return fig, axes
def plt_duration_exp(durations, title, vbins, fig_name):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    plt.style.use('default')

    # 定义 Exponential PDF（λ 参数）
    def exp_pdf(x, lam):
        return lam * np.exp(-lam * x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3), constrained_layout=True)
    _, colors = setup_colors()

    num_bins = len(vbins)
    avg = np.zeros(num_bins)
    std = np.zeros(num_bins)
    lam_vals = np.zeros(num_bins)
    lam_se = np.zeros(num_bins)

    # 置信区间存储（如有需要）
    mean_lower = np.zeros(num_bins)
    mean_upper = np.zeros(num_bins)
    std_lower = np.zeros(num_bins)
    std_upper = np.zeros(num_bins)

    # --- 子图 1：直方图 + 指数分布拟合 ---
    for ind, dur in enumerate(durations):
        bin_centers, hist, mean_val, std_val, ci_mean, ci_std = dur

        avg[ind] = mean_val
        std[ind] = std_val
        mean_lower[ind], mean_upper[ind] = ci_mean
        std_lower[ind], std_upper[ind] = ci_std

        x = bin_centers
        y = hist.copy()
        y[y == 0] = np.nan

        # 指数分布拟合区间
        mask_fit = (x > 0) & (x < 500) & ~np.isnan(y)
        x_fit = x[mask_fit]
        y_fit = y[mask_fit]

        # 绘制散点
        ax1.loglog(
            x, y,
            marker='o', markersize=3, linestyle='None',
            color=colors[ind * 16 + 10],
            label=f'WDF: {vbins[ind]:.2f}~{(0.99 if ind == num_bins-1 else vbins[ind + 1]):.2f}'
        )

        # 指数分布拟合
        try:
            popt, pcov = curve_fit(
                exp_pdf,
                x_fit,
                y_fit,
                p0=[0.1],  # 初始猜测 λ=0.1
                bounds=(0, np.inf),
            )
            lam_hat = popt[0]
            lam_vals[ind] = lam_hat
            lam_se[ind] = np.sqrt(np.diag(pcov))[0]

            x_exp = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
            y_exp = exp_pdf(x_exp, lam_hat)
            ax1.loglog(
                x_exp, y_exp,
                linestyle='--', linewidth=1.5,
                color=colors[ind * 16 + 10],
                # label=f'Exp fit λ={lam_hat:.2f}'
            )
        except Exception as e:
            print(f"[Exp fit failed for bin {ind}]:", e)

    ax1.set_xlabel(f'{title} (day)', fontsize=18)
    ax1.set_ylabel('Probability', fontsize=18)
    ax1.grid(axis='y', alpha=0.75, linestyle='--')
    ax1.set_xlim(1, 500)
    ax1.set_ylim(1e-9, 1)
    ax1.legend(loc='upper right', labelspacing=0.04, handletextpad=0.1)
    ax1.minorticks_on()

    # --- 子图 2：均值、标准差及 λ（带标准误） ---
    color_low = '#1f78b4'
    color_mid = '#e66101'

    line1, = ax2.plot(
        vbins, avg,
        marker='o', markersize=5, mfc='none', mec=color_low,
        linestyle='--', linewidth=1.5, color=color_low, label='Mean'
    )
    line2, = ax2.plot(
        vbins, std,
        marker='^', markersize=5, mfc='none', mec=color_low,
        linestyle='--', linewidth=1.5, color=color_low, label='Std'
    )
    ax2.set_xlabel('Wet-day frequency', fontsize=18)
    ax2.set_ylabel('Mean & Std (mm)', fontsize=18)

    # 双 y 轴显示 λ 值及其误差棒
    ax2_twin = ax2.twinx()
    line3 = ax2_twin.errorbar(
        vbins, lam_vals, yerr=lam_se,
        fmt='D', markersize=5,
        linestyle='-', linewidth=1.5,
        color=color_mid, ecolor=color_mid,
        capsize=6, capthick=1.5,
        markerfacecolor='none', markeredgecolor=color_mid,
        label='λ'
    )
    ax2_twin.set_ylabel('λ', rotation=0, color=color_mid, fontsize=18)
    ax2_twin.tick_params(axis='y', labelcolor=color_mid, colors=color_mid)

    # 合并图例
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', labelspacing=0.5)

    # 添加子图标注
    for idx, ax in enumerate([ax1, ax2]):
        ax.annotate(
            chr(97 + idx), xy=(0, 1), xycoords='axes fraction',
            xytext=(-30, 5), textcoords='offset points',
            fontsize=10, fontweight='bold', ha='left', va='bottom'
        )

    set_fonts_for_fig(fig, scale_factor=3, tick_scale_factor=0.8, legend_scale_factor=0.8)

    plt.savefig(fig_name, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    return ax1, ax2, ax2_twin

@auto_close_plot
def plt_duration_gamma(durations, title, vbins, fig_name):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gamma
    from scipy.optimize import curve_fit

    plt.style.use('default')

    # 定义 Gamma PDF（固定 loc=0）
    def gamma_pdf(x, a, scale):
        return gamma.pdf(x, a, loc=0, scale=scale)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3), constrained_layout=True)
    _, colors = setup_colors()

    num_bins = len(vbins)
    avg = np.zeros(num_bins)
    std = np.zeros(num_bins)

    # 新增：存储 Gamma 拟合参数及其误差
    a_hats = np.zeros(num_bins)
    a_errs = np.zeros(num_bins)
    scale_hats = np.zeros(num_bins)
    scale_errs = np.zeros(num_bins)

    # --- 子图 1：直方图 + Gamma 拟合 ---
    for ind, dur in enumerate(durations):
        bin_centers, hist, mean_val, std_val, ci_mean, ci_std = dur

        avg[ind] = mean_val
        std[ind] = std_val

        x = bin_centers
        y = hist.copy()
        y[y == 0] = np.nan

        # 画散点
        ax1.loglog(
            x, y,
            marker='o', markersize=3, linestyle='None',
            color=colors[ind * 16 + 10],
            label=f'WDF: {vbins[ind]:.2f}~{(0.99 if ind == num_bins-1 else vbins[ind + 1]):.2f}'
        )

        # —— 对直方图点做 Gamma 分布拟合 ——
        mask = ~np.isnan(y)
        try:
            popt, pcov = curve_fit(
                gamma_pdf,
                x[mask], y[mask],
                p0=[1.5, 10],
                bounds=(0, np.inf),
            )
            a_hat, scale_hat = popt
            # 从协方差矩阵中提取参数标准差
            perr = np.sqrt(np.diag(pcov))
            a_hats[ind], a_errs[ind] = a_hat, perr[0]
            scale_hats[ind], scale_errs[ind] = scale_hat, perr[1]

            # 生成并绘制 Gamma 拟合曲线
            x_gamma = np.logspace(np.log10(x[mask].min()), np.log10(x[mask].max()), 200)
            y_gamma = gamma_pdf(x_gamma, a_hat, scale_hat)
            ax1.loglog(
                x_gamma, y_gamma,
                linestyle='--', linewidth=1.5,
                color=colors[ind * 16 + 10],
                # label=f'Gamma fit α={a_hat:.2f}'
            )
        except Exception as e:
            print(f"[Gamma fit failed for bin {ind}]:", e)

    ax1.set_xlabel(f'{title} (day)', fontsize=18)
    ax1.set_ylabel('Probability', fontsize=18)
    ax1.grid(axis='y', alpha=0.75, linestyle='--')
    ax1.set_xlim(1, 500)
    ax1.set_ylim(1e-9, 1)
    ax1.legend(loc='upper right', labelspacing=0.04, handletextpad=0.1)
    ax1.minorticks_on()

    # --- 子图 2：均值、标准差及 Gamma 参数（带误差棒） ---
    color_low = '#1f78b4'
    color_mid = '#e66101'
    color_scale = '#33a02c'

    # 平均和标准差
    line1, = ax2.plot(
        vbins, avg,
        marker='o', markersize=5, mfc='none', mec=color_low,
        linestyle='--', linewidth=1.5, color=color_low, label='Mean'
    )
    line2, = ax2.plot(
        vbins, std,
        marker='^', markersize=5, mfc='none', mec=color_low,
        linestyle='--', linewidth=1.5, color=color_low, label='Std'
    )
    ax2.set_xlabel('Wet-day frequency', fontsize=18)
    ax2.set_ylabel('Mean & Std (mm)', fontsize=18)

    # Gamma 参数 α 和 scale
    ax2_twin = ax2.twinx()
    err_alpha = ax2_twin.errorbar(
        vbins, a_hats, yerr=a_errs,
        fmt='D', markersize=5, markerfacecolor='none', color=color_mid,markeredgecolor=color_mid,
        linestyle='-', linewidth=1.5, ecolor=color_mid, capsize=6, capthick=1.5,
        label=r'$\alpha$'
    )
    # err_scale = ax2_twin.errorbar(
    #     vbins, scale_hats, yerr=scale_errs,
    #     fmt='s', markersize=5, markerfacecolor='none', color=color_mid, markeredgecolor=color_scale,
    #     linestyle='-', linewidth=1.5, ecolor=color_scale, capsize=6, capthick=1.5,
    #     label='scale'
    # )
    ax2_twin.set_ylabel(r'$\alpha$',color=color_mid,rotation=0, fontsize=18)
    ax2_twin.tick_params(axis='y', labelcolor=color_mid)

    # 合并图例
    lines = [line1, line2, err_alpha]
    # lines = [line1, line2, err_alpha, err_scale]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center', labelspacing=0.5)

    # 子图标注
    for idx, ax in enumerate([ax1, ax2]):
        ax.annotate(
            chr(97 + idx), xy=(0, 1), xycoords='axes fraction',
            xytext=(-30, 5), textcoords='offset points',
            fontsize=10, fontweight='bold', ha='left', va='bottom'
        )

    set_fonts_for_fig(fig, scale_factor=3, tick_scale_factor=0.8, legend_scale_factor=0.8)

    plt.savefig(fig_name, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    return ax1, ax2, ax2_twin
@auto_close_plot
def plt_duration_origin(durations, title, vbins, fig_name):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from matplotlib.ticker import FuncFormatter

    plt.style.use('default')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3), constrained_layout=True)
    _, colors = setup_colors()

    num_bins = len(vbins)
    # 初始化统计量
    avg = np.zeros(num_bins)
    std = np.zeros(num_bins)
    med = np.zeros(num_bins)
    covs = np.zeros(num_bins)
    covs_se = np.zeros(num_bins)  # 存放斜率标准误

    # 初始化存储均值和标准差置信区间的数组（虽然现在不使用置信区间，但保留原变量）
    mean_lower = np.zeros(num_bins)
    mean_upper = np.zeros(num_bins)
    std_lower = np.zeros(num_bins)
    std_upper = np.zeros(num_bins)

    # 绘制第一个子图：概率分布与幂律拟合
    for ind, dur in enumerate(durations):
        bin_centers, hist, mean_val, std_val, ci_mean, ci_std = dur

        avg[ind] = mean_val
        std[ind] = std_val
        mean_lower[ind], mean_upper[ind] = ci_mean
        std_lower[ind], std_upper[ind] = ci_std

        x = bin_centers
        y = hist.copy()
        y[y == 0] = np.nan

        # 选取拟合区间
        mask_fit = (x > 5) & (x < 50) & ~np.isnan(y)
        x_fit = x[mask_fit]
        y_fit = y[mask_fit]

        # 绘制散点
        ax1.loglog(
            x, y,
            marker='o', markersize=3, linestyle='None',
            color=colors[ind * 16 + 10],
            label=f'WDF: {vbins[ind]:.2f}~{(0.99 if ind == num_bins-1 else vbins[ind + 1]):.2f}'
        )

        # 对数空间线性拟合，并获得协方差矩阵
        log_x = np.log(x_fit)
        log_y = np.log(y_fit)
        coef, cov_matrix = np.polyfit(log_x, log_y, 1, cov=True)
        slope = coef[0]
        slope_se = np.sqrt(cov_matrix[0, 0])

        covs[ind] = slope
        covs_se[ind] = slope_se

        # 绘制拟合直线
        x_line = np.linspace(2, 100, 1000)
        y_line = np.exp(np.polyval(coef, np.log(x_line)))
        ax1.loglog(x_line, y_line, linestyle='-', linewidth=1, color=colors[ind * 16 + 10])

    ax1.set_xlabel(f'{title} (day)', fontsize=18)
    ax1.set_ylabel('Probability', fontsize=18)
    ax1.grid(axis='y', alpha=0.75, linestyle='--')
    ax1.set_xlim(1, 500)
    ax1.set_ylim(1e-9, 1)
    ax1.legend(loc='upper right', labelspacing=0.04, handletextpad=0.1)
    ax1.minorticks_on()

    # 第二个子图：均值、标准差及斜率（带标准误）
    color_low = '#1f78b4'
    color_mid = '#e66101'

    # 绘制均值和标准差
    line1, = ax2.plot(
        vbins, avg,
        marker='o', markersize=5, mfc='none', mec=color_low,
        linestyle='--', linewidth=1.5, color=color_low, label='Mean'
    )
    line2, = ax2.plot(
        vbins, std,
        marker='^', markersize=5, mfc='none', mec=color_low,
        linestyle='--', linewidth=1.5, color=color_low, label='Std'
    )
    ax2.set_xlabel('Wet-day frequency', fontsize=18)
    ax2.set_ylabel('Mean & Std (mm)', fontsize=18)

    # 右侧 y 轴绘制斜率及其标准误
    ax2_twin = ax2.twinx()
    line3 = ax2_twin.errorbar(
        vbins, -covs, yerr=covs_se,
        fmt='D',                # 只画 marker，不画连线
        markersize=5,
        linestyle='-', linewidth=1.5, color=color_mid,
        markerfacecolor='none',
        markeredgecolor=color_mid,
        ecolor=color_mid,       # 误差棒颜色
        elinewidth=1.5,         # 误差线本身宽度
        capsize=6,              # 帽线半宽度（points）
        capthick=1.5,           # 帽线线宽
        label=r'$\alpha$'
    )
    ax2_twin.set_ylabel(r'$\alpha$', rotation=0, color=color_mid, fontsize=18)
    ax2_twin.tick_params(axis='y', labelcolor=color_mid, colors=color_mid)

    # 合并图例
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper center',labelspacing=0.5)

    # 添加子图编号
    for idx, ax in enumerate([ax1, ax2]):
        label = chr(97 + idx)
        ax.annotate(
            label, xy=(0, 1), xycoords='axes fraction',
            xytext=(-30, 5), textcoords='offset points',
            fontsize=10, fontweight='bold', ha='left', va='bottom'
        )

    set_fonts_for_fig(fig, scale_factor=3, tick_scale_factor=0.8, legend_scale_factor=0.8)

    plt.savefig(fig_name, bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    return ax1, ax2, ax2_twin
