from datetime import datetime, timedelta
import matplotlib
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # 新增用于设置 Seaborn 样式
import xarray as xr
from cartopy.util import add_cyclic_point

from workflow import load_data
from utils import point_path_data
from config import intern_data_path, onat_list
from config import path_png
from config_vis import set_fonts_for_fig, auto_close_plot
from config_vis import setup_colors
from map_lib import reverse_convert_longitude
from utils import convert_longitude


# 假设 setup_colors 和 adjust_all_font_sizes 已经定义
# 假设 auto_close_plot 装饰器已经定义
@auto_close_plot
def combined_plot(data_frequency, data_percentile, th, dr, onat_list, sp_fp):
    # 参数化字体大小和线宽，方便后续调整
    TITLE_FONT_SIZE = 28
    LABEL_FONT_SIZE = 30
    TICK_FONT_SIZE = 26
    LEGEND_FONT_SIZE = 24
    LINE_WIDTH = 1.5
    MARKER_SIZE = 10
    SPINE_WIDTH = 1
    COLORBAR_LABEL_SIZE = 24
    COLORBAR_TICK_SIZE = 22

    # 设置 Seaborn 样式
    sns.set_style('ticks')
    cmap, colors = setup_colors()
    # 创建一个更大的图，增大尺寸以适应更大的字体和边框
    scale_factor = 0.75
    # fig = plt.figure(figsize=(7.5, 6), constrained_layout=True)
    fig = plt.figure(figsize=(40 * 0.75, 24 * 0.75), constrained_layout=True)

    # 调整 GridSpec 为 2 行 2 列，每个子图单独占据一个网格
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        height_ratios=[1, 1],  # 两行高度相同
        width_ratios=[1, 1],  # 两列宽度相同
        hspace=0.4,  # 行间距
        wspace=0.2  # 列间距
    )

    # -------------------- 第一个子图：地图 --------------------
    # 使用 Mollweide 投影
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.Mollweide())

    ax_map.set_global()
    ax_map.coastlines(resolution='110m', linewidth=SPINE_WIDTH)

    gl = ax_map.gridlines(draw_labels=True, linewidth=1.5, color='gray', linestyle='--')
    # gl = ax_map.gridlines(draw_labels=True, linewidth=1.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = True
    # 调整标签样式，包括字体大小和对齐方式
    gl.xlabel_style = {'size': 0, 'ha': 'center', 'va': 'bottom'}
    gl.ylabel_style = {'size': LABEL_FONT_SIZE, 'ha': 'right', 'va': 'center'}
    # 手动调整 x 轴标签的位置

    # 假设数据最小值和最大值如下：

    # 手动生成 7 个边界值（6个区间），使得上限正好为 1
    levels = np.linspace(0, 1, 7)
    if data_frequency is not None and not data_frequency.squeeze().size == 0:
        infile = data_frequency.squeeze()
        if not hasattr(infile, 'longitude') or not hasattr(infile, 'latitude'):
            raise AttributeError("Dataset must contain 'longitude' and 'latitude' attributes.")

        tp, longitude = add_cyclic_point(infile, coord=infile.longitude)

        cont = ax_map.contourf(
            longitude,
            infile.latitude,
            tp,
            levels=levels,
            cmap=cmap,
            vmin=0,
            vmax=1,
            transform=ccrs.PlateCarree(),
            # extend = 'neither'
        )
        for i, (lon, lat) in enumerate(onat_list):
            print(f'per:{i} time:{infile.sel(longitude=lon, latitude=lat, method="nearest").values}')
            ax_map.scatter(reverse_convert_longitude(lon), lat, color='black', s=500, zorder=5,
                           transform=ccrs.PlateCarree())  # s是点的大小，zorder是图层顺序
            ax_map.text(reverse_convert_longitude(lon), lat, str(i + 1), color='white',
                        ha='center', fontsize='22', va='center', zorder=6,
                        transform=ccrs.PlateCarree())  # 在点上添加编号

    # -------------------- 在第一幅图内部添加 Colorbar --------------------
    if data_frequency is not None and not data_frequency.squeeze().size == 0:
        # 将 Colorbar 放置在地图下方，水平布局
        from matplotlib.ticker import FormatStrFormatter

        cbar = fig.colorbar(cont, ax=ax_map, orientation='horizontal', fraction=0.066, pad=0.06)
        # 设置刻度标签格式为保留两位小数
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        cbar.set_label('Wet-day Frequency', fontsize=COLORBAR_LABEL_SIZE, labelpad=10)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
        cbar.ax.xaxis.set_tick_params(width=SPINE_WIDTH)

    # -------------------- 第二个子图：分布图 --------------------
    ax_dist = fig.add_subplot(gs[1, 0])
    nbins = data_percentile.shape[0]
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, nbins)

    dist_arr = np.asarray(data_percentile)

    indices = np.linspace(0, 6, 6,endpoint=False, dtype=int)  # 选取等距离的6个索引
    # indices = np.linspace(0, 99, 6, dtype=int)  # 选取等距离的6个索引
    for ind in indices:
        d = dist_arr[ind, :]
        if isinstance(d, float):
            continue  # 忽略缺失数据
        else:
            color = colors[ind * max(1, round(len(colors) / nbins))+10]
            ax_dist.plot(d, np.arange(1, 101), '.', color=color, markersize=MARKER_SIZE)
            # ax_dist.plot(d, np.arange(1, 101), '.', color=color, markersize=MARKER_SIZE, alpha=0.6)

    ax_dist.set_ylabel('Percentile', fontsize=LABEL_FONT_SIZE)
    ax_dist.set_xlabel('Precipitation (mm/day)', fontsize=LABEL_FONT_SIZE)

    ax_dist.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    ax_dist.set_yticks([1, 10, 25, 50, 75, 90, 99])
    ax_dist.set_yticklabels([1, 10, 25, 50, 75, 90, 99], fontsize=LABEL_FONT_SIZE)

    ax_dist.set_xticks([1, 10, 100, 500])
    ax_dist.set_xticklabels([1, 10, 100, 500], fontsize=LABEL_FONT_SIZE)

    ax_dist.grid(ls="--", color='k', alpha=0.3)
    ax_dist.set_xscale("log")
    ax_dist.set_xlim(1, 500)

    for spine in ax_dist.spines.values():
        spine.set_linewidth(SPINE_WIDTH)

    ax_dist.tick_params(
        axis='both',
        which='both',
        direction='out',
        length=8,  # 增加刻度线长度
        width=2,  # 增加刻度线宽度
        right=False,  # 不启用右侧刻度线
        top=False,  # 不启用顶部刻度线
        labelsize=10
    )
    ax_dist.xaxis.set_ticks_position('bottom')
    ax_dist.yaxis.set_ticks_position('left')

    # 移除第二个子图中的 Colorbar 相关代码
    # -------------------- 第三个子图：时间序列 --------------------
    ax_pt = fig.add_subplot(gs[0, 1])

    if isinstance(dr, xr.DataArray):
        df_2011 = dr.sel(time=slice('2011-01-01', '2011-12-31'))
        times = pd.to_datetime(df_2011['time'].values)
        data = df_2011.values
    elif isinstance(dr, np.ndarray):
        start_date = datetime(2011, 1, 1)
        times = [start_date + timedelta(days=i) for i in range(len(dr))]
        df_2011 = dr[:365]
        times = times[:365]
        data = df_2011
    else:
        raise TypeError("dr must be of type xarray.DataArray or numpy.ndarray")

    precip_color = '#1f77b4'
    threshold_color = '#d62728'
    fill_color = '#ff7f0e'

    ax_pt.plot(times, data, label='Precipitation', color=precip_color)

    th_value = th.values if isinstance(th, xr.DataArray) else th
    ax_pt.axhline(y=th_value, linestyle='--', color=threshold_color, linewidth=3, label=f'Threshold {th_value}')

    above_th = data > th_value
    ax_pt.fill_between(times, data, th_value, where=above_th, interpolate=True, color=fill_color, alpha=0.5, label='Above Threshold')

    ax_pt.set_yscale('log')

    ax_pt.set_xlabel('Time', fontsize=LABEL_FONT_SIZE)
    ax_pt.set_ylabel('Precipitation (mm/day)', fontsize=LABEL_FONT_SIZE)

    if len(times) > 0:
        padding = pd.DateOffset(days=15)
        start = times[0] - padding
        end = times[-1] + padding
        ax_pt.set_xlim(start, end)

    ax_pt.set_ylim(0.0001, 100)

    ax_pt.xaxis.set_major_locator(mdates.MonthLocator())
    ax_pt.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.setp(ax_pt.get_xticklabels(), rotation=0, ha='center', fontsize=TICK_FONT_SIZE)

    ax_pt.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)

    ax_pt.tick_params(
        axis='both',
        which='both',
        direction='out',
        length=8,  # 增加刻度线长度
        width=2,  # 增加刻度线宽度
        right=False,  # 不启用右侧刻度线
        top=False,  # 不启用顶部刻度线
        labelsize=10
    )

    for spine in ax_pt.spines.values():
        spine.set_linewidth(SPINE_WIDTH)

    # -------------------- 第四个子图：放大时间序列中的阴影部分 --------------------
    ax_zoom = fig.add_subplot(gs[1, 1])

    # 选择放大区域，例如选择某一时间段或某一特定事件
    # 这里以放大前一个子图的一个特定时间段为例
    zoom_start = datetime(2011, 8, 1)
    zoom_end = datetime(2011, 8, 31)

    mask = (times >= zoom_start) & (times <= zoom_end)
    zoom_times = np.array(times)[mask]
    zoom_data = np.array(data)[mask]

    ax_zoom.plot(zoom_times, zoom_data, label='Precipitation', linewidth=3, color=precip_color)
    ax_zoom.axhline(y=th_value, linestyle='--', color=threshold_color, label=f'80th percentile Threshold')
    ax_zoom.fill_between(zoom_times, zoom_data, th_value, where=(zoom_data > th_value), interpolate=True, color=fill_color, alpha=0.5)

    ax_zoom.set_yscale('log')
    ax_zoom.set_xlabel('Time', fontsize=LABEL_FONT_SIZE)
    ax_zoom.set_ylabel('Precipitation (mm/day)', fontsize=LABEL_FONT_SIZE)
    ax_zoom.set_xlim(zoom_start, zoom_end)
    ax_zoom.set_ylim(0.01, 100)

    ax_zoom.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.setp(ax_zoom.get_xticklabels(), rotation=0, ha='center', fontsize=TICK_FONT_SIZE - 4)

    ax_zoom.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE - 2)

    def d_tick(ax_d):
        ax_d.tick_params(
            axis='both',
            which='both',
            direction='out',
            length=8,  # 增加刻度线长度
            width=2,  # 增加刻度线宽度
            right=False,  # 不启用右侧刻度线
            top=False,  # 不启用顶部刻度线
        )
        # 如果需要对主刻度和副刻度进行不同的设置，可以分开调用 tick_params
        ax_d.tick_params(
            axis='both',
            which='major',
            length=15,  # 主刻度线长度
            width=2  # 主刻度线宽度
        )

        ax_d.tick_params(
            axis='both',
            which='minor',
            length=10,  # 副刻度线长度
            width=1  # 副刻度线宽度
        )

    d_tick(ax_zoom)
    d_tick(ax_dist)
    d_tick(ax_pt)
    for spine in ax_zoom.spines.values():
        spine.set_linewidth(SPINE_WIDTH)

    # -------------------- 调整第一个子图的边框粗细 --------------------
    try:
        ax_map.outline_patch.set_linewidth(SPINE_WIDTH)
    except AttributeError:
        for spine in ax_map.spines.values():
            spine.set_linewidth(SPINE_WIDTH)

    # 保存和显示图像
    set_fonts_for_fig(fig, scale_factor=4 * 0.75, label_scale_factor=0.5, legend_scale_factor=0.4, tick_scale_factor=0.45)
    plt.savefig(f"{sp_fp}/combined_plot_4_final.svg", format='svg', bbox_inches='tight')
    plt.show()
    plt.close()


def get_current_font_sizes(fig=None, ax=None):
    """ 获取当前图表中各个元素的真实字体大小 """
    if not fig:
        fig = plt.gcf()
    if not ax:
        ax = plt.gca()

    sizes = {}

    # 标题相关
    sizes['figure_title'] = fig._suptitle.get_fontsize() if fig._suptitle else None
    sizes['axes_title'] = ax.title.get_fontsize() if ax.title.get_text() else None

    # 坐标轴标签
    sizes['xaxis_label'] = ax.xaxis.label.get_fontsize()
    sizes['yaxis_label'] = ax.yaxis.label.get_fontsize()

    # 刻度标签（取第一个有效值）
    xticks = ax.get_xticklabels()
    sizes['xtick_labels'] = xticks[0].get_fontsize() if xticks else None

    yticks = ax.get_yticklabels()
    sizes['ytick_labels'] = yticks[0].get_fontsize() if yticks else None

    # 图例
    legend = ax.get_legend()
    if legend and legend.get_texts():
        sizes['legend'] = legend.get_texts()[0].get_fontsize()
    else:
        sizes['legend'] = None

    return sizes

# 使用示例
if __name__ == "__main__":
    dataset = xr.open_dataset('./internal_data/data_dict.nc')
    dr = point_path_data('total_precipitation')
    all_th = xr.open_dataarray(f'{intern_data_path}/all_th.nc')  # 假设 all_th.nc 是一个 DataArray
    # %% fig1
    coordinates = [(-67, 0), (80, -5), (-30, -50), (-120, -35), (-105, 50), (110, -24)]
    onat_list_fig1 = coordinates
    data_percentile, bins, indices = load_data(f'internal_data/wet_era5/wet_data.npz')
    era5_frequency = xr.open_dataset('era5_frequency_processed.nc').to_array().squeeze()

    # 定义参数
    data_frequency = era5_frequency  # 替换为实际数据
    data_percentile = data_percentile  # 替换为实际数据
    onat = coordinates[2]
    # onat = (-120.26, -4.78)
    th = all_th.sel(longitude=convert_longitude(onat[0]), latitude=onat[1], top_bins=20, method='nearest')
    dr_list = dr.sel(longitude=convert_longitude(onat[0]), latitude=onat[1], method='nearest')
    # path_png = path_png  # 替换为实际路径

    # 调用合并绘图函数
    combined_plot(
        data_frequency=data_frequency,
        data_percentile=data_percentile,
        th=th,
        dr=dr_list,
        onat_list=onat_list_fig1,
        sp_fp=path_png
    )
