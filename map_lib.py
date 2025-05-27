from os import times

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import mplcursors
import numpy as np
import pandas as pd
import seaborn as sns
import xarray
import xarray as xr
from cartopy.util import add_cyclic_point
from dask.dot import label
from matplotlib import colors as clr
from matplotlib.colors import ListedColormap

from utils import get_list_form_onat
from config import onat_list
from config_vis import auto_close_plot, setup_colorbar, adjust_all_font_sizes, set_fonts_for_fig
from config_vis import setup_colors


# from matplotlib.ticker import FuncFormatter


def reverse_convert_longitude(lon):
    if lon > 180:
        return lon - 360
    else:
        return lon


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


def wdp_era5(data_frequency, data_percentile, cp_percentile, lsp_percentile, sp_fp, colorbar_title):
    cdict = {
        'red': ((0.0, inter_from_256(64), inter_from_256(64)),
                (1 / 5 * 1, inter_from_256(102), inter_from_256(102)),
                (1 / 5 * 2, inter_from_256(235), inter_from_256(235)),
                (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
                (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
                (1.0, inter_from_256(169), inter_from_256(169))),
        'green': ((0.0, inter_from_256(57), inter_from_256(57)),
                  (1 / 5 * 1, inter_from_256(178), inter_from_256(178)),
                  (1 / 5 * 2, inter_from_256(240), inter_from_256(240)),
                  (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
                  (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
                  (1 / 5 * 5, inter_from_256(23), inter_from_256(23))),
        'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
                 (1 / 5 * 1, inter_from_256(255), inter_from_256(255)),
                 (1 / 5 * 2, inter_from_256(185), inter_from_256(185)),
                 (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
                 (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
                 (1.0, inter_from_256(69), inter_from_256(69))),
    }
    all_area_num = data_percentile.shape[0]
    cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=all_area_num)
    colors = cmap(np.linspace(0, 1, 100))
    # infile = xarray.open_dataset(data_frequency).squeeze() * 100
    infile = data_frequency.squeeze() * 100
    dist = data_percentile
    dist_arr = np.asarray(dist)

    fig = plt.figure(figsize=(13, 15), constrained_layout=True)
    fig.tight_layout()

    ax = plt.subplot(4, 1, 1, projection=ccrs.PlateCarree())
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, 'a.', transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.coastlines()

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

    plt.xlabel('Longitude', fontsize='20')
    plt.ylabel('Latitude', fontsize='20')
    tp, longitude = add_cyclic_point(infile, infile.longitude)  # connects the two ends of the longitude array
    cont = plt.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap, vmin=0, vmax=100)
    # ax.set_global()

    ax = plt.subplot(4, 1, 2)
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, 'b.', transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

    for ind, d in enumerate(dist_arr[0:100, :]):
        if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
            None
        else:
            plt.plot(d, np.arange(1, 101), '.', color=colors[ind * round(100 / all_area_num)], markersize=10)

    plt.ylabel('Percentile', fontsize=15)
    plt.xlabel('Cumulative precipitation (mm/day)', fontsize=15)
    plt.yticks([1, 10, 25, 50, 75, 90, 99])
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.xscale("log")
    plt.xlim(1, 500)
    plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])

    ax = plt.subplot(4, 1, 3)
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, 'b.', transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

    for ind, d in enumerate(cp_percentile):
        if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
            None
        else:
            plt.plot(d, np.arange(1, 101), '.', color=colors[ind * round(100 / all_area_num)], markersize=10)

    plt.ylabel('Percentile', fontsize=15)
    plt.xlabel('Cumulative precipitation (mm/day)', fontsize=15)
    plt.yticks([1, 10, 25, 50, 75, 90, 99])
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.xscale("log")
    plt.xlim(1, 500)
    plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])

    ax = plt.subplot(4, 1, 4)
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, 'b.', transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold',
            bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

    for ind, d in enumerate(lsp_percentile):
        if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
            None
        else:
            plt.plot(d, np.arange(1, 101), '.', color=colors[ind * round(100 / all_area_num)], markersize=10)

    plt.ylabel('Percentile', fontsize=15)
    plt.xlabel('Cumulative precipitation (mm/day)', fontsize=15)
    plt.yticks([1, 10, 25, 50, 75, 90, 99])
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.xscale("log")
    plt.xlim(1, 500)
    plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])

    cmap = plt.get_cmap(cmap, 20)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)
    clbar.set_label(colorbar_title, fontsize='16')
    plt.savefig('F:/liusch/remote_project/climate_new/temp_fig/sp_fp/' + sp_fp)
    plt.show()


def wdp_era5_geography_interactive(data_frequency, sp_fp, all_area_num=20, cmap=None, gradient_data=None, projection=ccrs.PlateCarree):
    plt.ion()  # Turn on interactive mode
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, all_area_num)
    cmap, colors = setup_colors(cmap)

    # Data Preprocessing
    infile = data_frequency.squeeze()
    infile = infile.assign_coords(longitude=(infile.longitude - 180) % 360 - 180)
    infile = infile.sortby('longitude')
    tp, longitude = add_cyclic_point(infile, coord=infile.longitude)

    # Create figure and plot
    fig, ax1 = plt.subplots(figsize=(13, 10), subplot_kw={'projection': projection(central_longitude=180)}, constrained_layout=True)
    colorbar_title = data_frequency.name
    ax1.set_title(colorbar_title, fontsize=24)

    ax1.coastlines()
    if projection == ccrs.PlateCarree:
        ax1.set_xticks(np.arange(-180, 181, 60), crs=projection())
        ax1.set_yticks(np.arange(-90, 91, 30), crs=projection())
    if projection == ccrs.Robinson:
        ax1.gridlines()
    ax1.set_xlabel('Longitude', fontsize=20)
    ax1.set_ylabel('Latitude', fontsize=20)

    # ax1.pcolormesh(longitude, infile.latitude, tp, cmap=cmap, vmin=vbins[0], vmax=vbins[-1], transform=projection())
    # # Rasterize data and plot with imshow
    img = ax1.imshow(tp, extent=(longitude.min(), longitude.max(), infile.latitude.min(), infile.latitude.max()),
                     cmap=cmap, vmin=vbins[0], vmax=vbins[-1], origin='lower', transform=projection())

    # Add colorbar
    setup_colorbar(fig, vbins, cmap, orientation='horizontal', ax=ax1, title=data_frequency.name)
    adjust_all_font_sizes(fig, scale_factor=2.5)

    # Enable interactive hover
    mplcursors.cursor(img, hover=True)

    plt.show()


def wdp_era5_geography_in2(data_frequency, sp_fp, all_area_num=20, cmap=None, gradient_data=None, projection=ccrs.PlateCarree()):
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, all_area_num)
    cmap, colors = setup_colors(cmap)

    # Data preprocessing
    infile = data_frequency.squeeze()
    # infile = infile.assign_coords(longitude=(infile.longitude - 180))
    # infile = infile.sortby('longitude')
    tp, longitude = add_cyclic_point(infile, infile.longitude)

    # Create figure and plot
    fig, ax1 = plt.subplots(figsize=(13, 10), subplot_kw={'projection': projection}, constrained_layout=True)
    colorbar_title = data_frequency.name
    ax1.set_title(colorbar_title, fontsize=24)

    ax1.coastlines()
    if isinstance(projection, ccrs.PlateCarree):
        ax1.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=projection)
        ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=projection)
    if isinstance(projection, ccrs.Robinson):
        ax1.gridlines()

    plt.xlabel('Longitude', fontsize=20)
    plt.ylabel('Latitude', fontsize=20)

    # Plot data
    masked_data = np.isnan(tp)
    cmap_masked = ListedColormap(['none', 'grey'])
    cont = ax1.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap,
                        vmin=vbins[0], vmax=vbins[-1], transform=ccrs.PlateCarree())
    ax1.contourf(longitude, infile.latitude, masked_data, levels=[0, 0.5, 1], cmap=cmap_masked,
                 transform=ccrs.PlateCarree())

    # Plot gradient arrows
    if gradient_data is not None:
        u, v = gradient_data['u'], gradient_data['v']
        u, _ = add_cyclic_point(u, coord=gradient_data.longitude)
        v, _ = add_cyclic_point(v, coord=gradient_data.longitude)
        step = 3
        magnitude = np.sqrt(u ** 2 + v ** 2)
        max_magnitude = step
        u_normalized = u / magnitude * max_magnitude
        v_normalized = v / magnitude * max_magnitude
        ax1.quiver(longitude[::step], infile.latitude[::step], u_normalized[::step, ::step], v_normalized[::step, ::step],
                   transform=projection, color='black', scale=1, scale_units='xy', width=0.001, pivot='middle')
        fig.set_dpi(150)

    # Add colorbar
    setup_colorbar(fig, vbins, cmap, orientation='horizontal', ax=ax1, title=data_frequency.name)
    adjust_all_font_sizes(fig, scale_factor=2.5)

    return fig, ax1


@auto_close_plot
def wdp_era5_geography(
        data_frequency,
        sp_fp,
        all_area_num=20,
        cmap=None,
        gradient_data=None,
        projection=ccrs.Mollweide  # 修改默认投影为Mollweide
):
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, all_area_num)
    cmap, colors = setup_colors(cmap)

    # 数据预处理
    infile = data_frequency.squeeze()
    infile = infile.assign_coords(longitude=infile.longitude)
    infile = infile.sortby('longitude')
    tp, longitude = add_cyclic_point(infile, infile.longitude)

    # 创建图形和绘图
    fig, ax1 = plt.subplots(
        figsize=(15, 12),  # 调整图形尺寸适应椭圆投影
        subplot_kw={
            'projection': projection(central_longitude=0)  # 设置中央经线
        },
        constrained_layout=True
    )
    colorbar_title = data_frequency.name
    # ax1.set_title(colorbar_title, fontsize=24)
    ax1.set_global()

    # 地理特征设置
    ax1.coastlines(linewidth=0.5)
    ax1.gridlines(linewidth=0.3, color='gray', alpha=0.5)  # 添加网格线

    # 绘制数据（关键修改：添加transform参数）
    masked_data = np.isnan(tp)
    cmap_masked = ListedColormap(['none', 'grey'])

    # 绘制主数据
    cont = ax1.contourf(
        longitude,
        infile.latitude,
        tp,
        levels=all_area_num,
        cmap=cmap,
        vmin=vbins[0],
        vmax=vbins[-1],
        transform=ccrs.PlateCarree()  # 必须指定数据坐标系
    )

    # 覆盖缺失值
    ax1.contourf(
        longitude,
        infile.latitude,
        masked_data,
        levels=[0, 0.5, 1],
        cmap=cmap_masked,
        transform=ccrs.PlateCarree()  # 必须指定数据坐标系
    )

    # 绘制箭头梯度图
    if gradient_data is not None:
        u, v = gradient_data['u'], gradient_data['v']
        u, _ = add_cyclic_point(u, coord=gradient_data.longitude)
        v, _ = add_cyclic_point(v, coord=gradient_data.longitude)

        step = 4  # 适当增加步长以适应投影形变
        magnitude = np.sqrt(u**2 + v**2)
        max_magnitude = step * 2  # 增大标准化系数

        # 标准化箭头长度
        with np.errstate(divide='ignore', invalid='ignore'):
            u_normalized = np.where(magnitude > 0, u/magnitude*max_magnitude, 0)
            v_normalized = np.where(magnitude > 0, v/magnitude*max_magnitude, 0)

        # 绘制箭头（关键修改：transform参数）
        ax1.quiver(
            longitude[::step],
            infile.latitude[::step],
            u_normalized[::step, ::step],
            v_normalized[::step, ::step],
            transform=ccrs.PlateCarree(),  # 必须指定数据坐标系
            color='black',
            scale=25,  # 需要调整缩放系数
            width=0.002,
            headwidth=3,
            regrid_shape=20  # 改善箭头分布
        )

    # 添加colorbar
    cbar = fig.colorbar(cont, ax=ax1, orientation='horizontal', fraction=0.066, pad=0.06)
    cbar.set_label(colorbar_title, fontsize=24, labelpad=10)
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.xaxis.set_tick_params(width=1.5)
    # setup_colorbar(fig, vbins, cmap, orientation='horizontal', ax=ax1, title=data_frequency.name)
    adjust_all_font_sizes(fig, scale_factor=3,title_scale_factor=2)

    plt.savefig(sp_fp + '_geography.svg', bbox_inches='tight')  # 优化保存边界
    plt.show()


def plot_time_series_origin(dr, sp_fp_ts, onat_list, all_th,
                     edge_width=1, ax=None):
    """
    绘制时间序列图并保存。

    参数:
    - dr: 数据相关参数
    - sp_fp_ts: 字符串, 时间序列图像保存路径
    - onat_list: 列表, ONAT 列表
    - all_th: 阈值参数
    - edge_width: 边框宽度
    """
    sns.set_style('ticks')  # 设置 Seaborn 风格

    # 获取列表
    dr_list, th_list = get_list_form_onat(onat_list, dr, all_th)
    _, colors = setup_colors()
    colors = colors[::-1]
    # 绘制时间序列，创建6个子图，共享X轴
    fig_ts, axs = plt.subplots(6, 1, figsize=(14 * 1.25, 10), sharex=True,
                               gridspec_kw={'hspace': 0.00})  # 减小子图间距
    axs = axs.flatten()

    # 用于图例的元素
    legend_elements = []

    for idx, (dr_, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')

        # 使用2010年的数据
        if isinstance(dr_, xr.DataArray):
            df_2011 = dr_.sel(time=slice('2011-01-01', '2011-12-31'))
            time_coord = df_2011.time.to_pandas()
            values = df_2011.values
        elif isinstance(dr_, np.ndarray):
            df_2011 = dr_[0:365]
            time_coord = pd.date_range('2011-01-01', '2011-12-31', freq='D')
            values = dr_
        else:
            raise TypeError("dr 必须是 xarray.DataArray 或 numpy.ndarray 类型")

        # 绘制时间序列
        color = colors[idx * 16 + 10] if (idx * 16 + 10) < len(colors) else 'blue'
        axs[idx].plot(time_coord, values, linewidth=5, color=color, label=f'grid {idx+1}')
        # axs[idx].plot(time_coord, values, linewidth=5, color=color, label=f'(Lon: {onat[0]:.2f}, Lat: {onat[1]:.2f})')
        axs[idx].axhline(y=th, color=colors[idx] if idx < len(colors) else 'red', linestyle='--')

        # 添加高于阈值的阴影
        # axs[idx].fill_between(
        #     time_coord,
        #     10*2,
        #     0,
        #     where=(values > th),
        #     color=color,
        #     alpha=0.3  # 调整透明度使阴影更柔和
        # )

        axs[idx].set_yscale('log')
        # 设置 x 轴范围，确保数据从起点开始
        axs[idx].set_xlim(pd.to_datetime('2011-01-01'), pd.to_datetime('2011-12-31'))

        # 去掉内部网格线
        axs[idx].grid(False)

        # 去掉 Y 轴独立标签
        # axs[idx].set_ylabel(f'Area{idx+1}', fontsize=12, rotation=90, labelpad=5)

        # 去掉 X 轴标签和刻度标签
        if idx == 5:
            axs[idx].set_xlabel('Time')
        else:
            axs[idx].set_xlabel('')  # 移除 X 轴标签
        axs[idx].tick_params(axis='x', which='both', labelbottom=False)  # 移除 X 轴刻度标签

        # 设置 Y 轴范围和刻度
        axs[idx].set_ylim(1e-2, 1e2)
        axs[idx].set_yticks([1e-1, 1e0, 1e1])
        axs[idx].set_yticklabels(['0.1', '1', '10'], fontsize=10)

        # 调整刻度参数，添加右侧刻度线，增加刻度长度和宽度，方向指向图内

        # 调整 Y 轴刻度标签的距离
        axs[idx].tick_params(axis='y', which='both', pad=5)  # 调整标签与轴线的距离

        # 调整 X 轴刻度标签的位置，避免与 Y 轴刻度标签冲突
        axs[idx].tick_params(axis='x', which='major', length=8, width=2)  # 减少 pad
        axs[idx].tick_params(axis='y', which='minor', length=4, width=1)
        axs[idx].tick_params(axis='y', which='major', length=8, width=2, labelbottom=True)

        axs[idx].legend(loc='upper right', fontsize=10)
        # axs[idx].legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=10)

        # 设置时间序列子图边框厚度和颜色
        for spine in axs[idx].spines.values():
            spine.set_linewidth(edge_width)
            spine.set_color('black')  # 设置边框颜色为灰色

        # 收集图例元素
        # legend_elements.append(Patch(facecolor=color, label=f'Area{idx+1}'))

    # 添加统一的 Y 轴标签，居中显示
    fig_ts.text(0.03, 0.55, 'Precipitation (mm/day)', va='center', rotation='vertical', fontsize=4 * 2.4 * 0.5 * 0.75 * 10)

    # 创建统一的图例，放置在右侧中间
    # fig_ts.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.95, 0.5),
    #               title='Areas', fontsize=10, title_fontsize=12)
    #
    # 调整整体布局，确保图例和 Y 轴标签不被裁剪
    start_date = pd.to_datetime('2011-01-01')
    end_date = pd.to_datetime('2011-12-31')
    axs[5].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[5].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # axs[5].xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))

    # 如果希望刻度不从边界开始，可以调整刻度的起始位置
    # 例如，将第一个刻度推迟几天
    start_offset = pd.Timedelta(days=7)
    new_start = start_date + start_offset
    axs[5].set_xlim(new_start, end_date)
    axs[5].tick_params(axis='x', which='major', labelbottom=True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)

    # 调整整体字体大小（请确保 set_fonts_for_fig 函数的实现正确）
    set_fonts_for_fig(fig_ts, scale_factor=4 * 2.4 * 0.75, label_scale_factor=0.5, legend_scale_factor=0.5, tick_scale_factor=0.42)

    # 保存并显示时间序列图像
    plt.savefig(sp_fp_ts, bbox_inches='tight')
    plt.show()
    plt.close(fig_ts)  # 关闭时间序列图形，释放内存
def plot_time_series(dr, sp_fp_ts, onat_list, all_th,
                     edge_width=2, ax=None):
    """
    绘制时间序列图并保存。

    参数:
    - dr: 数据相关参数
    - sp_fp_ts: 字符串, 时间序列图像保存路径
    - onat_list: 列表, ONAT 列表
    - all_th: 阈值参数
    - edge_width: 边框宽度
    """
    sns.set_style('ticks')  # 设置 Seaborn 风格

    # 获取列表
    dr_list, th_list = get_list_form_onat(onat_list, dr, all_th)
    _, colors = setup_colors()
    colors = colors[::-1]
    # 绘制时间序列，创建6个子图，共享X轴
    fig_ts, axs = plt.subplots(6, 1, figsize=(14 * 1.25, 10), sharex=True,
                               gridspec_kw={'hspace': 0.00})  # 减小子图间距
    axs = axs.flatten()

    # 用于图例的元素
    legend_elements = []

    for idx, (dr_, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')

        # 使用2010年的数据
        if isinstance(dr_, xr.DataArray):
            df_2011 = dr_.sel(time=slice('2011-01-01', '2011-12-31'))
            time_coord = df_2011.time.to_pandas()
            values = df_2011.values
        elif isinstance(dr_, np.ndarray):
            df_2011 = dr_[0:365]
            time_coord = pd.date_range('2011-01-01', '2011-12-31', freq='D')
            values = dr_
        else:
            raise TypeError("dr 必须是 xarray.DataArray 或 numpy.ndarray 类型")

        # 绘制时间序列
        color = colors[idx * 16 + 10] if (idx * 16 + 10) < len(colors) else 'blue'
        axs[idx].plot(time_coord, values, linewidth=5, color=color, label=f'area {idx}')
        # axs[idx].plot(time_coord, values, linewidth=5, color=color, label=f'(Lon: {onat[0]:.2f}, Lat: {onat[1]:.2f})')
        axs[idx].axhline(y=th, color=colors[idx] if idx < len(colors) else 'red', linestyle='--')

        # 添加高于阈值的阴影
        axs[idx].fill_between(
            time_coord,
            10*2,
            0,
            where=(values > th),
            color=color,
            alpha=0.3  # 调整透明度使阴影更柔和
        )

        axs[idx].set_yscale('log')
        # 设置 x 轴范围，确保数据从起点开始
        axs[idx].set_xlim(pd.to_datetime('2011-01-01'), pd.to_datetime('2011-12-31'))

        # 去掉内部网格线
        axs[idx].grid(False)

        # 去掉 Y 轴独立标签
        # axs[idx].set_ylabel(f'Area{idx+1}', fontsize=12, rotation=90, labelpad=5)

        # 去掉 X 轴标签和刻度标签
        if idx == 5:
            axs[idx].set_xlabel('Time')
        else:
            axs[idx].set_xlabel('')  # 移除 X 轴标签
        axs[idx].tick_params(axis='x', which='both', labelbottom=False)  # 移除 X 轴刻度标签

        # 设置 Y 轴范围和刻度
        axs[idx].set_ylim(1e-2, 1e2)
        axs[idx].set_yticks([1e-1, 1e0, 1e1])
        axs[idx].set_yticklabels(['0.1', '1', '10'], fontsize=10)

        # 调整刻度参数，添加右侧刻度线，增加刻度长度和宽度，方向指向图内

        # 调整 Y 轴刻度标签的距离
        axs[idx].tick_params(axis='y', which='both', pad=5)  # 调整标签与轴线的距离

        # 调整 X 轴刻度标签的位置，避免与 Y 轴刻度标签冲突
        axs[idx].tick_params(axis='x', which='major', length=8, width=2)  # 减少 pad
        axs[idx].tick_params(axis='y', which='minor', length=4, width=1)
        axs[idx].tick_params(axis='y', which='major', length=8, width=2, labelbottom=True)

        axs[idx].legend(loc='upper right', fontsize=10)
        # axs[idx].legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=10)

        # 设置时间序列子图边框厚度和颜色
        for spine in axs[idx].spines.values():
            spine.set_linewidth(edge_width)
            spine.set_color('lightgrey')  # 设置边框颜色为灰色

        # 收集图例元素
        # legend_elements.append(Patch(facecolor=color, label=f'Area{idx+1}'))

    # 添加统一的 Y 轴标签，居中显示
    fig_ts.text(0.03, 0.55, 'Total Precipitation (mm)', va='center', rotation='vertical', fontsize=4 * 2.4 * 0.5 * 0.75 * 10)

    # 创建统一的图例，放置在右侧中间
    # fig_ts.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.95, 0.5),
    #               title='Areas', fontsize=10, title_fontsize=12)
    #
    # 调整整体布局，确保图例和 Y 轴标签不被裁剪
    start_date = pd.to_datetime('2011-01-01')
    end_date = pd.to_datetime('2011-12-31')
    axs[5].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axs[5].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # axs[5].xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))

    # 如果希望刻度不从边界开始，可以调整刻度的起始位置
    # 例如，将第一个刻度推迟几天
    start_offset = pd.Timedelta(days=7)
    new_start = start_date + start_offset
    axs[5].set_xlim(new_start, end_date)
    axs[5].tick_params(axis='x', which='major', labelbottom=True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)

    # 调整整体字体大小（请确保 set_fonts_for_fig 函数的实现正确）
    set_fonts_for_fig(fig_ts, scale_factor=4 * 2.4 * 0.75, label_scale_factor=0.5, legend_scale_factor=0.3, tick_scale_factor=0.42)

    # 保存并显示时间序列图像
    plt.savefig(sp_fp_ts, bbox_inches='tight')
    plt.show()
    plt.close(fig_ts)  # 关闭时间序列图形，释放内存

def plot_map(data_frequency, dr, sp_fp_map, onat_list, all_th, all_area_num=20, cmap=None,
             gradient_data=None, projection=ccrs.PlateCarree(), edge_width=1):
    """
    绘制地图并保存。

    参数:
    - data_frequency: xarray DataArray, 频率数据
    - dr: 数据相关参数
    - sp_fp_map: 字符串, 地图图像保存路径
    - onat_list: 列表, ONAT 列表
    - all_th: 阈值参数
    - all_area_num: 整数, 颜色区间数量
    - cmap: 颜色映射
    - gradient_data: 可选, 梯度数据
    - projection: cartopy 投影
    - edge_width: 边框宽度
    """
    sns.set_style('whitegrid')  # 设置 Seaborn 风格

    # 设置颜色映射和投影
    dr_list, th_list = get_list_form_onat(onat_list, dr, all_th)
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, all_area_num)
    cmap, colors = setup_colors(cmap)

    # 数据预处理
    infile = data_frequency.squeeze()
    infile = infile.assign_coords(longitude=infile.longitude)
    infile = infile.sortby('longitude')  # 确保数据按经度排序
    tp, longitude = add_cyclic_point(infile, infile.longitude)

    # 绘制地图
    fig_map = plt.figure(figsize=(10, 10))
    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    ax_map.coastlines(linewidth=edge_width)
    ax_map.gridlines(linewidth=edge_width)
    ax_map.set_global()

    # 绘制数据
    masked_data = np.isnan(tp)
    cmap_masked = ListedColormap(['none', 'grey'])
    cont = ax_map.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap,
                           vmin=vbins[0], vmax=vbins[-1], transform=ccrs.PlateCarree())
    ax_map.contourf(longitude, infile.latitude, masked_data, levels=[0, 0.5, 1], cmap=cmap_masked, transform=ccrs.PlateCarree())

    # 设置地图子图边框厚度和颜色
    for spine in ax_map.spines.values():
        spine.set_linewidth(edge_width)
        spine.set_color('black')  # 设置边框颜色为黑色

    # 添加 colorbar, 标题加(%)，并将ticks *100
    cb = setup_colorbar(fig_map, vbins, cmap, orientation='horizontal', ax=ax_map, title='wet-day frequency (%)')
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t * 100:.0f}' for t in ticks])

    # 调整整体字体大小
    set_fonts_for_fig(fig_map, scale_factor=4, legend_scale_factor=0.4, tick_scale_factor=0.45)

    # 显示并保存地图图像
    plt.show()
    plt.savefig(sp_fp_map, bbox_inches='tight')
    plt.close(fig_map)  # 关闭地图图形，释放内存

    print(f"地图已保存至: {sp_fp_map}")


def combined_plot_separate_saves(data_frequency, dr, sp_fp_map, sp_fp_ts, onat_list, all_th, all_area_num=20, cmap=None,
                                 gradient_data=None, projection=ccrs.PlateCarree(), edge_width=1):
    """
    在一个函数中绘制两个不同的图（地图和时间序列）并分别保存。

    参数:
    - data_frequency: xarray DataArray, 频率数据
    - dr: 数据相关参数
    - sp_fp_map: 字符串, 地图图像保存路径
    - sp_fp_ts: 字符串, 时间序列图像保存路径
    - onat_list: 列表, ONAT 列表
    - all_th: 阈值参数
    - all_area_num: 整数, 颜色区间数量
    - cmap: 颜色映射
    - gradient_data: 可选, 梯度数据
    - projection: cartopy 投影
    - edge_width: 边框宽度
    """

    # 设置 Seaborn 风格
    sns.set_style('whitegrid')  # 其他样式如 'ticks', 'darkgrid' 等

    # 设置颜色映射和投影
    dr_list, th_list = get_list_form_onat(onat_list, dr, all_th)
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, all_area_num)
    cmap, colors = setup_colors(cmap)

    # 数据预处理
    infile = data_frequency.squeeze()
    infile = infile.assign_coords(longitude=infile.longitude)
    infile = infile.sortby('longitude')  # 确保数据按经度排序
    tp, longitude = add_cyclic_point(infile, infile.longitude)

    # ===================== 绘制地图 =====================
    fig_map = plt.figure(figsize=(10, 10), dpi=100)
    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.Mollweide())
    ax_map.coastlines(linewidth=edge_width)
    ax_map.gridlines(linewidth=edge_width)
    ax_map.set_global()

    # 绘制数据
    masked_data = np.isnan(tp)
    cmap_masked = ListedColormap(['none', 'grey'])
    cont = ax_map.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap,
                           vmin=vbins[0], vmax=vbins[-1], transform=ccrs.PlateCarree())
    ax_map.contourf(longitude, infile.latitude, masked_data, levels=[0, 0.5, 1], cmap=cmap_masked, transform=ccrs.PlateCarree())

    # 设置地图子图边框厚度和颜色
    for spine in ax_map.spines.values():
        spine.set_linewidth(edge_width)
        spine.set_color('black')  # 设置边框颜色为黑色

    # 添加 colorbar, 标题加(%)，并将ticks *100
    cb = setup_colorbar(fig_map, vbins, cmap, orientation='horizontal', ax=ax_map, title='wet-day frequency (%)')
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t * 100:.0f}' for t in ticks])

    # 调整整体字体大小
    set_fonts_for_fig(fig_map, scale_factor=4, legend_scale_factor=0.4, tick_scale_factor=0.6)

    # 保存地图图像
    plt.show()
    # plt.savefig(sp_fp_map, bbox_inches='tight')
    plt.close(fig_map)  # 关闭地图图形，释放内存

    # ===================== 绘制时间序列 =====================
    fig_ts = plt.figure(figsize=(12, 5), dpi=100)
    gs_right = gridspec.GridSpec(6, 1, figure=fig_ts, hspace=0, height_ratios=[1] * 6)
    axs = []
    for i in range(6):
        ax = fig_ts.add_subplot(gs_right[i, 0])
        axs.append(ax)

    for idx, (dr_, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')

        # 使用2010年的数据
        if isinstance(dr_, xr.DataArray):
            df_2010 = dr_.sel(time=slice('2010-01-01', '2010-12-31'))
            time_coord = df_2010.time.to_pandas()
            values = df_2010.values
        elif isinstance(dr_, np.ndarray):
            df_2010 = dr_[0:365]
            time_coord = pd.date_range('2010-01-01', '2010-12-31', freq='D')
            values = dr_
        else:
            raise TypeError("dr 必须是 xarray.DataArray 或 numpy.ndarray 类型")

        # 绘制时间序列
        axs[idx].plot(time_coord, values, label='precipitation', color=colors[idx * 16 + 10])
        axs[idx].axhline(y=th, color=colors[idx], linestyle='--')
        axs[idx].set_yscale('log')

        # 去掉内部网格线
        axs[idx].grid(False)

        # 添加 Y 轴标题
        axs[idx].set_ylabel(f'Area{idx + 1}', fontsize=12, rotation=90)

        if idx == 5:
            axs[idx].set_xlabel('Time', fontsize=14)
            axs[idx].tick_params(axis='x', which='major', labelbottom=True)
            # 设置 x 轴范围，确保数据从起点开始
            axs[idx].set_xlim(pd.to_datetime('2010-01-01'), pd.to_datetime('2010-12-31'))
            # 设置刻度格式
            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            # 设置刻度从2月、5月、8月、11月开始，每3个月一个刻度
            axs[idx].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 5, 8, 11)))
        else:
            axs[idx].tick_params(axis='x', which='major', labelbottom=False)
            # 设置每3个月一个刻度
            axs[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # 设置 y 轴范围和刻度
        axs[idx].set_ylim(1e-2, 1e2)
        axs[idx].set_yticks([1e-1, 1e0, 1e1])
        axs[idx].set_yticklabels(['0.1', '1', '10'])

        # # 调整 y 轴刻度标签的对齐方式
        # for label in axs[idx].get_yticklabels():
        #     label.set_horizontalalignment('center')  # 可根据需要调整为 'right' 或 'left'

        # 调整刻度参数，添加右侧刻度线，增加刻度长度和宽度，方向指向图内
        axs[idx].tick_params(
            axis='both',
            which='both',
            direction='in',
            length=8,  # 增加刻度线长度
            width=2,  # 增加刻度线宽度
            right=False,  # 如果需要启用右侧刻度线，设置为 True
            top=False,  # 如果不需要顶部刻度线，可以保持为 False
            labelsize=10
        )

        # 调整 y 轴刻度标签的距离
        axs[idx].tick_params(axis='y', which='both', pad=10)  # 调整标签与轴线的距离

        # 调整 x 轴刻度标签的位置，避免与 y 轴刻度标签冲突
        axs[idx].tick_params(axis='x', which='major', pad=5)  # 减少 pad

        # 添加图例
        axs[idx].legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=10)
        axs[idx].tick_params(axis='both', labelsize=10)

        # 设置时间序列子图边框厚度和颜色
        for spine in axs[idx].spines.values():
            spine.set_linewidth(edge_width)
            spine.set_color('black')  # 设置边框颜色为黑色

    # 调整整体字体大小
    set_fonts_for_fig(fig_ts, scale_factor=4, label_scale_factor=0.6, legend_scale_factor=0.4, tick_scale_factor=0.6)

    # 保存时间序列图像
    plt.show()
    # plt.savefig(sp_fp_ts, bbox_inches='tight')
    plt.close(fig_ts)  # 关闭时间序列图形，释放内存

    print(f"地图已保存至: {sp_fp_map}")
    print(f"时间序列图已保存至: {sp_fp_ts}")


def combined_plot(data_frequency, dr, sp_fp, onat_list, all_th, all_area_num=20, cmap=None,
                  gradient_data=None, projection=ccrs.PlateCarree(), edge_width=1.5):
    # 设置 Seaborn 风格
    sns.set_style('whitegrid')  # 您可以选择 'ticks', 'darkgrid', 等其他样式

    # 设置颜色映射和投影
    dr_list, th_list = get_list_form_onat(onat_list, dr, all_th)
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, all_area_num)
    cmap, colors = setup_colors(cmap)

    # 数据预处理
    infile = data_frequency.squeeze()
    infile = infile.assign_coords(longitude=infile.longitude)
    infile = infile.sortby('longitude')  # 确保数据按经度排序
    tp, longitude = add_cyclic_point(infile, infile.longitude)

    # 创建主图形和GridSpec
    fig = plt.figure(figsize=(25, 7.5), dpi=100, constrained_layout=False)  # 增加高度以适应多个子图
    outer_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6], figure=fig)  # 调整右边宽度比例

    # 左边的地图子图, 使用Mollweide投影
    ax1 = fig.add_subplot(outer_gs[0, 0], projection=ccrs.Mollweide())
    ax1.coastlines(linewidth=edge_width)
    ax1.gridlines(linewidth=edge_width)
    ax1.set_global()

    # 绘制数据
    masked_data = np.isnan(tp)
    cmap_masked = ListedColormap(['none', 'grey'])
    cont = ax1.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap,
                        vmin=vbins[0], vmax=vbins[-1], transform=ccrs.PlateCarree())
    ax1.contourf(longitude, infile.latitude, masked_data, levels=[0, 0.5, 1], cmap=cmap_masked, transform=ccrs.PlateCarree())

    # 设置地图子图边框厚度和颜色
    for spine in ax1.spines.values():
        spine.set_linewidth(edge_width)
        spine.set_color('black')  # 设置边框颜色为黑色

    # 添加 colorbar, 标题加(%)，并将ticks *100
    cb = setup_colorbar(fig, vbins, cmap, orientation='horizontal', ax=ax1, title='wet-day frequency (%)')
    ticks = cb.get_ticks()
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t * 100:.0f}' for t in ticks])

    # 右边的时间序列子图，更加紧凑的间距
    # 使用共享 y 轴以确保边界对齐
    gs_right = outer_gs[0, 1].subgridspec(6, 1, hspace=0, height_ratios=[1] * 6)
    axs = []
    for i in range(6):
        if i == 0:
            ax = fig.add_subplot(gs_right[i, 0])
        else:
            ax = fig.add_subplot(gs_right[i, 0], sharex=axs[0], sharey=axs[0])
        axs.append(ax)

    for idx, (dr_, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')

        # 使用2010年的数据
        if isinstance(dr_, xr.DataArray):
            df_2010 = dr_.sel(time=slice('2010-01-01', '2010-12-31'))
            time_coord = df_2010.time.to_pandas()
            values = df_2010.values
        elif isinstance(dr_, np.ndarray):
            df_2010 = dr_[0:365]
            time_coord = pd.date_range('2010-01-01', '2010-12-31', freq='D')
            values = dr_
        else:
            raise TypeError("dr 必须是 xarray.DataArray 或 numpy.ndarray 类型")

        # 绘制时间序列
        axs[idx].plot(time_coord, values, label='precipitation', color=colors[idx * 16 + 10])
        axs[idx].axhline(y=th, color=colors[idx], linestyle='--')
        axs[idx].set_yscale('log')

        # 1. 去掉内部网格线
        axs[idx].grid(False)

        # 添加 Y 轴标题
        axs[idx].set_ylabel(f'Area{idx + 1}', fontsize=12, rotation=90, labelpad=10)

        if idx == 5:
            axs[idx].set_xlabel('Time', fontsize=14)
            axs[idx].tick_params(axis='x', which='major', labelbottom=True)
            # 设置 x 轴范围，确保数据从起点开始
            axs[idx].set_xlim(pd.to_datetime('2010-01-01'), pd.to_datetime('2010-12-31'))
            # 设置刻度格式
            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            # 设置刻度从2月、5月、8月、11月开始，每3个月一个刻度
            axs[idx].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 5, 8, 11)))
        else:
            axs[idx].tick_params(axis='x', which='major', labelbottom=False)
            # 设置每3个月一个刻度
            axs[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # 设置 y 轴范围和刻度
        axs[idx].set_ylim(1e-2, 1e2)
        axs[idx].set_yticks([1e-1, 1e0, 1e1])
        axs[idx].set_yticklabels(['0.1', '1', '10'])

        # 调整 y 轴刻度标签的对齐方式
        for label in axs[idx].get_yticklabels():
            label.set_horizontalalignment('center')  # 可根据需要调整为 'right' 或 'left'

        # 调整刻度参数，添加右侧刻度线，增加刻度长度和宽度，方向指向图内
        axs[idx].tick_params(
            axis='both',
            which='both',
            direction='in',
            length=8,  # 增加刻度线长度
            width=2,  # 增加刻度线宽度
            right=False,  # 如果需要启用右侧刻度线，设置为 True
            top=False,  # 如果不需要顶部刻度线，可以保持为 False
            labelsize=10
        )

        # 调整 y 轴刻度标签的距离
        axs[idx].tick_params(axis='y', which='both', pad=15)  # 调整标签与轴线的距离

        # 调整 x 轴刻度标签的位置，避免与 y 轴刻度标签冲突
        axs[idx].tick_params(axis='x', which='major', pad=5)  # 减少 pad

        # 添加图例
        axs[idx].legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=10)
        axs[idx].tick_params(axis='both', labelsize=10)

        # 设置时间序列子图边框厚度和颜色
        for spine in axs[idx].spines.values():
            spine.set_linewidth(edge_width)
            spine.set_color('black')  # 设置边框颜色为黑色

    # 调整子图的边距，避免刻度标签重叠
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0)

    # 调整整体字体大小
    set_fonts_for_fig(fig, scale_factor=4, legend_scale_factor=0.4, tick_scale_factor=0.6)

    # 保存和显示图形
    plt.savefig(f"{sp_fp}_combined.png", bbox_inches='tight')
    plt.show()


def plot_map_1(data_frequency, colorbar_title, sp_fp, extent=None):
    """
    绘制地图并保存为PNG文件。

    参数:
    - data_frequency: 数据集，包含经纬度和频率数据。
    - colorbar_title: 颜色条的标题。
    - sp_fp: 保存路径。
    - map_name: 地图名称，用于文件名。
    - extent: 地图范围，格式为 [西经, 东经, 南纬, 北纬]，单位为度。
              例如，中国范围可以设置为 [70, 140, 15, 55]。
              默认为None，表示全局范围。
    """
    cmap, colors = setup_colors()

    # 使用方正的 PlateCarree 投影并设置全局范围
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()

    # 添加全局海岸线
    ax.coastlines(resolution='110m')

    # 添加经纬度网格线并显示标签
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # 检查并添加数据
    if data_frequency is not None and not data_frequency.squeeze().size == 0:
        infile = data_frequency.squeeze()
        # 确保数据包含经纬度信息
        if not hasattr(infile, 'longitude') or not hasattr(infile, 'latitude'):
            raise AttributeError("数据集必须包含 'longitude' 和 'latitude' 属性。")

        tp, longitude = add_cyclic_point(infile, coord=infile.longitude)

        cont = ax.contourf(longitude, infile.latitude, tp, levels=20, cmap=cmap,
                           vmin=infile.min().compute().item(),
                           vmax=infile.max().compute().item(),
                           transform=ccrs.PlateCarree())

        # 设置颜色条，并添加标题
        cbar = plt.colorbar(cont, orientation='horizontal', pad=0.1, aspect=50, ax=ax)
        cbar.set_label(colorbar_title, fontsize=12)

    # 调整字体大小
    adjust_all_font_sizes(fig, scale_factor=1.2)

    # 保存并显示图像
    plt.savefig(f"{sp_fp}/map.png", bbox_inches='tight')
    plt.show()


def plot_distribution(data_percentile, data_frequency, sp_fp):
    plt.close()
    nbins = data_percentile.shape[0]
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, nbins)
    cmap, colors = setup_colors()

    # Make the figure wider by adjusting figsize (e.g., 15 inches wide instead of 10)
    fig, ax = plt.subplots(figsize=(15, 10))  # Increased width from 10 to 15

    # Remove the title
    # ax.set_title('Total Precipitation Distribution', fontsize=24, pad=20)

    # Add label with increased font size and better positioning
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    dist_arr = np.asarray(data_percentile)

    for ind, d in enumerate(dist_arr[0:100, :]):
        if isinstance(d, float):
            continue  # Ignore missing data
        else:
            color = colors[ind * max(1, round(len(colors) / nbins))]
            ax.plot(d, np.arange(1, 101), '.', color=color, markersize=12)  # Increased markersize from 10 to 12

    # Set labels with increased font sizes
    ax.set_ylabel('Percentile', fontsize=32)  # Increased from 15 to 20
    ax.set_xlabel('Cumulative Precipitation (mm/day)', fontsize=32)  # Increased from 15 to 20

    # Set tick parameters with larger font sizes
    ax.tick_params(axis='both', which='major', labelsize=16)  # Increased from default to 16

    # Customize y-ticks
    ax.set_yticks([1, 10, 25, 50, 75, 90, 99])
    ax.set_yticklabels([1, 10, 25, 50, 75, 90, 99], fontsize=24)  # Ensure y-tick labels are larger

    # Customize x-ticks
    ax.set_xticks([1, 10, 100, 500])
    ax.set_xticklabels([1, 10, 100, 500], fontsize=24)  # Ensure x-tick labels are larger

    # Enable grid with adjusted style
    ax.grid(ls="--", color='k', alpha=0.3)  # Reduced alpha for a subtler grid

    # Set logarithmic scale and limits
    ax.set_xscale("log")
    ax.set_xlim(1, 500)

    # Enhance spines (border lines) for better visibility
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Increased from default to 2

    # Optionally, set tick directions outward for better aesthetics
    ax.tick_params(direction='out')

    # Optionally, make sure the ticks are on the outside
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set image resolution
    # fig.set_dpi(150)  # Increased from 100 to 150 for better clarity

    # Optimize layout
    plt.tight_layout()

    # Save and display the image
    plt.savefig(sp_fp + 'distribution.png', bbox_inches='tight')  # Ensure dpi matches fig.set_dpi
    plt.show()


@auto_close_plot
def wdp_era5_lfp(data_frequency, data_percentile, sp_fp, colorbar_title):
    nbins = data_percentile.shape[0]
    vbins = np.linspace(data_frequency.min().values, data_frequency.max().values, nbins)
    cmap, colors = setup_colors()

    # 使用 GridSpec 来更好地控制子图布局
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)  # 调整宽度比和子图间距

    infile = data_frequency.squeeze()
    dist = data_percentile
    dist_arr = np.asarray(dist)

    # 图1参数设置 - 使用 EqualEarth 投影
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.EqualEarth())
    ax1.set_title(colorbar_title, fontsize=24, pad=20)
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax1.text(0.0, 1.0, 'a.', transform=ax1.transAxes + trans,
             fontsize='large', verticalalignment='top', fontfamily='sans-serif',
             weight='bold', color='black',
             bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax1.coastlines()

    # 设置经纬度网格线和标签
    gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 15}
    gl.ylabel_style = {'fontsize': 15}
    gl.xlocator = plt.FixedLocator(np.linspace(-180, 180, 7))
    gl.ylocator = plt.FixedLocator(np.linspace(-90, 90, 5))
    gl.xformatter = ccrs.cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = ccrs.cartopy.mpl.gridliner.LATITUDE_FORMATTER

    # 将图1点绘制到图上
    tp, longitude = infile, infile.longitude  # 连接经度数组的两端
    # tp, longitude = add_cyclic_point(infile, infile.longitude)  # 连接经度数组的两端
    for i, (lon, lat) in enumerate(onat_list):
        print(f'per:{i} time:{infile.sel(longitude=lon, latitude=lat, method="nearest").values}')
        ax1.scatter(reverse_convert_longitude(lon), lat, color='black', s=250, zorder=5,
                    transform=ccrs.PlateCarree())  # s是点的大小，zorder是图层顺序
        ax1.text(reverse_convert_longitude(lon), lat, str(i + 1), color='white',
                 ha='center', fontsize='18', va='center', zorder=6,
                 transform=ccrs.PlateCarree())  # 在点上添加编号

    # 绘制图1的内容
    masked_data = np.isnan(tp)
    cmap2 = ListedColormap(['none', 'grey'])
    cont = ax1.contourf(longitude, infile.latitude, tp, levels=20, cmap=cmap,
                        vmin=infile.min().compute().item(),
                        vmax=infile.max().compute().item(),
                        transform=ccrs.PlateCarree())
    ax1.contourf(longitude, infile.latitude, masked_data, levels=[0, 0.5, 1],
                 cmap=cmap2, transform=ccrs.PlateCarree())

    # 绘制第二幅图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Total Precipitation Distribution', fontsize=24, pad=20)
    ax2.text(0.0, 1.0, 'b.', transform=ax2.transAxes + trans,
             fontsize='large', verticalalignment='top', fontfamily='sans-serif',
             weight='bold', bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

    for ind, d in enumerate(dist_arr[0:100, :]):
        if isinstance(d, float):
            continue  # 忽略不存在的数据
        else:
            ax2.plot(d, np.arange(1, 101), '.', color=colors[ind * round(100 / nbins)],
                     markersize=10)

    ax2.set_ylabel('Percentile', fontsize=15)
    ax2.set_xlabel('Cumulative Precipitation (mm/day)', fontsize=15)
    ax2.set_yticks([1, 10, 25, 50, 75, 90, 99])
    ax2.grid(ls="--", color='k', alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 500)
    ax2.set_xticks([1, 10, 100, 500])
    ax2.set_xticklabels([1, 10, 100, 500])

    # 设置色条
    setup_colorbar(fig, vbins, cmap, orientation='horizontal', ax=ax1)

    # 调整所有字体大小
    adjust_all_font_sizes(fig, scale_factor=2.5)

    # 设置图像分辨率
    fig.set_dpi(100)  # 根据需要调整 DPI

    # 保存并显示图像
    plt.savefig(sp_fp + 'distribution.png', bbox_inches='tight')
    plt.show()


def get_cdict():
    cdict = {
        'red': ((0.0, inter_from_256(64), inter_from_256(64)),
                (1 / 5 * 1, inter_from_256(102), inter_from_256(102)),
                (1 / 5 * 2, inter_from_256(235), inter_from_256(235)),
                (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
                (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
                (1.0, inter_from_256(169), inter_from_256(169))),
        'green': ((0.0, inter_from_256(57), inter_from_256(57)),
                  (1 / 5 * 1, inter_from_256(178), inter_from_256(178)),
                  (1 / 5 * 2, inter_from_256(240), inter_from_256(240)),
                  (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
                  (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
                  (1 / 5 * 5, inter_from_256(23), inter_from_256(23))),
        'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
                 (1 / 5 * 1, inter_from_256(255), inter_from_256(255)),
                 (1 / 5 * 2, inter_from_256(185), inter_from_256(185)),
                 (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
                 (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
                 (1.0, inter_from_256(69), inter_from_256(69))),
    }
    return cdict


def wdp_era5_3percentile(lsprf_frequency, lspf_frequency, cp_frequency, sp_fp, colorbar_title):
    cdict = {
        'red': ((0.0, inter_from_256(64), inter_from_256(64)),
                (1 / 5 * 1, inter_from_256(102), inter_from_256(102)),
                (1 / 5 * 2, inter_from_256(235), inter_from_256(235)),
                (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
                (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
                (1.0, inter_from_256(169), inter_from_256(169))),
        'green': ((0.0, inter_from_256(57), inter_from_256(57)),
                  (1 / 5 * 1, inter_from_256(178), inter_from_256(178)),
                  (1 / 5 * 2, inter_from_256(240), inter_from_256(240)),
                  (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
                  (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
                  (1 / 5 * 5, inter_from_256(23), inter_from_256(23))),
        'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
                 (1 / 5 * 1, inter_from_256(255), inter_from_256(255)),
                 (1 / 5 * 2, inter_from_256(185), inter_from_256(185)),
                 (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
                 (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
                 (1.0, inter_from_256(69), inter_from_256(69))),
    }
    plt.close()
    all_area_num = 6
    cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=all_area_num)
    colors = cmap(np.linspace(0, 1, 100))
    fig = plt.figure(figsize=(13, 20), constrained_layout=True)
    fig.tight_layout()

    ax = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
    plt.title('large scale precipitation', fontsize=24)
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, 'a.large scale precipitation', transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.coastlines()

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

    plt.xlabel('Longitude', fontsize='20')
    plt.ylabel('Latitude', fontsize='20')
    tp, longitude = add_cyclic_point(lsprf_frequency, lsprf_frequency.longitude)  # connects the two ends of the longitude array
    cont = plt.contourf(longitude, lsprf_frequency.latitude, tp, levels=all_area_num, cmap=cmap, vmin=0, vmax=100)

    ax = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
    plt.title('LSP cover time', fontsize=24)
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, 'a.LSP cover time', transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.coastlines()

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

    plt.xlabel('Longitude', fontsize='20')
    plt.ylabel('Latitude', fontsize='20')
    tp, longitude = add_cyclic_point(lspf_frequency, lspf_frequency.longitude)  # connects the two ends of the longitude array
    cont = plt.contourf(longitude, lspf_frequency.latitude, tp, levels=all_area_num, cmap=cmap, vmin=0, vmax=100)

    ax = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
    plt.title('convective precipitation', fontsize=24)
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, 'a.convective precipitation', transform=ax.transAxes + trans,
            fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.coastlines()

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

    plt.xlabel('Longitude', fontsize='20')
    plt.ylabel('Latitude', fontsize='20')
    tp, longitude = add_cyclic_point(cp_frequency, cp_frequency.longitude)  # connects the two ends of the longitude array
    cont = plt.contourf(longitude, cp_frequency.latitude, tp, levels=all_area_num, cmap=cmap, vmin=0, vmax=100)
    cmap = plt.get_cmap(cmap, 20)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)
    clbar.set_label(colorbar_title, fontsize='16')
    plt.savefig('F:/liusch/remote_project/climate_new/temp_fig\\sp_fp\\' + sp_fp)
    plr.close()
    plt.show()


if __name__ == '__main__':
    figure_title = f'frequency'
    colorbar_title = f'frequency'
    figure_title_font = 24
    lat_range = 60
    # path_var = path_all + var
    path_out = "C:\\ERA5\\1980-2019\\outer_klag_rain\\"
    lsprf_frequency_path = xarray.open_dataset(f'{path_out}large_scale_precipitation_frequency_lat{lat_range}.nc').to_array().squeeze() * 100
    # tp_frequency_path = f'{path_out} large_scale_precipitation_fraction_frequency_lat{lat_range}.nc'
    lspf_frequency_path = xarray.open_dataset(f'{path_out}large_scale_precipitation_fraction_frequency_lat{lat_range}.nc').to_array().squeeze() * 100
    cp_frequency_path = xarray.open_dataset(f'{path_out}convective_precipitation_frequency_lat{lat_range}.nc').to_array().squeeze() * 100
    wdp_era5_3percentile(cp_frequency=cp_frequency_path, lsprf_frequency=lsprf_frequency_path, lspf_frequency=lspf_frequency_path,
                         sp_fp='3frequency', colorbar_title=colorbar_title)
