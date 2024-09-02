import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
from matplotlib import colors as clr
import matplotlib.transforms as mtransforms
from cartopy.util import add_cyclic_point
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from lag_path_parameter import onat_list, onat_list_one


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
    plt.savefig('F:\\liusch\\remote_project\\climate_new\\temp_fig\\sp_fp\\' + sp_fp)
    plt.show()


def wdp_era5_geography(data_frequency, sp_fp, colorbar_title):
    cdict = get_cdict()  # 假设这个函数在其他地方定义
    all_area_num = 6
    cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=all_area_num)

    infile = data_frequency.squeeze()
    tp, longitude = add_cyclic_point(infile, infile.longitude)

    fig, ax1 = plt.subplots(figsize=(13, 10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
    ax1.set_title(colorbar_title, fontsize=24)

    # 添加图标标记
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax1.text(0.0, 1.0, 'a.', transform=ax1.transAxes + trans,
             fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold', color='black',
             bbox=dict(facecolor='white', edgecolor='none', pad=1.0))

    ax1.coastlines()
    ax1.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    plt.xlabel('Longitude', fontsize=20)
    plt.ylabel('Latitude', fontsize=20)

    # 绘制数据
    masked_data = np.isnan(tp)
    cmap_masked = ListedColormap(['none', 'grey'])
    cont = plt.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap,
                        vmin=infile.min().compute().item(), vmax=infile.max().compute().item())
    plt.contourf(longitude, infile.latitude, masked_data, levels=[0, 0.5, 1], cmap=cmap_masked)

    # 添加 colorbar
    norm = mpl.colors.Normalize(vmin=data_frequency.min().compute().item(), vmax=data_frequency.max().compute().item())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    clbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.1, aspect=30, shrink=0.9)
    clbar.set_label('', fontsize=24)

    plt.savefig(sp_fp + 'distribution.png')
    plt.show()


def wdp_era5_lfp(data_frequency, data_percentile, sp_fp, colorbar_title):
    cdict = get_cdict()
    all_area_num = data_percentile.shape[0]
    cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=all_area_num)
    colors = cmap(np.linspace(0, 1, 100))
    # infile = data_frequency.squeeze().compute()
    infile = data_frequency.squeeze()
    dist = data_percentile
    dist_arr = np.asarray(dist)

    fig = plt.figure(figsize=(13, 20), constrained_layout=True)
    fig.tight_layout()

    # 图1参数设置
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    plt.title(colorbar_title, fontsize=24)
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax1.text(0.0, 1.0, 'a.', transform=ax1.transAxes + trans,
             fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold', color='black',
             bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax1.coastlines()
    ax1.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    plt.xlabel('Longitude', fontsize='20')
    plt.ylabel('Latitude', fontsize='20')

    # 将图1点绘制到图上
    tp, longitude = add_cyclic_point(infile, infile.longitude)  # connects the two ends of the longitude array
    for i, (lon, lat) in enumerate(onat_list):
        # print(f'per:{i} time:{infile.sel(longitude=lon, latitude=lat, method="nearest").values}')
        plt.scatter(reverse_convert_longitude(lon), lat, color='black', s=250, zorder=5)  # s是点的大小，zorder是图层顺序，确保点在最上面
        plt.text(reverse_convert_longitude(lon), lat, str(i + 1), color='white', ha='center', fontsize='18', va='center', zorder=6)  # 在点上添加编号

    # 绘制图1的内容
    masked_data = np.isnan(tp)
    cmap2 = ListedColormap(['none', 'grey'])
    # cont = plt.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap, vmin=0, vmax=infile.max().compute().item())
    cont = plt.contourf(longitude, infile.latitude, tp, levels=all_area_num, cmap=cmap, vmin=infile.min().compute().item(), vmax=infile.max().compute().item())
    plt.contourf(longitude, infile.latitude, masked_data, levels=[0, 0.5, 1], cmap=cmap2)

    # 绘制第二幅图
    ax = plt.subplot(2, 1, 2)
    plt.title('total precipitation distribution', fontsize=24)
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

    # 添加 colorbar
    cmap = plt.get_cmap(cmap, 20)
    # norm = mpl.colors.Normalize(vmin=0, vmax=data_frequency.max().compute().item())
    norm = mpl.colors.Normalize(vmin=data_frequency.min().compute().item(), vmax=data_frequency.max().compute().item())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.subplots_adjust(left=0.05, right=0.85)
    clbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.1, aspect=30, shrink=0.9)
    clbar.set_label(colorbar_title, fontsize='24')
    plt.savefig(sp_fp + 'distribution.png')
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
    plt.savefig('F:\\liusch\\remote_project\\climate_new\\temp_fig\\sp_fp\\' + sp_fp)
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
