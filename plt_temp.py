import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr


# 获取一个颜色映射
# cmap = cm.get_cmap('viridis')
def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


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

# all_area_num = data_percentile.shape[0]
cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=6)  # todo:未参数化
colors = cmap(np.linspace(0, 1, 6))


# infile = xarray.open_dataset(data_frequency).squeeze() * 100


def draw_area_heap(matrix, name):
    matrix2 = np.flipud(np.roll(matrix, 180, axis=1))
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    lon = np.linspace(-180, 180, 360)
    lat = np.linspace(-90, 90, 120)
    Lon, Lat = np.meshgrid(lon, lat)
    c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap='Reds')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.savefig(".\\temp_fig\\" + str(name) + '.png')
    plt.close()


def draw_area_heap_cover(matrix, cover, name):
    colors = ['Blues', 'green', 'red']
    matrix2 = np.flipud(np.roll(matrix, 180, axis=1))
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    lon = np.linspace(-180, 180, 360)
    lat = np.linspace(-60, 60, 120)
    Lon, Lat = np.meshgrid(lon, lat)
    c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap=colors[0])
    fig.colorbar(c, ax=ax)
    # 计算每个循环迭代应该使用的颜色映射索引
    # 计算每个循环迭代应该使用的颜色映射索引
    # colors = [cmap(i / len(cover)) for i in range(len(cover))]
    # colors = [cmap(i) for i in colors]
    for ind, cv in enumerate(cover):
        cv = np.where(cv == 0, np.nan, cv)  # 添加这一行，将0值替换为np.nan
        cv = np.flipud(np.roll(cv, 180, axis=1))
        c = ax.contourf(Lon, Lat, cv, transform=ccrs.PlateCarree(), colors=colors[ind + 1])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.savefig(".\\temp_fig\\fig\\" + str(name) + '.png')
    plt.close()


def era5_draw_area_dataArray(dataArray, name):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    c = ax.contourf(dataArray.longitude, dataArray.latitude, dataArray, transform=ccrs.PlateCarree(), levels=20, cmap='rainbow')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    plt.savefig(".\\temp_fig\\" + str(name) + '.png')


def draw_area_heap_1deg(matrix, name):
    matrix2 = np.flipud(np.roll(matrix, 720, axis=1))
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    lon = np.linspace(-180, 180, 1440)
    lat = np.linspace(-90, 90, 721)
    Lon, Lat = np.meshgrid(lon, lat)
    c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap='rainbow')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.savefig(".\\temp_fig\\" + str(name) + '.png')


def test_3matrix(raw, area_wetday, lag):
    raw[area_wetday] = -999
    raw[lag] = np.nan
    matrix2 = np.flipud(np.roll(matrix, 720, axis=1))
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    lon = np.linspace(-180, 180, 1440)
    lat = np.linspace(-90, 90, 721)
    Lon, Lat = np.meshgrid(lon, lat)
    c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap='rainbow')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.savefig(".\\temp_fig\\" + str(name) + '.png')


def test_plot(data_pairs):
    """
    绘制多组数据的函数。每组数据由一对 x 和 y 组成。
    :param data_pairs: 包含多组 (x, y) 数据的列表。
    """
    plt.close()
    temp_mark = ['o', 's', 'o']
    plt.figure(figsize=(10, 6))
    colors = ['yellow', 'black', 'blue']
    # 为每组数据绘制一条线，并标记
    for i, (x, y) in enumerate(data_pairs):
        if i == 0:
            plt.plot(x, y, label=f'Line {i + 1}', marker=temp_mark[i], color=colors[i], linestyle="none", markerfacecolor="none")
        else:
            plt.plot(x, y, label=f'Line {i + 1}', marker=temp_mark[i], color=colors[i], linestyle="none")
    plt.xscale('log')  # 设置 x 轴为对数刻度
    plt.xlabel('X-axis (log scale)')
    plt.ylabel('Y-axis')
    plt.title('Custom Plot with Multiple Data Sets')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.show()


def plt_duration(durations, fig_name):
    fig = plt.figure()

    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])
    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm)
    clbar.set_label('Area (frequency)', fontsize='16')
    # 设置 bins 的边界为对数刻度
    for ind, dur in enumerate(durations):
        bins = np.unique(np.logspace(np.log10(min(dur)), np.log10(max(dur)), 30).round())

        # 计算直方图数据，density=True 以获取频率
        hist, bin_edges = np.histogram(dur, bins=bins, density=True)

        # 计算 bin 中心点，用于作为 x 轴数据
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # 绘制线图表示概率分布
        plt.plot(bin_centers, hist, '*', color=colors[ind])

    # 设置图表的标题和坐标轴标签
    plt.title('Probability Distribution of Durations')
    plt.xlabel('Duration(day)')
    plt.ylabel('Probability')

    # 添加网格线
    plt.grid(axis='y', alpha=0.75)

    # 设置y轴为对数刻度
    plt.yscale('log')
    # 如果你需要，也可以设置x轴为对数刻度
    plt.xscale('log')

    plt.savefig(f'.\\temp_fig\\durations_time\\{fig_name}')
    plt.close()
# def draw_area_heap_ax(matrix, name):
#     matrix2 = np.flipud(np.roll(matrix, 720, axis=1))
#     fig = plt.figure(figsize=(10, 5))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
#     lon = np.linspace(-180, 180, 1440)
#     lat = np.linspace(-90, 90, 721)
#     Lon, Lat = np.meshgrid(lon, lat)
#     c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap='rainbow')
#     fig.colorbar(c, ax=ax)
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
#     gl.top_labels = False  # 关闭顶部的经度标签
#     gl.right_labels = False  # 关闭右侧的纬度标签
#     ax = plt.savefig(".\\temp_fig\\" + str(name) + '.png')
#     return ax
