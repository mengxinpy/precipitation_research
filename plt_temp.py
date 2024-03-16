import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.cm as cm

# 获取一个颜色映射
cmap = cm.get_cmap('viridis')


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
