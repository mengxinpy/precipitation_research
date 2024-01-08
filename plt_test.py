import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def draw_area_heap(matrix, name):
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
