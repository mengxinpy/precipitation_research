import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def draw_area_heap(matrix):
    # 创建图形和坐标轴
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.coastlines()
    c = ax.contourf(matrix, transform=ccrs.PlateCarree(), levels=20, cmap='jet')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.show()
    # plt.savefig('/temp_fig')
