import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr
import general_graph_setting


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
    plt.close()


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
        era5_draw_area_dataArray(one_day_data, f'{sp}all_area_amsr2\\{name}')
