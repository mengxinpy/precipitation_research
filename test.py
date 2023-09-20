import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from plt_temp import *

wetday = np.flipud(np.roll(np.load('_wetday.npy') / 365,720,axis=1))
# plt.imshow(wetday)
# draw_area_heap(wetday)
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
lon = np.linspace(-180, 180, 1440)
lat = np.linspace(-90, 90, 721)
Lon, Lat = np.meshgrid(lon, lat)
c = ax.contourf(Lon, Lat, wetday, transform=ccrs.PlateCarree(), levels=20, cmap='jet')
fig.colorbar(c, ax=ax)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False  # 关闭顶部的经度标签
gl.right_labels = False  # 关闭右侧的纬度标签
plt.show()
