import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 假设你已经有了经度和纬度的数据
lon = np.linspace(-180, 180, 1440)  # 经度范围
lat = np.linspace(-90, 90, 720)  # 纬度范围

# 假设你的数据是这样的：
data = np.random.rand(181, 361)  # 随机生成一些数据
data[:60, :] = np.nan  # 将纬度大于20的部分设为nan
data[-60:, :] = np.nan  # 将纬度小于-20的部分设为nan

# 创建图形和坐标轴
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

# 画热度图
c = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), cmap='Spectral_r')

# 添加颜色条
fig.colorbar(c, ax=ax)

# 显示图形
plt.show()
