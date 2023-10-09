import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from plt_temp import *

import plotly.graph_objects as go


# 创建一个文本用于显示数据值

# 定义事件处理函数



# 连接事件处理函数

wetday = np.flipud(np.roll(np.load('_wetday.npy') / (365 * 10 + 3), 720, axis=1))
# plt.imshow(wetday)
# draw_area_heap(wetday)

lon = np.linspace(-180, 180, 1440)
lat = np.linspace(-90, 90, 721)
Lon, Lat = np.meshgrid(lon, lat)
# 降采样，每10个数据点取一个
lon = lon[::10]
lat = lat[::10]
wetday = wetday[::10, ::10]
fig = go.Figure(data=go.Contour(x=lon, y=lat, z=wetday, colorscale='Rainbow'))
fig.show()
# 阻止脚本退出

# 关闭交互模式
