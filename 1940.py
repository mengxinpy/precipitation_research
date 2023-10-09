#%%
import numpy as np
import matplotlib.path as mpath
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 读取数据
ds = xr.open_dataset(r'F:\zhaodan\1940t_merged.nc')
# ds = xr.open_dataset(r'E:\ERA5\Run_data2\merge\10\t_merged.nc')
# 提取温度数据
temperature = ds['t'].sel(time=slice('1940-01-01', '1940-12-31'))
temperature=temperature.squeeze()
# 计算标准差
std_dev = temperature.std(dim='time')
std_dev = std_dev.squeeze()


# 创建投影
lons = np.linspace(-180, 180, num=144)
lats = np.linspace(0,90, num=37)
fig=plt.figure(figsize=(10,10),dpi=300)
ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.75],projection = ccrs.NorthPolarStereo())
# proj =ccrs.NorthPolarStereo(central_longitude=90)
ax1.set_extent([-180,180,0,90],ccrs.PlateCarree())
# ax1.gridlines()  添加栅格线
ax1.coastlines(alpha=1,linewidth=2)
# add features
ax1.add_feature(cfeature.LAND)
ax1.add_feature(cfeature.OCEAN)
ax1.add_feature(cfeature.COASTLINE)
# ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))

#纬度和经度标注和添加栅格线
gl = ax1.gridlines(draw_labels=True, linewidth=2,linestyle='--' ,alpha=1)
# gl.xlabels_top = False ##关闭上侧坐标显示
# gl.ylabels_right = False ##关闭右侧坐标显示
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 60))
gl.ylocator = mticker.FixedLocator(np.arange(-60, 90, 30))
gl.xlabel_style={'size':10}
gl.ylabel_style={'size':10}

#######以下为网格线的参数######
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)

###颜色条
c7 = ax1.contourf(std_dev.longitude,std_dev.latitude,std_dev, zorder=0, extend='both',cmap = plt.cm.RdBu_r, transform=ccrs.PlateCarree())
cb = plt.colorbar(c7, ax=ax1, orientation='vertical', pad=0.03, fraction=0.02)
# cb.set_label('Standard Deviation of Temperature', rotation=270, labelpad=15)  # 可根据实际情况修改颜色条的标签

ax1.set_title('1940-2022(all month) temperature sigma at 10 hPa' ,fontsize=30 , y = 1.05)
plt.show()
#%%
