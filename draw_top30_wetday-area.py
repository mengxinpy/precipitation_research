import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import dask as da
import xarray as xr
import numpy as np

pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"

# vapor_month = xr.open_mfdataset(pathv + '198001_processed_day_0.25.nc')
rain_month = xr.open_mfdataset(pathr + '198001_processed_day_0.25.nc')
for i in range(31):
    print(f'png:{i}')
    vapor = vapor_month.isel(time=i)['tcwv']
    rain = rain_month.isel(time=i)['tp']
    # 使用 apply_ufunc 应用函数
    modified_vapor = xr.apply_ufunc(select_top_30_percent, vapor, dask='parallelized', output_dtypes=[float])
    modified_rain = xr.apply_ufunc(select_top_30_percent, rain, dask='parallelized', output_dtypes=[float])  # 使用where方法找出两个DataArray中哪些位置的元素至少有一个为nan
    difference = vapor.where(~np.isnan(modified_vapor) & np.isnan(modified_rain))
    difference_rain = rain.where(~np.isnan(modified_vapor) & np.isnan(modified_rain))
    # vapor_maximum_part = maximum_part_of_matrix(vapor)
    # rain_maximum_part = maximum_part_of_matrix(rain)
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    plt.title('vapor', fontsize=18)
    # trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    plt.xlabel('Longitude', fontsize='15')
    plt.ylabel('Latitude', fontsize='15')
    cont = plt.contourf(modified_vapor.longitude, modified_vapor.latitude, modified_vapor, cmap='jet')
    plt.colorbar()
    ax = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    plt.title('rain', fontsize=18)
    # trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    plt.xlabel('Longitude', fontsize='15')
    plt.ylabel('Latitude', fontsize='15')
    cont = plt.contourf(modified_rain.longitude, modified_rain.latitude, modified_rain, cmap='jet')
    plt.colorbar()
    ax = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    # trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    plt.xlabel('Longitude', fontsize='15')
    plt.ylabel('Latitude', fontsize='15')
    # vapor_normalized = (modified_vapor - modified_vapor.min()) / modified_vapor.max() - modified_vapor.min()
    # rain_normalized = (modified_rain - modified_rain.min()) / modified_rain.max() - modified_rain.min()
    # difference = vapor_normalized - rain_normalized
    cont = plt.contourf(difference.longitude, difference.latitude, difference, cmap='jet')
    plt.colorbar()
    ax = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    plt.xlabel('Longitude', fontsize='15')
    plt.ylabel('Latitude', fontsize='15')
    rain = rain_month.isel(time=i)['tp']
    cont = plt.contourf(difference_rain.longitude, difference_rain.latitude, difference_rain, cmap='jet')
    plt.colorbar()
    plt.savefig(f'.\\temp_fig\\vapor-rain-difference\\top30-200001{str(i + 1).zfill(2)}')
