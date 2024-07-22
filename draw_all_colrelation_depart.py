import numpy as np

from plt_temp import scatter_plots, scatter_plots_depart
from lag_path_parameter import path_out
import xarray as xr

var_list = [[], [], []]
key_list = ['k', 'qk', 'duration', 'quiet', 'power', 'wet', 'dfa']
low_lat_list = []
mid_lat_list = []
for ind, key in enumerate(key_list):
    file_name = f'wetday_vt_{key}_frequency_lat60.nc'
    data = xr.open_dataarray(path_out + file_name)

    low_lat = data.sel(latitude=slice(30, -30))

    south_hemisphere = data.sel(latitude=slice(-30, -60))
    north_hemisphere = data.sel(latitude=slice(60, 30))
    # 合并南北半球的切片
    combined = xr.concat([north_hemisphere, south_hemisphere], dim="latitude")
    # 确保纬度按升序排列
    mid_lat = combined.sortby('latitude')

    if key in ['duration', 'quiet']:
        low_lat = np.log(low_lat)
        mid_lat = np.log(mid_lat)
    low_lat_list.append(low_lat)
    mid_lat_list.append(mid_lat)
scatter_plots_depart(matrices_low=low_lat_list, matrices_mid=mid_lat_list, var_names=key_list, figure_name=f'all_correlation_depart')
#
#
# for ind_v, var in enumerate(var_list):
#     for ind, key in enumerate(key_list):
#         file_name = f'wetday_vt_{key}_frequency_lat60.nc'
#         data = xr.open_dataarray(path_out + file_name)
#         if ind_v == 0:
#             combined = data.sel(latitude=slice(60, -60))
#         elif ind_v == 1:
#             combined = data.sel(latitude=slice(30, -30))
#         else:
#             south_hemisphere = data.sel(latitude=slice(-30, -60))
#
#             # 切片北半球纬度 30 到 60
#             north_hemisphere = data.sel(latitude=slice(60, 30))
#
#             # 合并南北半球的切片
#             combined = xr.concat([north_hemisphere, south_hemisphere], dim="latitude")
#
#             # 确保纬度按升序排列
#             combined = combined.sortby('latitude')
#         if key in ['duration', 'quiet']:
#             combined = np.log(combined)
#         var.append(combined)
#     scatter_plots(var, var_names=key_list, figure_name=f'correlation_{ind_v}.png')
