import xarray as xr
from function_percentile import random_percentile
from function_era5_narea_ptop_klag_1deg import era5_narea_ptop_klag_1deg
from function_draw_memory_topper_1deg_6area import dm_area_top
from function_wdp import wdp_era5
from lag_path_parameter import log_points, path_all, path_out, path_png
import numpy as np
import os

# 大尺度降水的覆盖时间
filename = os.path.splitext(os.path.basename(__file__))[0]

var = filename.split('auto_run_')[-1]

figure_title = f'day-in-40years_{var}_180day_lag'
colorbar_title = f'{var} cover time (%)'
figure_title_font = 24
lat_range = 60
path_var = path_all + var


def point_path_data(var, lat):
    path_all = f'C:\\ERA5\\1980-2019\\{var}\\'
    data_point = xr.open_mfdataset(path_all + '*processed_day_1.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


sp_frequency = f'{path_out}{var}_frequency_lat{lat_range}.nc'
sp_percentile = f'{path_out}{var}_percentile_lat{lat_range}.npy'
bins, indices, data_percentile, data_frequency = random_percentile(dr=point_path_data('total_precipitation', lat=lat_range),
                                                                   original_dataarray=point_path_data('total_precipitation', lat=60)[0])  # 输出参数

wdp_era5(data_frequency=data_frequency, data_percentile=data_percentile, sp_fp=var, colorbar_title=colorbar_title)

selected_columns = [49, 59, 69, 79]
area_top_per_all = data_percentile[:, selected_columns].T

ltp_out = f'{path_out}ltp_all_{var}_lat60.npy'
result_ltp = era5_narea_ptop_klag_1deg(log_points=log_points, dr=point_path_data('total_precipitation', lat=lat_range), bins=bins, indices=indices,
                                       area_top_per_all=area_top_per_all,
                                       sp_out=ltp_out)

dm_lagtime = f'{path_png}tmp_{var}_{lat_range}.png'
dm_frequency = f'{path_png}amt_{var}_{lat_range}.png'
dm_area_top(bins=bins, log_points=log_points, area_top_per_all=area_top_per_all, selected_columns=selected_columns, dm_in=result_ltp,
            sp_dm_frequency=dm_frequency, sp_dm_lagtime=dm_lagtime,
            figure_title=figure_title, figure_title_font=figure_title_font, colorbar_title=colorbar_title)
