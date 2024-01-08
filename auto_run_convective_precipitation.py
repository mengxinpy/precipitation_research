import xarray as xr
from function_percentile import cp_percentile
from function_era5_narea_ptop_klag_1deg import era5_narea_ptop_klag_1deg
from function_draw_memory_topper_1deg_6area import dm_area_top
from function_wdp import wdp_era5
import numpy as np
import os

filename = os.path.splitext(os.path.basename(__file__))[0]

var = filename.split('auto_run_')[-1]

figure_title = f'day-in-40years_{var}_180day_lag'
colorbar_title = f'{var} fraction (%)'
figure_title_font = 24
lat_range = 60
start = 1
end = 180
num_points = 30

base = end ** (1 / (num_points - 1))  # 生成对数点并取最近的整数
log_points = np.unique([int(round(start * base ** n)) for n in range(num_points)])
path_out = "E:\\ERA5\\1980-2019\\outer_klag_rain\\"
path_png = f'F:\\liusch\\remote_project\\climate_new\\temp_fig\\ear5_lag_area\\'
path_all = 'E:\\ERA5\\1980-2019\\'
path_var = path_all + var


def point_path_data(var, lat):
    path_all = f'E:\\ERA5\\1980-2019\\{var}\\'
    data_point = xr.open_mfdataset(path_all + '*processed_day_1.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


sp_frequency = f'{var}_frequency_lat{lat_range}.nc'
sp_percentile = f'{var}_percentile_lat{lat_range}.npy'
bins, indices, data_percentile, data_frequency = cp_percentile(dr=point_path_data('total_precipitation', lat=lat_range), cp=point_path_data(var, lat=lat_range)  # 路径参数
                                                               , sp_frequency=sp_frequency, sp_percentile=sp_percentile)  # 输出参数

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
