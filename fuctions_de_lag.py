import numpy as np
import xarray as xr
from function_percentile import lspf_percentile
from function_era5_narea_ptop_klag_1deg import era5_narea_ptop_klag_1deg
from function_draw_memory_topper_1deg_6area import dm_area_top
from function_wdp import wdp_era5

# 图形参数
figure_title = f'day-in-40years_6LSPF_180day_lag'
colorbar_title = f'Large Scale Precipitation Fraction (%)'
colorbar_title_lspf = f'Large Scale Precipitation Fraction (%) cover'
figure_title_font = 24
# 定义范围和点的数量
start = 1
end = 180
num_points = 30

# 计算对数底数
base = end ** (1 / (num_points - 1))  # 生成对数点并取最近的整数
log_points = np.unique([int(round(start * base ** n)) for n in range(num_points)])
# 输入参数
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"
path_lspf = 'E:\\ERA5\\1980-2019\\large_scale_precipitation_fraction\\'
path_lsp = 'E:\\ERA5\\1980-2019\\large_scale_precipitation\\'
path_out = "E:\\ERA5\\1980-2019\\outer_klag_rain\\"

dr_lat60 = xr.open_mfdataset(pathr + '*processed_day_1.nc')['tp'].sel(latitude=slice(60, -60))
lspf_lat60 = xr.open_mfdataset(path_lspf + '*processed_day_1.nc')['lspf'].sel(latitude=slice(60, -60))
lsp_lat60 = xr.open_mfdataset(path_lsp + '*processed_day_1.nc')['lsp'].sel(latitude=slice(60, -60))

# dr_all = xr.open_mfdataset(pathr + '*processed_day_1.nc')['tp']
# lspf_all = xr.open_mfdataset(path_lspf + '*processed_day_1.nc')['lspf']
# lsp_all = xr.open_mfdataset(path_lsp + '*processed_day_1.nc')['lsp']

# lsp_path = lsp_lat60
# dr_path = dr_lat60
# lspf_path = lspf_lat60
# 输出参数
# sp_frequency = 'lsprf_frequency.nc'
# sp_percentile = 'lsprf_percentile.npy'
# sp_frequency = 'lsprf_frequency_lat60.nc'
# sp_percentile = 'lsprf_percentile_lat60.npy'
sp_frequency = 'lspf_frequency_lat60.nc'
sp_percentile = 'lspf_percentile_lat60.npy'
bins, indices, data_percentile, data_frequency = lspf_percentile(dr=dr_lat60, lspf=lspf_lat60, sp_frequency=sp_frequency, sp_percentile=sp_percentile)
wdp_era5(data_frequency=data_frequency, data_percentile=data_percentile, colorbar_title=colorbar_title_lspf)

# wetday频率地理分布
# area6_percentile_lsprf_lat60 = np.load('lsprf_percentile_lat60.npy')
# lspf_frequency = xr.open_dataset('lspf_frequency.nc').to_array().values.squeeze()
# frequency = lspf_frequency
# 区域切分参量
# bins = np.load('bins.npy')
# indices = np.load('indices.npy')
# 前百分位数
# area6_percentile_lspf = np.load('lspf_percentile.npy')
# area6_percentile_lsprf = np.load('lsprf_percentile.npy')
# area6_percentile_lspf_lat60 = np.load('lspf_percentile_lat60.npy')
# # area_top_per_all_lsprf = area6_percentile_lsprf[:, selected_columns].T  # 只有5个区域
# area_top_per_all_lspf_lat60 = area6_percentile_lspf_lat60[:, selected_columns].T
# area_top_per_all = area_top_per_all_lspf
selected_columns = [49, 59, 69, 79]
area_top_per_all = data_percentile[:, selected_columns].T

# sp_out_all = f'{path_out}result_klag_1deg_6area_topper_lsprf.npy'
# sp_out_lat60 = f'{path_out}result_klag_1deg_6area_topper_lsprf_lat60.npy'
sp_out_lspf_lat60 = f'{path_out}result_klag_1deg_6area_topper_lspf_lat60.npy'
result_lag_percentile = era5_narea_ptop_klag_1deg(log_points=log_points, dr=dr_lat60, bins=bins, indices=indices, area_top_per_all=area_top_per_all, sp_out=sp_out_lspf_lat60)
# dm_in = sp_out
# all_area_num = len(bins)

# 输入draw_memory_topper-1deg-6area.py
# 输出draw_memory_topper-1deg-6area.py
# f1 = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-all6area_lsprf.png'
# f2 = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-allTopPer_lsprf.png'
# f3 = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-all6area_lsprf_lat60.png'
# f4 = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-allTopPer_lsprf_lat60.png'
dm_lagtime_lspf_lat60 = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-all6area_lspf_lat60.png'
dm_frequency_lspf_lat60 = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-allTopPer_lspf_lat60.png'
# sp_dm_lagtime = dm_latime_lspf
# sp_dm_frequency = dm_frequency_lspf
dm_area_top(bins=bins, area_top_per_all=area_top_per_all, selected_columns=selected_columns, dm_in=result_lag_percentile,
            sp_dm_frequency=dm_frequency_lspf_lat60, sp_dm_lagtime=dm_lagtime_lspf_lat60,
            figure_title=figure_title, figure_title_font=figure_title_font, colorbar_title=colorbar_title_lspf)
