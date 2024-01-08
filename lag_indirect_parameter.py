import numpy as np
import xarray as xr

# wetday频率地理分布
# lspf_frequency = xr.open_dataset('lspf_frequency.nc').to_array().values.squeeze()
# frequency = lspf_frequency
# 区域切分参量
bins = np.load('bins.npy')
indices = np.load('indices.npy')
# 前百分位数
selected_columns = [49, 59, 69, 79]
area6_percentile_lspf = np.load('lspf_percentile.npy')[:, selected_columns].T  # 只有5个区域
area6_percentile_lsprf = np.load('lsprf_percentile.npy')[:, selected_columns].T  # 只有5个区域
area6_percentile_lsprf_lat60 = np.load('lsprf_percentile_lat60.npy')[:, selected_columns].T  # 只有5个区域
area6_percentile_lspf_lat60 = np.load('lspf_percentile_lat60.npy')[:, selected_columns].T  # 只有5个区域

path_out = "E:\\ERA5\\1980-2019\\outer_klag_rain\\"
sp_out_all = f'{path_out}result_klag_1deg_6area_topper_lsprf.npy'
sp_out_lat60 = f'{path_out}result_klag_1deg_6area_topper_lsprf_lat60.npy'
sp_out_lspf_lat60 = f'{path_out}result_klag_1deg_6area_topper_lspf_lat60.npy'
sp_out_lsprf_lat60 = f'{path_out}result_klag_1deg_6area_topper_lsprf_lat60.npy'
dm_lagtime_lsprf = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-all6area_lsprf_lat60.png'
dm_frequency_lsprf = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-allTopPer_lsprf_lat60.png'
# 11111111111111111111111111111111111111111111111111111111111111111111111111111111
# 输入era5_narea_ptop_klag-1deg.py
bins = bins
indices = indices
area_top_per_all = area6_percentile_lsprf_lat60
# 输出era5_narea_ptop_klag-1deg.py
sp_out = sp_out_lsprf_lat60

# 22222222222222222222222222222222222222222222222222222222222222222222222222222222
# 输入draw_memory_topper-1deg-6area.py
bins = bins
selected_columns = selected_columns
dm_in = sp_out
all_area_num = len(bins)

figure_title = f'day-in-40years_6LSPF_180day_lag'
colorbar_title = f'Large Scale Precipitation Fraction (%)'
figure_title_font = 24
# 输出draw_memory_topper-1deg-6area.py

sp_dm_lagtime = dm_lagtime_lsprf
sp_dm_frequency = dm_frequency_lsprf

# 33333333333333333333333333333333333333333333333333333333333333333333333333333333
# 输入draw_distribution_topper-1deg-6area.py
# bins = bins
# dm_in = sp_out
# figure_title = f'day-in-40years_6LSPF_180day_lag_distribution'
# colorbar_title = f'Large Scale Precipitation Fraction (%)'
# figure_title_font = 24

# colorbar_title_lspf = f'Large Scale Precipitation Fraction (%) cover'
# 输出draw_memory_topper-1deg-6area.py
# f1 = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-all6area_lsprf.png'
# f2 = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-allTopPer_lsprf.png'
# dm_latime_lspf = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-all6area_lspf_lat60.png'
# dm_frequency_lspf = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-allTopPer_lspf_lat60.png'
# dm_lagtime_lsprf = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-all6area_lsprf_lat60.png'
# dm_frequency_lsprf = f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-allTopPer_lsprf_lat60.png'
#
# sp_dm_lagtime = dm_lagtime_lsprf
# sp_dm_frequency = dm_frequency_lsprf
