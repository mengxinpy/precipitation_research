import numpy as np
from workflow import point_path_data
import xarray as xr
from utils import get_refine_da_list
from plot_lib import scatter_plots_depart, plt_duration_origin, plt_duration_combined, scatter_plots_combined_final, scatter_plots_combined_verify_turning_points
from utils import depart_ml_lat
from plot_lib import single_scatter_plot
from plot_lib import plt_duration
from workflow import load_data
from config import intern_data_path
data_set_list = ['era5', 'mswep', 'persiann']
data_set = data_set_list[1]
dr = point_path_data('total_precipitation', data_set)
data_dict = xr.open_dataset('./internal_data/data_dict.nc')

# %%fig1
data_percentile, bins, indices = load_data(f'internal_data/wet_{data_set}/wet_data.npz')
bins_6=bins
duration_hist=np.load(f'internal_data/wet_{data_set}/ltp_duration_newest.npy',allow_pickle=True)
# 调用新函数一次性绘制所有子图
fig, axes = plt_duration_combined(duration_hist, title='Duration', vbins=bins_6, fig_name=f'fig/sts_duration_combined_{data_set}.svg')

# %%fig2
from map_lib import wdp_era5_geography
dataset = xr.open_dataset('./internal_data/data_dict.nc')
dr = point_path_data('total_precipitation')
duration=dataset['duration']
power=dataset['power']
duration = duration.rename(r'Duration $\log_{10}\,(\mathrm{day})$')
power = power.rename('Yearly spectrum')
wdp_era5_geography(power,'./fig/Yearly_power.svg')
wdp_era5_geography(duration,'./fig/duration.svg')
# %%fig3
from map_lib import wdp_era5_geography
dataset = xr.open_dataset('./internal_data/data_dict.nc')
cape_frequency=xr.open_dataarray(f'w850_frequency.nc',engine='netcdf4').coarsen(longitude=4,latitude=4,boundary='trim').mean().sel(latitude=slice(60,-60))
w850=xr.open_dataarray(f'cape_frequency.nc',engine='netcdf4').coarsen(longitude=4,latitude=4,boundary='trim').mean().sel(latitude=slice(60,-60))

w850 = w850.rename('CAPE')/100
cape_frequency = cape_frequency.rename('Vertical velocity')/100
wdp_era5_geography(w850,'./fig/w850.svg')
wdp_era5_geography(cape_frequency,'./fig/cape.svg')

# %%fig4
from map_lib import wdp_era5_geography
wind_10=xr.open_dataarray(f'/Users/kustai/PycharmProjects/ERA5/data_0.nc').coarsen(longitude=4,latitude=4,boundary='trim').mean().sel(latitude=slice(60,-60))
wind_10=wind_10.mean(dim='date')
wind_10 = wind_10.rename('Mean wind speed (m/s)')
wdp_era5_geography(wind_10,'./fig/wind.svg')
# %%fig6
dataset = xr.open_dataset('./internal_data/data_dict.nc')
w850=xr.open_dataarray(f'w850_frequency.nc',engine='netcdf4')
cape_frequency=xr.open_dataarray(f'cape_frequency.nc',engine='netcdf4')

key_list = ['duration', 'wet']
data_list = get_refine_da_list(key_list, log_duration_etc=True, unify=False,wet_th=True)
data_list.append(w850)
data_list.append(cape_frequency)
low_lat_list, mid_lat_list = depart_ml_lat(data_list)
scatter_plots_depart(low_lat_list, mid_lat_list, save_path='./fig/wet_duration_si10_')
# %%fig7

dataset='era5'
key_list = ['duration', 'wet']
data_list = get_refine_da_list(['top20%','wet'], dataset=data_set, log_duration_etc=False, unify=False, wet_th=True)
data_list[0]=data_list[0].rename('Duration (day)')

low_lat_list, mid_lat_list = depart_ml_lat(data_list)

# 调用修改后的函数，添加range_filter参数
kb_mid = scatter_plots_combined_verify_turning_points(
    low_lat_list,
    mid_lat_list,
    save_path=f'./fig/wet_{data_set}/supplement-fig-7',
)
