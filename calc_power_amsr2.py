import numpy as np
import xarray as xr
from config import *
import os
import dask

path = "F:\\liusch\\amsr2\\"
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('processed_clat.nc')]

wetday = xr.open_dataset('era5_frequency.nc')['tp'].sel(latitude=slice(flat, -flat))
wetday = wetday.coarsen(longitude=int(deg / 0.25), latitude=int(deg / 0.25), boundary='trim').mean()
wetday_bins = np.arange(0, np.max(wetday), wetday_gap)  # num是你想要的分组数量
vapor_bins = np.linspace(vapor_strat, vapor_end, num=vapor_bins_count)  # num是你想要的分组数量
results = np.zeros((len(wetday_bins), vapor_bins_count))
count = np.zeros((len(wetday_bins), vapor_bins_count))
results_p4 = np.zeros((len(wetday_bins), vapor_bins_count))
results_p2 = np.zeros((len(wetday_bins), vapor_bins_count))

for i, file in enumerate(nc_files):
    # 在这里，i是文件的索引，file是文件本身
    print(f"处理第{i}个文件: {file}")
    # 在这里执行您需要的操作
    ds = xr.open_dataset(file)
    water_vapor = ds['water_vapor'].values
    rain_rate = ds['rain_rate'].values
    effective_region = (0 <= water_vapor) & (water_vapor < 120) & (rain_rate > -1)
    water_vapor = water_vapor[effective_region]
    rain_rate = rain_rate[effective_region]
    # 检查哪些位置是NaN
    # nan_mask = np.isnan(water_vapor)
    vapor_indices = np.digitize(water_vapor, vapor_bins)
    # 使用numpy.where将mask为1的地方置为numpy.nan
    # vapor_indices = np.where(nan_mask, np.nan, vapor_indices)
    # vapor_indices = xr.apply_ufunc(np.digitize, water_vapor, vapor_bins)
    rainfall_sum = [np.sum(rain_rate[vapor_indices == j]) for j in range(1, len(vapor_bins) + 1)]
    rainfall_p4_sum = [np.sum(rain_rate[vapor_indices == j] ** 4) for j in range(1, len(vapor_bins) + 1)]
    rainfall_p2_sum = [np.sum(rain_rate[vapor_indices == j] ** 2) for j in range(1, len(vapor_bins) + 1)]
    count_sum = [np.sum(vapor_indices == j) for j in range(1, len(vapor_bins) + 1)]

    results = results + rainfall_sum
    results_p2 = results_p2 + rainfall_p2_sum
    results_p4 = results_p4 + rainfall_p4_sum
    count = count + count_sum
avg_rain = results / count
second_moment = results_p2 / count
binder_4th_order_cumulant = 1 - (results_p4 / count) / (3 * second_moment ** 2)
var = second_moment - avg_rain ** 2
# avg_rain = rainfall_sum / count_sum
# second_moment = rainfall_p2_sum / count_sum
# binder_4th_order_cumulant = 1 - (rainfall_p4_sum / count_sum) / (3 * second_moment ** 2)
# var = second_moment - avg_rain ** 2
np.save(f".\\temp_data\\amsr2_power_{deg}_{vapor_bins_count}_{flat}_avg_rain.npy", avg_rain)
np.save(f".\\temp_data\\amsr2_power_{deg}_{vapor_bins_count}_{flat}_count.npy", count)
np.save(f".\\temp_data\\amsr2_power_{deg}_{vapor_bins_count}_{flat}_binder.npy", binder_4th_order_cumulant)
np.save(f".\\temp_data\\amsr2_power_{deg}_{vapor_bins_count}_{flat}_var.npy", var)
