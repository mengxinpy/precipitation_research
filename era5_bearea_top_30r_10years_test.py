import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time

pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"
area_top_30 = np.load('era5_percentile_area.npy')[:, 69]
rainfall_frequency = xr.open_dataset('era5_frequency.nc').to_array().values.squeeze()
bins = np.arange(0, np.max(rainfall_frequency), 0.01)  # 你可以根据实际情况调整
indices = np.digitize(rainfall_frequency, bins)

dr = xr.open_mfdataset(pathr + '199*processed_day_0.25.nc')
# dv = xr.open_mfdataset(pathv + '*processed_day_0.25.nc')
# n = len(dr.time)
# step = n // 20
result_0lag = np.zeros((100, 100))
result_klag = np.zeros((100, 100))
result_all = np.zeros((100, 100))
result_fix_top30 = np.zeros((100, 100))
# for i in range(20):
#     print('part:', i)
#     subsetr = dr.isel(time=slice(i * step, (i + 1) * step if i < 19 else n))
# subsetv = dv.isel(time=slice(i * step, (i + 1) * step if i < 19 else n))
raw_data = dr.to_array().values.squeeze()
k = 1000
for j in range(len(bins)):
    print(f'area:{j}')

    area_rain = raw_data[(raw_data > 1) & (indices == j + 1)]
    result_all[j, :] = np.nanpercentile(area_rain, np.arange(1, 101))

    area_0lag = raw_data[(raw_data > area_top_30[j]) & (indices == j + 1)]
    result_0lag[j, :] = np.nanpercentile(area_0lag, np.arange(1, 101))
    area_klag_condition = np.roll((raw_data > area_top_30[j]) & (indices == j + 1), shift=k, axis=0)
    # area_1lag_condition[0, :, :] = False
    top_klag_area_rain = raw_data[area_klag_condition & (raw_data > 1)]
    result_klag[j, :] = np.nanpercentile(top_klag_area_rain, np.arange(1, 101))

    area_fix_top30 = raw_data[:, (raw_data[0] > area_top_30[j]) & (indices == j + 1)]
    area_fix_top30 = area_fix_top30[area_fix_top30 > 1]
    result_fix_top30[j, :] = np.nanpercentile(area_fix_top30, np.arange(1, 101))
np.save(f'era5_percentile_{k}lag_top30_10years', result_klag)
np.save('era5_percentile_0lag_top30_10years', result_0lag)
np.save('era5_percentile_all_top30_10years', result_all)
np.save('era5_percentile_fix_top30_10years', result_fix_top30)

# rainfall_in_this_area = subsetr.to_array().values.squeeze()[:, indices == j + 1].flatten()
# vapor_in_this_area = subsetv.to_array().values.squeeze()[:, indices == j + 1].flatten()
# bins_v = np.array([45, 50, 55, 60, 65])

# indicesv = np.digitize(vapor_in_this_area, bins_v)
#
# split_rainfall = [rainfall_in_this_area[np.where((indicesv == k) & (rainfall_in_this_area > 1))] for k in range(0, len(bins_v) + 1)]
# 你可以将这个列表转换为对象数组
# arr = np.array(split_rainfall, dtype=object)
# np.save(f'.\\temp_data_era5\\era5_processed_data_area_vapor{j}_part{i}.npy', arr)
