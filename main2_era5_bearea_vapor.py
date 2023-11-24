import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time

pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"
rainfall_frequency = xr.open_dataset('era5_frequency.nc').to_array().values.squeeze()
bins = np.arange(0, np.max(rainfall_frequency), 0.01)  # 你可以根据实际情况调整
indices = np.digitize(rainfall_frequency, bins)

dr = xr.open_mfdataset(pathr + '*processed_day_0.25.nc')
dv = xr.open_mfdataset(pathv + '*processed_day_0.25.nc')
n = len(dr.time)
step = n // 20

for i in range(20):
    print('part:', i)
    subsetr = dr.isel(time=slice(i * step, (i + 1) * step if i < 19 else n))
    subsetv = dv.isel(time=slice(i * step, (i + 1) * step if i < 19 else n))
    for j in range(len(bins)):
        print(f'area:{j}')
        rainfall_in_this_area = subsetr.to_array().values.squeeze()[:, indices == j + 1].flatten()
        vapor_in_this_area = subsetv.to_array().values.squeeze()[:, indices == j + 1].flatten()
        bins_v = np.array([45, 50, 55, 60, 65])

        indicesv = np.digitize(vapor_in_this_area, bins_v)

        split_rainfall = [rainfall_in_this_area[np.where((indicesv == k) & (rainfall_in_this_area > 1))] for k in range(0, len(bins_v) + 1)]
        # 你可以将这个列表转换为对象数组
        arr = np.array(split_rainfall, dtype=object)
        np.save(f'.\\temp_data_era5\\era5_processed_data_area_vapor{j}_part{i}.npy', arr)
