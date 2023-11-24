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

area_top_30 = np.load('era5_percentile_area.npy')[:, 69]
dr = xr.open_mfdataset(pathr + '*processed_day_0.25.nc')
n = len(dr.time)
step = n // 20

for i in range(20):
    print('part:', i)
    subsetr = dr.isel(time=slice(i * step, (i + 1) * step if i < 19 else n))
    raw_data = subsetr.to_array().values.squeeze()
    condition_wetday = raw_data > 1
    for j in range(len(bins)):

        print(f'area:{j}')
        condition_top30 = raw_data > area_top_30[j]
        condition_area = indices == j + 1
        condition_area_top30 = condition_top30 & condition_area

        area_rain = raw_data[condition_wetday & condition_area]

        area_top30__rain = raw_data[condition_area_top30]
        area_top30__rain_klag = []
        for k in [1, 2, 3, 4, 5, 10, 50, 100]:
            area_klag_condition = np.roll(condition_area_top30, shift=k)
            area_klag_condition[0:k - 1, :, :] = False
            top_k_lag_area_rain = raw_data[area_klag_condition]
            area_top30__rain_klag.append(top_k_lag_area_rain)
        area_top30__rain_klag = np.array(area_top30__rain_klag, dtype=object)

        np.save(f'.\\temp_data_era5\\era5_rain_area_{j}_part{i}.npy', area_rain)
        np.save(f'.\\temp_data_era5\\era5_top30-rain_area_{j}_part{i}.npy', area_top30__rain)
        np.save(f'.\\temp_data_era5\\era5_top30-klag-rain_area_{j}_part{i}.npy', area_top30__rain_klag)
