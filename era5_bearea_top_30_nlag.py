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

for i in range(0, 19):
    print('part:', i)

    subset = dr.isel(time=slice(i * step, (i + 1) * step))
    subset_later = dr.isel(time=slice((i + 1) * step, (i + 2) * step if i < 18 else n))

    raw_data = subset.to_array().values.squeeze()
    raw_data_later = subset_later.to_array().values.squeeze()

    condition_wetday = raw_data > 1
    condition_wetday_later = raw_data_later > 1
    for j in range(len(bins)):

        print(f'area:{j}')

        condition_top30 = raw_data > area_top_30[j]
        condition_top30_later = raw_data_later > area_top_30[j]
        condition_area = indices == j + 1
        condition_area_top30 = condition_top30 & condition_area
        condition_area_top30_later = condition_top30_later & condition_area

        area_top30__rain_klag = []
        area_top30__rain_klag_later = []
        for k in [1, 2, 3, 5, 10, 50, 100, 500]:
            area_klag_condition = np.roll(condition_area_top30, shift=k, axis=0)
            area_klag_condition_later = np.roll(condition_area_top30_later, shift=k, axis=0)

            area_klag_condition_later[0:k, :, :] = area_klag_condition[0:k, :, :]

            area_klag_condition = area_klag_condition & condition_wetday
            area_klag_condition_later = area_klag_condition_later & condition_wetday_later

            if i == 0:
                area_klag_condition[0:k - 1, :, :] = False
                area_klag_rain = raw_data[area_klag_condition]
                area_top30__rain_klag.append(area_klag_rain)

            area_klag_rain_later = raw_data_later[area_klag_condition_later]
            area_top30__rain_klag_later.append(area_klag_rain_later)

        area_top30__rain_klag = np.array(area_top30__rain_klag, dtype=object)
        area_top30__rain_klag_later = np.array(area_top30__rain_klag_later, dtype=object)
        if i == 0:
            np.save(f'.\\temp_data_era5\\era5_top30-klag-rain_area_{j}_part{i}.npy', area_top30__rain_klag)
        np.save(f'.\\temp_data_era5\\era5_top30-klag-rain_area_{j}_part{i + 1}.npy', area_top30__rain_klag_later)
