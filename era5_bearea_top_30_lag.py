import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
from plt_test import draw_area_heap

pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"
path_out = "E:\\ERA5\\1980-2019\\outer_klag_rain\\"
rainfall_frequency = xr.open_dataset('era5_frequency.nc').to_array().values.squeeze()
bins = np.arange(0, np.max(rainfall_frequency), 0.01)  # 你可以根据实际情况调整
indices = np.digitize(rainfall_frequency, bins)
area_top_30 = np.load('era5_percentile_area.npy')[:, 69]
dr = xr.open_mfdataset(pathr + '*processed_day_0.25.nc')
n = len(dr.time)
step = n // 20

for i in range(0, 19):
    print('part:', i)

    subset_after = dr.isel(time=slice(i * step, (i + 1) * step))
    subset_later = dr.isel(time=slice((i + 1) * step, (i + 2) * step if i < 18 else n))
    print(i * step, '|', (i + 1) * step)

    raw_data_after = subset_after.to_array().values.squeeze()
    raw_data_later = subset_later.to_array().values.squeeze()

    condition_wetday_after = raw_data_after > 1
    condition_wetday_later = raw_data_later > 1
    for j in range(indices.min(), indices.max() + 1):

        print(f'area:{j}')

        condition_top30_after = raw_data_after > area_top_30[j - 1]  # top30条件
        condition_top30_later = raw_data_later > area_top_30[j - 1]

        condition_area = indices == j  # 区域条件
        condition_area_top30_after = condition_top30_after & condition_area  # 合并条件
        condition_area_top30_later = condition_top30_later & condition_area

        area_top30__rain_klag_after = []
        area_top30__rain_klag_later = []
        for k in range(1, 360, 6):
            # if i == 0:
            # if 70 <= j <= 70:
            #     draw_area_heap(np.where(condition_area_top30_later[0], raw_data_later[0], np.nan), f'condition_area_top30_{k}lag_later0')
            #     draw_area_heap(np.where(condition_area_top30_later[1], raw_data_later[0], np.nan), f'condition_area_top30_{k}lag_later1')
            #     draw_area_heap(np.where(condition_area_top30_later[2], raw_data_later[0], np.nan), f'condition_area_top30_{k}lag_later2')
            # test_klag_af = condition_area_top30_after[-k:, :, :]
            condition_area_top30_klag_after = np.roll(condition_area_top30_after, shift=k, axis=0)
            condition_area_top30_klag_later = np.roll(condition_area_top30_later, shift=k, axis=0)
            condition_area_top30_klag_later[0:k, :, :] = condition_area_top30_klag_after[0:k, :, :]
            # assert np.array_equal(test_klag_af, condition_area_top30_klag_later[:k])
            condition_area_top30_klag_after = condition_area_top30_klag_after & condition_wetday_after
            condition_area_top30_klag_later = condition_area_top30_klag_later & condition_wetday_later

            if i == 0:
                #     if 70 <= j <= 70:
                #         draw_area_heap(np.where(condition_area_top30_klag_later[1], raw_data_later[0], np.nan), f'condition_area_top30_{k}lag_later1_ink')
                #         draw_area_heap(np.where(condition_area_top30_klag_later[2], raw_data_later[0], np.nan), f'condition_area_top30_{k}lag_later2_ink')
                condition_area_top30_klag_after[0:k - 1, :, :] = False
                area_klag_rain_after = raw_data_after[condition_area_top30_klag_after]
                area_top30__rain_klag_after.append(area_klag_rain_after)

            area_klag_rain_later = raw_data_later[condition_area_top30_klag_later]
            area_top30__rain_klag_later.append(area_klag_rain_later)

        area_top30__rain_klag_after = np.array(area_top30__rain_klag_after, dtype=object)
        area_top30__rain_klag_later = np.array(area_top30__rain_klag_later, dtype=object)
        if i == 0:
            area_rain = raw_data_after[condition_wetday_after & condition_area]
            area_top30__rain = raw_data_after[condition_area_top30_after]
            np.save(f'{path_out}\\era5_rain_area_{j - 1}_part{i}.npy', area_rain)
            np.save(f'{path_out}\\era5_top30-rain_area_{j - 1}_part{i}.npy', area_top30__rain)
            np.save(f'{path_out}\\era5_top30-klag-rain_area_{j - 1}_part{i}.npy', area_top30__rain_klag_after)
        area_rain_later = raw_data_later[condition_wetday_later & condition_area]
        area_top30__rain_later = raw_data_later[condition_area_top30_later]
        np.save(f'{path_out}\\era5_rain_area_{j - 1}_part{i + 1}.npy', area_rain_later)
        np.save(f'{path_out}\\era5_top30-rain_area_{j - 1}_part{i + 1}.npy', area_top30__rain_later)
        np.save(f'{path_out}\\era5_top30-klag-rain_area_{j - 1}_part{i + 1}.npy', area_top30__rain_klag_later)
