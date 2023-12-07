import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

path_out = "E:\\ERA5\\1980-2019\\outer_klag_rain\\"
percentile_all = np.zeros((100, 100))
percentile_top30 = np.zeros((100, 100))
percentile_lag = np.zeros((60, 100, 100))  # 注意这个固定参数
for area_num in range(100):
    print(area_num)
    # files = glob.glob(f'temp_data_era5\\era5_processed_data_area_vapor90_part0.npy')
    files_all = glob.glob(f'{path_out}\\era5_rain_area_{area_num}_part*.npy')
    files_top30 = glob.glob(f'{path_out}\\era5_top30-rain_area_{area_num}_part*.npy')
    files_lag = glob.glob(f'{path_out}\\era5_top30-klag-rain_area_{area_num}_part*.npy')
    arrays_all = [np.load(file, allow_pickle=True) for file in files_all]
    arrays_top30 = [np.load(file, allow_pickle=True) for file in files_top30]
    arrays_lags = [np.load(file, allow_pickle=True) for file in files_lag]
    # 使用列表推导式和concatenate函数
    percentile_all[area_num, :] = np.nanpercentile(np.concatenate(arrays_all), np.arange(1, 101))
    percentile_top30[area_num, :] = np.nanpercentile(np.concatenate(arrays_top30), np.arange(1, 101))

    for k in range(len(arrays_lags[0])):
        concatenated = np.concatenate([sub[k] for sub in arrays_lags if len(sub) > k])
        # concatenated.append(np.concatenate([sub[k] for sub in arrays_lags if len(sub) > k]))

        if np.size(concatenated) == 0:
            percentile_lag[k, area_num, :] = np.nan
        else:
            percentile_lag[k, area_num, :] = np.nanpercentile(concatenated, np.arange(1, 101))
np.save('era5_percentile_all_40years', percentile_all)
np.save('era5_percentile_top30_40years', percentile_top30)
np.save('era5_percentile_klag_40years', percentile_lag)
