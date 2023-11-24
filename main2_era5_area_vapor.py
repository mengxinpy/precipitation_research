import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

result = np.zeros((6, 100, 100))
for area_num in range(100):
    print(area_num)
    # files = glob.glob(f'temp_data_era5\\era5_processed_data_area_vapor90_part0.npy')
    files = glob.glob(f'temp_data_era5\\era5_processed_data_area_vapor{area_num}_part*.npy')
    arrays = [np.load(file, allow_pickle=True) for file in files]
    # 使用列表推导式和concatenate函数
    concatenated = []
    for v in range(len(arrays[0])):
        concatenated.append(np.concatenate([sub[v] for sub in arrays if len(sub) > v]))

        if np.size(concatenated[v]) == 0:
            result[v, area_num, :] = np.nan
        else:
            result[v, area_num, :] = np.nanpercentile(concatenated[v], np.arange(1, 101))
np.save('era5_percentile_area_vapor_single.npy', result)
