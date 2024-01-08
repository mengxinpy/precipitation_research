import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

bins_v = np.array([45, 50, 55, 60, 65])
result = np.zeros((10, 6, 100))
for area_num in range(10):
    print(area_num)
    files = [glob.glob(f'temp_data_era5\\era5_processed_data_area_vapor{i}_part*.npy') for i in range(area_num * 10, area_num * 10 + 10)]
    files = [item for sublist in files for item in sublist]  # Flatten the list
    arrays = [np.load(file, allow_pickle=True) for file in files]
    concatenated = []
    for v in range(len(bins_v) + 1):
        concatenated.append(np.concatenate([sub[v] for sub in arrays if len(sub) > v]))

        if np.size(concatenated[v]) == 0:
            result[area_num, v, :] = np.nan
        else:
            result[area_num, v, :] = np.nanpercentile(concatenated[v], np.arange(1, 101))
np.save('era5_percentile_10area_vapor.npy', result)
