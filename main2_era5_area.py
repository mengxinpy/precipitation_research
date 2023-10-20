import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

result = np.zeros((100, 100))
for area_num in range(100):
    print(area_num)
    files = glob.glob(f'temp_data_era5\\era5_processed_data_area{area_num}_part*.npy')
    arrays = [np.load(file) for file in files]
    if np.size(arrays) == 0:
        result[area_num, :] = np.nan
    else:
        result[area_num, :] = np.nanpercentile(np.concatenate(arrays), np.arange(1, 101))
np.save('era5_percentile_area.npy', result)
