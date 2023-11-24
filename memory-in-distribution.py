import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

percentile_values = np.load('era5_percentile_area.npy')
err5_frequency = xr.open_dataset('era5_frequency.nc')
result = np.zeros((100, 100))
for area_num in range(100):
    print(area_num)
    # files = glob.glob(f'temp_data_era5\\era5_processed_data_area_vapor90_part0.npy')
    files = glob.glob(f'temp_data_era5\\era5_processed_data_area_vapor{area_num}_part*.npy')
    arrays = [np.concatenate(np.load(file, allow_pickle=True)) for file in files]
    concatenated_area = np.concatenate(arrays)
    top_30_percent = int(len(concatenated_area) * 0.3)

    # 使用np.partition找到前30%最大的数
    # np.partition会将数组分割为两部分，左边是最大的n个数，右边是其他数
    # 使用[-top_30_percent]可以获取这些最大的数
    top_data = np.partition(concatenated_area, -top_30_percent)[-top_30_percent:]
    result = np.percentile(top_data, np.arange(0, 99, 1))
    # 使用列表推导式和concatenate函数
    # concatenated = []
    # for v in range(len(arrays[0])):
    #     concatenated.append(np.concatenate([sub[v] for sub in arrays if len(sub) > v]))
    #
    #     if np.size(concatenated[v]) == 0:
    #         result[v, area_num, :] = np.nan
    #     else:
    #         result[v, area_num, :] = np.nanpercentile(concatenated[v], np.arange(1, 101))
np.save('era5_percentile_top_30.npy', result)
