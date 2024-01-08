import xarray as xr
# from lag_parameter import indices, bins
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import os
import fnmatch

start_time = time.time()

# path = "E:\\ERA5\\1980-2019\\total_precipitation\\"

path_lsp = 'E:\\ERA5\\1980-2019\\large_scale_precipitation\\'  # 存放文件夹
nc_files = [os.path.join(path_lsp, f) for f in os.listdir(path_lsp) if fnmatch.fnmatch(f, '*processed_day_1.nc') and '_processed_day_processed_day' not in f]
merra2_me = xr.open_mfdataset(nc_files)['lsp']
cmorph_process = merra2_me.where(merra2_me > 1, 0)
cmorph_process = cmorph_process.where(merra2_me < 1, 1)
cmorph_process = cmorph_process.sum(dim='time') / cmorph_process.shape[0]
cmorph_process.to_netcdf('lsp_frequency_1.nc')
wetday_frequency = cmorph_process.values
bins = np.arange(0, np.max(wetday_frequency), 0.01)  # 你可以根据实际情况调整
# bins = np.arange(0, np.max(wetday_frequency), 0.17)  # 你可以根据实际情况调整
indices = np.digitize(wetday_frequency, bins=bins)
raw_data = merra2_me.values
assert len(bins) == indices.max()
result_percentile = np.zeros((len(bins), 100))
assert indices.min() == 1
for area_num in range(len(bins)):
    print(area_num)
    condition_area = area_num + 1 == indices
    condition_wetday = raw_data > 1
    result_percentile[area_num, :] = np.nanpercentile(raw_data[condition_wetday & condition_area], np.arange(1, 101))
np.save('ear5_percentile_area_1deg_6area_lsp.npy', result_percentile)
end_time = time.time()
print('程序运行时间: ', end_time - start_time)
