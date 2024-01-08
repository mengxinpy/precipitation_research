import time
import numpy as np
from lag_parameter import dr_path, lspf_path, sp_frequency, sp_percentile
import xarray as xr
from plt_test import era5_draw_area_dataArray

start_time = time.time()

dr = dr_path
lspf = lspf_path

all_time_len = 3600 * 24 * lspf.shape[0]
lspf = lspf.sum(dim='time')
lspf_frequency = lspf / all_time_len
lspf_frequency.to_netcdf(sp_frequency)

raw_data = dr.values
frequency = lspf_frequency.values
bins = np.linspace(0, np.max(frequency), 6, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
print(f'max_frequency:{np.max(frequency)}')
indices = np.digitize(frequency, bins)
result_percentile = np.zeros((len(bins), 100))

assert len(bins) == indices.max() == 6
assert indices.min() == 1

for area_num in range(len(bins)):
    print(area_num)
    condition_area = area_num + 1 == indices
    condition_wetday = raw_data > 1
    result_percentile[area_num, :] = np.nanpercentile(raw_data[condition_wetday & condition_area], np.arange(1, 101))

np.save(sp_percentile, result_percentile)
np.save('bins.npy', bins)
np.save('indices.npy', indices)
end_time = time.time()
print('程序运行时间: ', end_time - start_time)
