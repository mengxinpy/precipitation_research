import time
import numpy as np
from lag_parameter import dr_path, lsp_path, sp_frequency, sp_percentile
import xarray as xr

start_time = time.time()

lsp = lsp_path
dr = dr_path

# lspf频率
lsp_st = lsp.sum(dim='time')
dr_st = dr.sum(dim='time')
lsprf_frequency = lsp_st / dr_st
lsprf_frequency.to_netcdf(sp_frequency)

raw_data = dr.values
frequency = lsprf_frequency.values
bins = np.linspace(0, np.max(frequency), 6, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
indices = np.digitize(frequency, bins)
result_percentile = np.zeros((len(bins), 100))

assert len(bins) == indices.max()
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
