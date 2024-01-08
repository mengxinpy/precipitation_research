import xarray as xr
import numpy as np
from plt_test import era5_draw_area_dataArray

path_lsp = 'E:\\ERA5\\1980-2019\\large_scale_precipitation\\'  # 存放文件夹
path_rain = 'E:\\ERA5\\1980-2019\\total_precipitation\\'  # 存放文件夹
path_lspf = 'E:\\ERA5\\1980-2019\\large_scale_precipitation_fraction\\'  # 存放文件夹
lsp = xr.open_mfdataset(path_lsp + '*processed_day_1.nc')['lsp']
tp = xr.open_mfdataset(path_rain + '*processed_day_1.nc')['tp']
# lsp = xr.open_mfdataset(path_lsp + '2000*processed_day_1.nc')['lsp']
lsp = lsp.sum(dim='time')
tp = tp.sum(dim='time')
lsprf = lsp / tp
# lsp_frequency = lsp / lsp.max()
# lsp_frequency.to_netcdf('lsp_frequency.nc')
# lspf = xr.open_mfdataset(path_lspf + '*processed_day_1.nc')['lspf'].sel(latitude=slice(60, -60))#注意这里的时间范围-------------------------------------------
# lspf = lspf.sum(dim='time')
# lspf_frequency = lspf / lspf.max()
# lspf_frequency.to_netcdf('lspf_frequency_lat60.nc')
# np.save('lspf_frequency.npy', lspf_frequency)
# 假设 'precipitation' 是两个数据集中的变量名
# 假设 'lsp' 和 'tp' 是两个数据集中的变量名
# 假设 'lsp' 和 'tp' 分别是两个 Dataset 中的变量名
# lsp['lsp'] = xr.where(lsp['lsp'] < 0, 0, lsp['lsp'])
# tp['tp'] = xr.where(tp['tp'] < 0, 0, tp['tp'])
# lsp = lsp.where(lsp >= 0, 0)
# tp = tp.where(tp >= 0, 0)
#
# 计算比值
# lspf = lsp['lsp'] / tp['tp']
era5_draw_area_dataArray(lsprf, 'lsprf_frequency')
# era5_draw_area_dataArray((lsp > tp)[0], 'lsp_morethan_tp')
aa = lsprf.values
