import numpy as np
import xarray as xr

# 定义范围和点的数量
start = 1
end = 180
num_points = 30

# 计算对数底数
base = end ** (1 / (num_points - 1))  # 生成对数点并取最近的整数
log_points = np.unique([int(round(start * base ** n)) for n in range(num_points)])
# 输入参数
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"
path_lspf = 'E:\\ERA5\\1980-2019\\large_scale_precipitation_fraction\\'
path_lsp = 'E:\\ERA5\\1980-2019\\large_scale_precipitation\\'
path_cp = 'E:\\ERA5\\1980-2019\\convective_precipitation\\'

dr_lat60 = xr.open_mfdataset(pathr + '*processed_day_1.nc').sel(latitude=slice(60, -60))['tp']
lspf_lat60 = xr.open_mfdataset(path_lspf + '*processed_day_1.nc').sel(latitude=slice(60, -60))['lspf']
lsp_lat60 = xr.open_mfdataset(path_lsp + '*processed_day_1.nc').sel(latitude=slice(60, -60))['lsp']
cp_lat60 = xr.open_mfdataset(path_cp + '*processed_day_1.nc').sel(latitude=slice(60, -60))['cp']

dr_all = xr.open_mfdataset(pathr + '*processed_day_1.nc')['tp']  # 这几个变量时间长度不是一直的如果需要还要进一步下载文件--------------------------------
lspf_all = xr.open_mfdataset(path_lspf + '*processed_day_1.nc')['lspf']
lsp_all = xr.open_mfdataset(path_lsp + '*processed_day_1.nc')['lsp']

# percentile script input parameter
lsp_path = lsp_lat60
dr_path = dr_lat60
# lspf_path = lspf_lat60

# percentile script input parameter
# sp_frequency = 'lsprf_frequency.nc'
# sp_percentile = 'lsprf_percentile.npy'
# sp_frequency = 'lspf_frequency_lat60.nc'
# sp_percentile = 'lspf_percentile_lat60.npy'
# sp_frequency = 'lsprf_frequency_lat60.nc'
# sp_percentile = 'lsprf_percentile_lat60.npy'
# cp percentile input
cp_path = cp_lat60
# k cp percentile output
sp_frequency = 'cp_frequency_lat60.nc'
sp_percentile = 'cp_percentile_lat60.npy'
