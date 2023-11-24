import xarray as xr
import fnmatch
import os

# 现在，nc_files列表就只包含你想要的.nc文件了
import numpy as np

# 获取文件夹下所有.nc文件
path = "F:\\liusch\\amsr2\\"
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_processed_day.nc')]
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.nc')]
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('processed.nc')]
# 找到所有.nc文件，但不包含'_processed_day'的.nc文件
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*_processed_day_processed_day_1.nc')]
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*_processed_day.nc') and '_1' not in f and '_0.5' not in f]

# ds['water_vapor'] = ds['water_vapor'].loc[ds['pass'] == 0]
# ds['rain_rate'] = ds['rain_rate'].loc[ds['pass'] == 0]
# 对每个文件进行处理
# 打开每个NetCDF文件，并将数据集放入一个列表中
# Use dask.delayed to open each file and store the results in a list
# datasets = [xr.open_dataset(f, chunks={'lon': 5}) for f in nc_files]

# 使用xr.concat函数合并数据集
flat = 20
# ds = xr.concat(datasets, dim='time').sel(lat=slice(-flat, flat)).sel({'pass': 1})
# print('ok')
# ds.to_netcdf('amsr2_lat20_concat.nc')
# print('ok')
# aa = ds['water_vapor'].values
# # subset = ds.sel({'pass': 1})
# # subset.to_netcdf('amsr2_lat20_timeconcat.nc')
# i = 1
for nc_file in nc_files:
    # 打开.nc文件
    print(nc_file)
    ds = xr.open_dataset(nc_file).sel(lat=slice(-flat, flat)).sel({'pass': 1})
    # da = ds['water_vapor']
    # dp = ds[['water_vapor', 'rain_rate']]
    # 将数据转化为float32精度，并保留三位小数
    # ds = ds.astype('float32')
    # ds = (ds.astype('float32') * 1000)
    # ds = ds.resample(time='1D').sum(dim='time')
    # ds = ds.coarsen(longitude=2, latitude=2, boundary='trim').mean()
    # 保存为新的.nc文件
    new_nc_file = nc_file.replace('_processed.nc', '_processed_clat.nc')
    ds.to_netcdf(new_nc_file)
    # new_nc_file = nc_file.replace('.nc', '_processed_day.nc')
    # new_nc_file = nc_file.replace('.nc', '_processed.nc')

    # 删除原.nc文件
    # os.remove(nc_file)
    # os.remove(nc_file.replace('.nc', '_processed_day_0.5.nc'))
