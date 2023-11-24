import xarray as xr
import fnmatch
import os

# 现在，nc_files列表就只包含你想要的.nc文件了
import numpy as np

# 获取文件夹下所有.nc文件
# path = "E:\\ERA5\\1980-2019\\total_precipitation\\"
path = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_processed_day.nc')]
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_processed_day.nc')]
# 找到所有.nc文件，但不包含'_processed_day'的.nc文件
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*_processed_day_processed_day_1.nc')]
nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '199*.nc') and '_processed_day' not in f]

# nc_files2 = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '200*.nc') and '_processed_day' not in f]
# nc_files = nc_files1 + nc_files2
# 对每个文件进行处理
for nc_file in nc_files:
    # 打开.nc文件
    print(nc_file)
    ds = xr.open_dataset(nc_file)
    # 将数据转化为float32精度，并保留三位小数
    # ds = ds.astype('float32')
    # ds = (ds.astype('float32') * 1000)
    # ds = ds.resample(time='1D').sum(dim='time')
    ds = ds.coarsen(longitude=2, latitude=2, boundary='trim').mean()
    # 保存为新的.nc文件
    new_nc_file = nc_file.replace('.nc', 'hour_0.5.nc')
    ds.to_netcdf(new_nc_file)
    # new_nc_file = nc_file.replace('.nc', '_processed_day.nc')
    # new_nc_file = nc_file.replace('.nc', '_processed.nc')

    # 删除原.nc文件
    # os.remove(nc_file)
    # os.remove(nc_file.replace('.nc', '_processed_day_0.5.nc'))
