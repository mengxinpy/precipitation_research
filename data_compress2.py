import xarray as xr
import os
import numpy as np
# #
# # 获取文件夹下所有.nc文件
# path = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
# # nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_day.nc')]
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_processed_day.nc')]
#
# # 对每个文件进行处理
# for nc_file in nc_files:
#     # 打开.nc文件
#     print(nc_file)
#     ds = xr.open_dataset(nc_file)
#     # 将数据转化为float32精度，并保留三位小数
#     # ds = ds.astype('float32')
#     # ds = (ds.astype('float32') * 1000)
#     # ds = ds.resample(time='1D').mean(dim='time')
#     ds = ds.coarsen(longitude=4, latitude=2, boundary='trim').mean()
#     # 保存为新的.nc文件
#     new_nc_file = nc_file.replace('.nc', '_0.5.nc')
#     # new_nc_file = nc_file.replace('.nc', '_processed_day.nc')
#     # new_nc_file = nc_file.replace('.nc', '_processed.nc')
#     ds.to_netcdf(new_nc_file)
#
#     # 删除原.nc文件
#     # os.remove(nc_file)
#     # os.remove(nc_file.replace('_day', ''))
#
# 指定你要改变的文件夹路径
# folder_path = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
folder_path = "E:\\ERA5\\1980-2019\\total_precipitation\\"

for filename in os.listdir(folder_path):
    if filename.endswith("processed_day.nc"):
        new_filename = filename.replace("processed_day.nc", "processed_day_0.25.nc")
        print(new_filename)
        source = os.path.join(folder_path, filename)
        destination = os.path.join(folder_path, new_filename)
        os.rename(source, destination)  # rename() function is used to change the name of the file
