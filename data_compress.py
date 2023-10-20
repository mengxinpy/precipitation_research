import xarray as xr
import os

# 获取文件夹下所有.nc文件
path = "F:\\liusch\\ERA5\\1980-2019\\total_precipitation\\"
nc_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_day.nc')]
# 获取文件夹下所有.nc文件

# 对每个文件进行处理
for nc_file in nc_files:
    # 打开.nc文件
    print(nc_file)
    ds = xr.open_dataset(nc_file)

    # 将数据转化为float16精度
    ds = ds.astype('float32')

    # 保存为新的.nc文件
    ds.to_netcdf(nc_file.replace('.nc', '_processed.nc'))
