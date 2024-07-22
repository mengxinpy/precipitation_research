import xarray as xr
import re
import os
import fnmatch


def select_first(block, axis=None):
    # 选择每个块的第一个元素
    # 'axis'参数在这里不使用，但是定义它是为了避免TypeError
    return block[:, :, 0, :, 0]
    # return block.isel(longitude=0, latitude=0)


def data_compress_hour_1year_1deg(path, unit=1):
    nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*.nc') and not fnmatch.fnmatch(f, '*processed_day*.nc')]
    files_to_open = [f for f in nc_files if '198001' <= f[-9:-3] <= '198010']

    for nc_file in nc_files:
        print(nc_file)
        ds = xr.open_dataset(nc_file)
        ds = ds.astype('float32')
        # ds = ds.resample(time='1D').sum(dim='time')
        ds = ds * unit
        # 使用coarsen和reduce选取每4个经纬度格点的第一个
        ds = ds.coarsen(longitude=4, latitude=4, boundary='trim').mean()

        # new_nc_file = nc_file.replace('.nc', '_processed_hour_1year_1deg.nc')
        # new_nc_file = new_nc_file.replace('D', 'C')
        ds.to_netcdf(f'C:\\ERA5\\1980-2019\\total_precipitation_1year\\{nc_file[-9:-3]}_processed_hour_1year_1deg.nc')


def data_compress_day_1deg(path, unit=1):
    nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*.nc') and not fnmatch.fnmatch(f, '*processed_day*.nc')]

    for nc_file in nc_files:
        print(nc_file)
        ds = xr.open_dataset(nc_file)
        ds = ds.astype('float32')
        ds = ds.resample(time='1D').sum(dim='time')
        ds = ds * unit
        # 使用coarsen和reduce选取每4个经纬度格点的第一个
        ds = ds.coarsen(longitude=4, latitude=4, boundary='trim').mean()

        new_nc_file = nc_file.replace('.nc', '_processed_day_1.nc')
        ds.to_netcdf(new_nc_file)


def data_compress_day_amsr2(path, unit=1):
    nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*processed.nc')]

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)

        # 提取名字
        match = re.search(r'(\d{4}-\d{2}-\d{2})', nc_file)
        if not match:
            raise ValueError(f"无法从文件名中提取时间信息: {nc_file}")
        date_str = match.group(1)

        # 将时间信息添加到数据集的属性中
        ds.attrs['date'] = date_str
        ds = ds.astype('float32')

        # 化为天
        ds = ds * 24
        ds = ds * unit

        # 修改 lon 和 lat 的名称为 longitude 和 latitude
        ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})

        # 使用coarsen和reduce选取每4个经纬度格点的第一个
        ds = ds.coarsen(longitude=4, latitude=4, boundary='trim').mean(skipna=True)

        # 选择 pass 为 1 的数据
        ds = ds.sel({'pass': 1})

        # 选择 rain_rate 变量并将其重命名为 tp
        ds = ds[['rain_rate']].rename({'rain_rate': 'tp'})

        # 生成新的文件名
        new_filename = f"F:\\liusch\\amsr2\\processed_amsr\\{date_str}_processed_amsr2.nc"

        # 保存修改后的数据到新的 NetCDF 文件
        ds.to_netcdf(new_filename)
        print(f"Processed {nc_file} and saved as {new_filename}")


if __name__ == '__main__':
    # data_compress_day_1deg(path='C:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
    # data_compress_day_1deg(path='C:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
    data_compress_day_amsr2(path='F:\\liusch\\amsr2\\', unit=1)
    # data_compress_hour_1year_1deg(path='D:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
