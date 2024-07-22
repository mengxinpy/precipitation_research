import xarray as xr
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


if __name__ == '__main__':
    # data_compress_day_1deg(path='C:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
    # data_compress_day_1deg(path='C:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
    data_compress_hour_1year_1deg(path='D:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
