import xarray as xr
import os
import fnmatch


def data_compress_day_1deg(path, unit=1):
    nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*.nc') and not fnmatch.fnmatch(f, '*processed_day*.nc')]

    for nc_file in nc_files:
        print(nc_file)
        ds = xr.open_dataset(nc_file)
        ds = ds.astype('float32')
        ds = ds.resample(time='1D').sum(dim='time')
        ds = ds * unit
        ds = ds.coarsen(longitude=4, latitude=4, boundary='trim').median()

        new_nc_file = nc_file.replace('.nc', '_processed_day_1.nc')
        ds.to_netcdf(new_nc_file)


if __name__ == '__main__':
    # data_compress_day_1deg(path='E:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
    data_compress_day_1deg(path='E:\\ERA5\\1980-2019\\large_scale_precipitation\\', unit=1000)