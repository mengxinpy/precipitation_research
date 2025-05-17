import xarray as xr
import re
import os
import fnmatch


def seasonal_split(ds):
    # Extract time, lon, lat dimensions
    time = ds['time']

    # Create a mask for each season
    winter_mask = (time.dt.month == 12) | (time.dt.month <= 2)
    spring_mask = (time.dt.month >= 3) & (time.dt.month <= 5)
    summer_mask = (time.dt.month >= 6) & (time.dt.month <= 8)
    autumn_mask = (time.dt.month >= 9) & (time.dt.month <= 11)

    # Use the masks to select data for each season
    seasons = {
        'winter': ds.sel(time=winter_mask),
        'spring': ds.sel(time=spring_mask),
        'summer': ds.sel(time=summer_mask),
        'autumn': ds.sel(time=autumn_mask)
    }

    return seasons


def data_refactor(path):
    nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*.nc')]

    # 打开数据集
    ds = xr.open_mfdataset(nc_files)

    # 按季节分割数据
    seasonal_data = seasonal_split(ds)

    # 为每个季节创建文件夹并保存数据
    for season, data in seasonal_data.items():
        season_dir = os.path.join(path, season)
        os.makedirs(season_dir, exist_ok=True)  # 创建文件夹（如果不存在）

        # 保存数据到相应的文件夹
        output_file = os.path.join(season_dir, f'{season}_data.nc')
        data.to_netcdf(output_file)
        print(f'Saved {season} data to: {output_file}')


if __name__ == '__main__':
    # data_compress_day_1deg(path='C:/ERA5\\1980-2019\\total_precipitation\\', unit=1000)
    # data_compress_day_1deg(path='C:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
    # 调用函数
    data_refactor(path='C:\\ERA5\\1980-2019\\total_precipitation\\')
    # data_compress_hour_1year_1deg(path='D:\\ERA5\\1980-2019\\total_precipitation\\', unit=1000)
