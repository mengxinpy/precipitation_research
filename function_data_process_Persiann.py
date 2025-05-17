import os
import gzip
import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

def convert_persiann_to_nc(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    latitudes = np.linspace(59.875, -59.875, 480)
    longitudes = np.linspace(0.125, 359.875, 1440)
    gz_files = sorted(f for f in os.listdir(input_dir) if f.startswith('ms6s4_d') and f.endswith('.bin.gz'))

    for fname in gz_files:
        path = os.path.join(input_dir, fname)
        print(f"Converting {fname} ...")

        with gzip.open(path, 'rb') as f:
            raw = f.read()
        # 立即拷贝，确保是可写的 array
        arr = np.frombuffer(raw, dtype='>f4').copy()
        if arr.size != 480 * 1440:
            print(f"  ⚠️ 数据长度不符（{arr.size} vs {480*1440}），跳过")
            continue
        arr = arr.reshape((480, 1440))

        # 现在就可以原地赋值了
        arr[arr == -9999] = np.nan


        # 从文件名解析日期：ms6s4_dYYDDD.bin.gz → YYDDD
        yyddd = fname.split('_d')[-1].split('.bin')[0]
        YY = int(yyddd[:2])
        DDD = int(yyddd[2:])
        year = 2000 + YY
        date = datetime(year, 1, 1) + timedelta(days=DDD - 1)

        # 构造 xarray DataArray + Dataset
        da = xr.DataArray(
            data=arr[np.newaxis, :, :],
            dims=('time', 'latitude', 'longitude'),
            coords={
                'time': [date],
                'latitude': latitudes,
                'longitude': longitudes
            },
            name='precipitation'
        )
        ds = da.to_dataset()

        # 保存为 .nc
        outname = fname.replace('.bin.gz', '.nc')
        outpath = os.path.join(output_dir, outname)
        ds.to_netcdf(outpath)
        print(f"  → saved {outname}")

def data_compress_persiann(nc_dir, output_dir, unit=1):
    """
    读取单日 .nc：
    - precipitation → tp
    - cast float32 * unit
    - coarsen 10×10 平均
    - 保存 *_processed_day_1.nc
    最后合并所有 *_processed_day_1.nc 为 PERSIANN_processed_combined.nc
    """
    os.makedirs(output_dir, exist_ok=True)

    nc_files = sorted(f for f in os.listdir(nc_dir) if f.endswith('.nc'))
    for fname in nc_files:
        path = os.path.join(nc_dir, fname)
        print(f"Processing {fname} ...")

        ds = xr.open_dataset(path)

        # 重命名变量
        if 'precipitation' in ds.data_vars:
            ds = ds.rename({'precipitation': 'tp'})
        else:
            print(f"  ⚠️ 未找到 'precipitation' 变量，跳过重命名")

        # cast & 单位换算
        ds = ds.astype('float32') * unit

        # coarsen 并求平均
        ds = ds.coarsen(latitude=4, longitude=4, boundary='trim').mean()

        # 写入单日处理文件
        outname = fname.replace('.nc', '_processed_day_1.nc')
        outpath = os.path.join(output_dir, outname)
        ds.to_netcdf(outpath)
        print(f"  → saved {outname}")

    # 合并所有日文件
    print("\n开始合并所有已处理文件...")
    pattern = os.path.join(output_dir, '*_processed_day_1.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        print("  ⚠️ 未找到任何 *_processed_day_1.nc，跳过合并")
        return

    ds_all = xr.open_mfdataset(files, combine='by_coords')
    merged_path = os.path.join(output_dir, 'PERSIANN_processed_combined.nc')
    ds_all.to_netcdf(merged_path)
    print(f"合并完成，保存为：{merged_path}")

if __name__ == '__main__':
    # 输入和输出路径 —— 请根据实际修改
    raw_bin_dir = '/Volumes/DiskShared/Download_Persiann_Daily'
    nc_dir      = '/Volumes/DiskShared/Download_Persiann_Daily/processed_nc'
    proc_dir    = '/Volumes/DiskShared/Download_Persiann_Daily/processed'

    # 1. 转换 .bin.gz → .nc
    convert_persiann_to_nc(raw_bin_dir, nc_dir)

    # 2. 后续处理（重命名、cast、coarsen、合并）
    data_compress_persiann(nc_dir, proc_dir, unit=1)