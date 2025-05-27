import xarray as xr
import os
import fnmatch
import glob

def data_compress_MSWEP(path, output_dir, unit=1):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 列出所有 .nc 文件
    nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*.nc')]

    for nc_file in nc_files:
        print("Processing:", nc_file)
        ds = xr.open_dataset(nc_file)

        # 重命名维度为全称
        ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})

        # 重命名数据变量
        if 'precipitation' in ds.data_vars:
            ds = ds.rename({'precipitation': 'tp'})
        else:
            print("  Warning: 'precipitation' not found in", nc_file)

        # 转成 float32 并乘以单位
        ds = ds.astype('float32') * unit

        # 以 10×10 网格块降尺度并取平均
        ds = ds.coarsen(longitude=10, latitude=10, boundary='trim').mean()

        # 构造输出文件路径并保存
        basename = os.path.basename(nc_file).replace('.nc', '_processed_day_1.nc')
        new_nc_file = os.path.join(output_dir, basename)
        ds.to_netcdf(new_nc_file)
        print("Saved to:", new_nc_file)

    # —— 以下为新增：合并所有已处理文件并保存为一个单文件 ——
    print("\n开始合并所有已处理文件...")
    processed_files = sorted(glob.glob(os.path.join(output_dir, '*_processed_day_1.nc')))
    if not processed_files:
        print("未找到任何已处理文件，跳过合并。")
        return

    # 按坐标自动拼接（适用于有时间等坐标维度的文件）
    ds_combined = xr.open_mfdataset(processed_files, combine='by_coords')

    combined_path = os.path.join(output_dir, 'MSWEP_processed_combined.nc')
    ds_combined.to_netcdf(combined_path)
    print("合并完成，保存为：", combined_path)

if __name__ == '__main__':
    input_path = '/Volumes/DiskShared/MSWEP_Daily'
    output_path = '/Volumes/DiskShared/MSWEP_Daily/processed'
    data_compress_MSWEP(path=input_path, output_dir=output_path, unit=1)
