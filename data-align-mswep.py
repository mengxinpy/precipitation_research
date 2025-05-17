import os
import xarray as xr
import numpy as np

# —— 用户需要修改的部分 —— #
data_path = '/Volumes/DiskShared/MSWEP_Daily/processed_combined/'
orig_fname = 'MSWEP_processed_combined.nc'
var = 'total_precipitation'     # 原始代码里传入的 var，比如 'wet_day_frequency'
lat = 60                      # 你的切片纬度参数
# —————————————————— #

# 自动生成 varname（取每个下划线分段的首字母）
varname = ''.join(word[0] for word in var.split('_') if word)

# 打开数据
ds = xr.open_dataset(os.path.join(data_path, orig_fname))

# 选取 DataArray 并按 lat 切片
da = ds[varname].sel(latitude=slice(lat, -lat))

# 1) 先把经度范围从 [-179.5, 179.5] 映射到 [0, 360)
#    （这里用 (lon + 360) % 360，然后排序）
new_lon = (da.longitude + 360) % 360
da = da.assign_coords(longitude=new_lon).sortby('longitude')

# 2) 如果你同时还想改纬度坐标，或和另一个已有的 DataArray（如 intensity）一致，可以再 assign_coords：
# da = da.assign_coords(latitude=intensity.latitude)

# 3) （可选）改名，让输出里就叫 “Wet-day frequency”

# 4) 保存到原目录，文件名加后缀
out_fname = orig_fname.replace('.nc', '_align.nc')
out_path = os.path.join(data_path, out_fname)
da.to_netcdf(out_path)

print(f"调整后数据已保存到：{out_path}")