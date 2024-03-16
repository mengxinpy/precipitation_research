import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import os
import fnmatch

start_time = time.time()
# 设置开始和结束日期
# start_date = datetime(2001, 1, 1)
# end_date = datetime(2001, 1, 2)

path = "E:\\ERA5\\1980-2019\\total_precipitation\\"
from plt_temp import era5_draw_area_dataArray

# path2 = "E:\\1979-1989\\"
# 遍历日期
# 打开所有.nc文件
# dataset = xr.open_mfdataset(path)
#
# # 读取变量
# precipitation = dataset['HQprecipitation']
# wet_day_mask = precipitation.where(precipitation > 0.001, 0)
# wet_day_mask = wet_day_mask.where(precipitation < 0.001, 1)
# wet_day_mask = wet_day_mask.sum(dim='time') / wet_day_mask.shape[0]
# wet_day_mask.to_netcdf('wet_day_mask_imerg.nc')
# imerg_paper=xr.open_dataset('F:\\liusch\\remote_project\\climate_new\\precipitationnature-v2 (1)\\CameronMcE-precipitationnature-8349226\\Extended Data\\EDF6\\IMERG_wet_day_frequency.nc')
#  ipv=imerg_pacmorph_proper['precipitationCal'].values

# merra2_paper=xr.open_dataset('F:\\liusch\\remote_project\\climate_new\\precipitationnature-v2 (1)\\CameronMcE-precipitationnature-8349226\\Extended Data\\EDF6\\MERRA2_wet_day_frequency.nc')
# mpv=merra2_paper['PRECTOT'].values
# merra2_me = xr.open_mfdataset('E:\\ERA5\\1980-2019\\total_precipitation\\*_processed_day_0.5.nc')['tp']
# nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*.nc') and '_processed_day' not in f]
nc_files = [os.path.join(path, f) for f in os.listdir(path) if fnmatch.fnmatch(f, '*_processed_day_0.25.nc') and '_processed_day_processed_day' not in f]
merra2_me = xr.open_mfdataset(nc_files)['tp']
# mpv = merra2_me.values
cmorph_process = merra2_me.where(merra2_me > 1, 0)
cmorph_process = cmorph_process.where(merra2_me < 1, 1)
cmorph_process = cmorph_process.sum(dim='time') / cmorph_process.shape[0]
era5_draw_area_dataArray(cmorph_process > 0.3, 'era5_frequency_30')
cmorph_process.to_netcdf('era5_frequency.nc')
# cmv= cmorph_process.values
# np.save('wet_day_mask.npy', wet_day_mask)
# prv1 = wet_day_mask[:, :].values
# print(prv1)
#
# current_date = start_date
# while current_date <= end_date:
#     # year, month, day = current_date.year, current_date.month, current_date.day
#     formatted_date = current_date.strftime('%Y%m%d')  # 格式化日期
#     current_date += timedelta(days=1)
#     ds = xr.open_mfdataset(path+'3B-DAY.MS.MRG.3IMERG.'+ str(formatted_date)+ '-S000000-E235959.V06.nc4.SUB')
#     print(ds)
# 读取.nc文件
# wet_day = np.zeros((721, 1440))
# for y in range(2001, 2001):
#     for m in range(1, 2):  # 遍历月
#         print('正在处理:', y, '年', m, '月')
#         # day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
#
#         # 将时间维度从小时转换为天
#         ds = ds.resample(time='1D').sum(dim='time')
#
#         # 保存为新的.nc文件
#         ds.to_netcdf(path + str(y) + str(m).zfill(2) + '_day.nc')
#         ds.close()
#         # 定义阈值
#         threshold = 0.001  # 需要根据实际情况设定阈值
#
#         # 使用where()函数创建掩码矩阵
#         mask = ds > threshold
#         mask.astype(int)
#         # 沿着时间维度求和
#         monthly_sum = mask.sum(dim='time')
#         wet_day = wet_day + np.squeeze(monthly_sum.to_array().values)

# 保存为新的.nc文件
# np.save(path + '_wetday_1year.npy', wet_day)
end_time = time.time()
print('程序运行时间: ', end_time - start_time)
