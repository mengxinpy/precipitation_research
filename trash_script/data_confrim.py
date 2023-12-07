import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
from plt_test import *

path = "F:\\liusch\\ERA5\\1980-2019\\total_precipitation\\"
# path2 = "E:\\1979-1989\\"
start_time = time.time()
# 读取.nc文件
wet_day = np.zeros((721, 1440))
for y in range(1979, 1980):
    for m in range(1, 13):  # 遍历月
        print('正在处理:', y, '年', m, '月')
        # day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
        ds = xr.open_mfdataset(path + str(y) + str(m).zfill(2) + '.nc')
        # dsv = ds.to_array().values
        # 将时间维度从小时转换为天
        # 使用nearest
        # dsr = ds.resample(time='3H').nearest() * 3
        dsr = ds.resample(time='3H').nearest().resample(time='1D').sum(dim='time')
        # dsrv = dsr.to_array().values

        # 保存为新的.nc文件
        # ds.to_netcdf(path + str(y) + str(m).zfill(2) + '_day.nc')
        # 定义阈值
        threshold = 0.001  # 需要根据实际情况设定阈值

        # 使用where()函数创建掩码矩阵
        mask = dsr > threshold
        mask.astype(int)
        # 沿着时间维度求和
        monthly_sum = mask.sum(dim='time')
        wet_day = wet_day + np.squeeze(monthly_sum.to_array().values)
        draw_area_heap(wet_day, '3hour_div3' + str(m))
        ds.close()
        # 保存为新的.nc文件
# np.save(path + '_wetday_1year.npy', wet_day)
end_time = time.time()
print('程序运行时间: ', end_time - start_time)
