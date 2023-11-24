import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
from plt_temp import *

path = "F:\\liusch\\remote_project\\climate_new\\precipitationnature-v2 (1)\\CameronMcE-precipitationnature-8349226\\Figure1\\"
# path2 = "E:\\1979-1989\\"
start_time = time.time()
# 读取.nc文件
wet_day = np.zeros((721, 1440))
for y in range(1979, 1980):
    for m in range(1, 2):  # 遍历月
        print('正在处理:', y, '年', m, '月')
        # day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
        ds = xr.open_mfdataset(path + '1980_2019_total_precipitation_masked.nc')
        dsv = ds.to_array().values
        draw_area_heap(np.squeeze(ds.to_array().values), 'paper')
        # 使用where()函数创建掩码矩阵
        ds.close()

        # 保存为新的.nc文件
# np.save(path + '_wetday_1year.npy', wet_day)
end_time = time.time()
print('程序运行时间: ', end_time - start_time)
