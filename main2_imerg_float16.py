import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time

# 注意初始值要乘1000
path = "F:\\liusch\\IMEGR5oringinal\\"  # path2 = "E:\\1979-1989\\" start_time = time.time()
# 定义降雨量的bins
rainfall_bins = np.logspace(np.log10(1), np.log10(500), num=500)  # 对数尺度的bins
# rainfall_frequency = np.squeeze(np.load('paper.npy')) / 100
rainfall_frequency = np.squeeze(xr.open_dataset('imerg5_process.nc').to_array().values)
# rainfall_frequency = xr.open_dataset('cmporph_process_12.nc')
# 对降雨频率进行分类
bins = np.arange(0, np.max(rainfall_frequency), 0.01)  # 你可以根据实际情况调整
indices = np.digitize(rainfall_frequency, bins)

# 初始化直方图
# histogram = np.zeros((len(bins), len(rainfall_bins) - 1))
# 读取.nc文件
wet_day = np.zeros((721, 1440))
ds = np.squeeze(xr.open_mfdataset(path + '*.nc4')['precipitationCal'].astype('float16').sel(lat=slice(-60, 60)).values)
result = np.zeros((len(bins), 100))
for i in range(len(bins)):
    print(i)
    rainfall_in_this_area = ds[:, indices == i + 1]
    rainfall_in_this_area = rainfall_in_this_area[rainfall_in_this_area > 1]
    if np.size(rainfall_in_this_area) == 0:
        print('hah')
        result[i] = np.nan
    else:
        result[i] = np.nanpercentile(rainfall_in_this_area, np.arange(1, 101))

np.save('imerg_float16_percentile', result)
# for y in range(1979, 1989):
#     for m in range(1, 13):  # 遍历月
#         print('正在处理:', y, '年', m, '月')
#         # day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
#         for i in range(len(bins)):
#             histogram[i, :] += np.histogram(rainfall_in_this_area, bins=rainfall_bins)[0]
#
#         ds.close()
#         # 保存为新的.nc文件
# # 计算CDF
# cdf = np.cumsum(histogram, axis=1) / np.sum(histogram, axis=1)[:, np.newaxis]
# for i in range(histogram.shape[0]):
#     plt.plot(rainfall_bins[1:], cdf[i], label=f'Line {i + 1}')
# # 绘制CDF图
# plt.xlim(left=1)
# plt.xscale('log')
# plt.xlabel('Log of rainfall amount (mm/day)')
# plt.ylabel('CDF')
# plt.show()
# end_time = time.time()
# print('程序运行时间: ', end_time - start_time)
