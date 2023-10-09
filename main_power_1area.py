import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.stats as stats

pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "F:\\liusch\\ERA5\\1980-2019\\total_precipitation\\"
# 假设你已经有了三个矩阵vapor, rainfall和wetday
# 导入之后要把单位统一
# vapor = xr.open_mfdataset(path + 'vapor_day.nc')
# rainfall = xr.open_mfdataset(path + 'vapor_day.nc')
# wetday = xr.open_mfdataset(path + 'vapor_day.nc')
# wetday = np.load('_wetday_40year.npy') / (365 * 40 + 11)
# # 注意wetday的区域表示
# # 定义wetday的bins
# wetday_bins = np.arange(np.min(wetday), np.max(wetday), 0.05)  # num是你想要的分组数量
# wetday_indices = np.digitize(wetday, wetday_bins)
vapor_bins_count = 210
# 初始化一个空的列表来保存结果
results = np.zeros((1, vapor_bins_count))
count = np.zeros((1, vapor_bins_count))
vapor_bins = np.linspace(20, 90, num=vapor_bins_count)  # num是你想要的分组数量
for y in range(1980, 1981):
    for m in range(1, 13):  # 遍历月

        print('正在处理:', y, '年', m, '月')
        # ds = xr.open_dataset(path + str(y) + str(m).zfill(2)) * 1000
        vapor = xr.open_dataset(pathv + str(y) + str(m).zfill(2) + '.nc')
        rainfall = xr.open_dataset(pathr + str(y) + str(m).zfill(2) + '.nc') * 1000

        # 对每个wetday区域进行循环
        # 提取这个区域的vapor和rainfall数据
        # vapor_region = np.squeeze(vapor.to_array().values)[:, wetday_indices == i]
        # rainfall_region = np.squeeze(rainfall.to_array().values)[:, wetday_indices == i]
        # vapor_region = vapor.where(xr.DataArray(wetday_indices, dims=('latitude', 'longitude',)) == i, drop=True)
        # # rainfall_region = rainfall[:, wetday_indices == i]
        # rainfall_region = rainfall.where(xr.DataArray(wetday_indices) == i, drop=True)

        # 对这个区域的vapor进行分组
        vapor_indices = np.digitize(vapor.to_array().values, vapor_bins)

        # 计算每个组的降雨率的平均值
        rainfall_sum = [np.sum(rainfall.to_array().values[vapor_indices == j]) for j in range(1, len(vapor_bins) + 1)]
        count_sum = [np.sum(vapor_indices == j) for j in range(1, len(vapor_bins) + 1)]
        # 将结果保存到列表
        results = results + rainfall_sum
        count = count + count_sum
results_final = results / count
np.save('only_power.npy', results_final)
plt.plot(vapor_bins,np.squeeze(results_final))
plt.show()
#
# path = "F:\\liusch\\ERA5\\1980-2019\\total_precipitation\\"
# # path2 = "E:\\1979-1989\\"
# start_time = time.time
# # 定义降雨量的bins
# rainfall_bins = np.logspace  # 对数尺度的bins
# # 对降雨频率进行分类
# bins = np.arange  # 你可以根据实际情况调整
# rainfall_frequency = np.squeeze(np.load('paper.npy')) / 100
#
# indices = np.digitize(rainfall_frequency, bins)
#
# # 初始化直方图
# histogram = np.zeros((len(bins), len(rainfall_bins) - 1))
# # 读取.nc文件
# wet_day = np.zeros((721, 1440))
# for y in range(1979, 1989):
#     for m in range(1, 13):  # 遍历月
#         print('正在处理:', y, '年', m, '月')
#         # day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
#         ds = xr.open_mfdataset(path + str(y) + str(m).zfill(2) + '_day.nc') * 1000
#         for i in range(len(bins)):
#             rainfall_in_this_area = np.squeeze(ds.to_array().values)[:, indices == i + 1]
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
end_time = time.time()
print('程序运行时间: ', end_time - start_time)
