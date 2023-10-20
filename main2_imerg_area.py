import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import glob

result = np.zeros((100, 100))
for area_num in range(100):
    # 获取该地区的所有文件
    print(area_num)
    # if area_num == 7:
    #     continue
    files = glob.glob(f'temp_data\\imerg_processed_data_area{area_num}_part*.nc')
    all_data = []
    for file in files:
        print(file)
        # 打开文件
        ds = xr.open_dataset(file)
        # 提取你感兴趣的变量
        data = ds['precipitationCal'].values
        # 将数据转化为一维数组
        data = data.flatten()
        # 将数据添加到列表中
        all_data.append(data)
    # 使用NumPy的concatenate函数来合并所有的数组
    # 读取所有文件
    # ds = xr.open_mfdataset(files, combine='nested', concat_dim='new_dim')['precipitationCal']
    # ds = ds[ds > 1]
    if np.size(all_data) == 0:
        result[area_num, :] = np.nan
    else:
        all_data = np.concatenate(all_data)
        all_data = all_data[all_data > 1]
        result[area_num, :] = np.nanpercentile(all_data, np.arange(1, 101))
np.save('imerg_percentile_area.npy', result)
# 注意初始值要乘1000
# path = "F:\\liusch\\IMEGR5oringinal\\"  # path2 = "E:\\1979-1989\\" start_time = time.time()
# # 读取.nc文件
# wet_day = np.zeros((721, 1440))
# ds = np.squeeze(xr.open_mfdataset(path + '*_processed.nc4')['precipitationCal'].values)
# result = np.zeros((len(bins), 100))
# for i in range(len(bins)):
#     print(i)
#     rainfall_in_this_area = ds[:, indices == i + 1]
#     rainfall_in_this_area = rainfall_in_this_area[rainfall_in_this_area > 1]
#     if np.size(rainfall_in_this_area) == 0:
#         print('hah')
#         result[i] = np.nan
#     else:
#         result[i] = np.nanpercentile(rainfall_in_this_area, np.arange(1, 101))
#
# np.save('imerg_float16_percentile', result)
# # for y in range(1979, 1989):
# #     for m in range(1, 13):  # 遍历月
# #         print('正在处理:', y, '年', m, '月')
# #         # day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
# #         for i in range(len(bins)):
# #             histogram[i, :] += np.histogram(rainfall_in_this_area, bins=rainfall_bins)[0]
# #
# #         ds.close()
# #         # 保存为新的.nc文件
# # # 计算CDF
# # cdf = np.cumsum(histogram, axis=1) / np.sum(histogram, axis=1)[:, np.newaxis]
# # for i in range(histogram.shape[0]):
# #     plt.plot(rainfall_bins[1:], cdf[i], label=f'Line {i + 1}')
# # # 绘制CDF图
# # plt.xlim(left=1)
# # plt.xscale('log')
# # plt.xlabel('Log of rainfall amount (mm/day)')
# # plt.ylabel('CDF')
# # plt.show()
# # end_time = time.time()
# # print('程序运行时间: ', end_time - start_time)
