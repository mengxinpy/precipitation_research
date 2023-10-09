import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.stats as stats
start_time = time.time()
pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "F:\\liusch\\ERA5\\1980-2019\\total_precipitation\\"
wetday = np.load('_wetday_40year.npy') / (365 * 40 + 11)
wetday_bins = np.arange(np.min(wetday), np.max(wetday), 0.05)  # num是你想要的分组数量
wetday_indices = np.digitize(wetday, wetday_bins)
vapor_bins_count = 500
results = np.zeros((len(wetday_bins), vapor_bins_count))
count = np.zeros((len(wetday_bins), vapor_bins_count))
vapor_bins = np.linspace(20, 120, num=vapor_bins_count)  # num是你想要的分组数量
for y in range(1980, 1981):
    for m in range(1, 13):  # 遍历月
        print('正在处理:', y, '年', m, '月')
        vapor = xr.open_dataset(pathv + str(y) + str(m).zfill(2) + '.nc')
        rainfall = xr.open_dataset(pathr + str(y) + str(m).zfill(2) + '.nc') * 1000
        for i in range(1, len(wetday_bins) + 1):
            print('bins:', i)
            vapor_region = np.squeeze(vapor.to_array().values)[:, wetday_indices == i]
            rainfall_region = np.squeeze(rainfall.to_array().values)[:, wetday_indices == i]

            vapor_indices = np.digitize(vapor_region, vapor_bins)

            rainfall_sum = [np.sum(rainfall_region[vapor_indices == j]) for j in range(1, len(vapor_bins) + 1)]
            count_sum = [np.sum(vapor_indices == j) for j in range(1, len(vapor_bins) + 1)]

            results[i - 1, :] = results[i - 1, :] + rainfall_sum
            count[i - 1, :] = count[i - 1, :] + count_sum
results_final = results / count
np.save('area_rainfall_power_1month.npy', results_final)
end_time = time.time()
print('程序运行时间: ', end_time - start_time)
