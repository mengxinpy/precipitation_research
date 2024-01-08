import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as stats

start_time = time.time()
pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"
wetday = np.load('_wetday_40year.npy') / (365 * 40 + 10)
wetday_bins = np.arange(0, np.max(wetday), 0.1)  # num是你想要的分组数量
wetday_indices = np.digitize(wetday, wetday_bins)
vapor_bins_count = 210
results = np.zeros((len(wetday_bins), vapor_bins_count))
count = np.zeros((len(wetday_bins), vapor_bins_count))
results_p4 = np.zeros((len(wetday_bins), vapor_bins_count))
results_p2 = np.zeros((len(wetday_bins), vapor_bins_count))
# rain_results = np.zeros((len(wetday_bins), vapor_bins_count))
# rain_count = np.zeros((len(wetday_bins), vapor_bins_count))
vapor_bins = np.linspace(50, 80, num=vapor_bins_count)  # num是你想要的分组数量
# Calculate the area weights
weight1 = xr.open_dataset(pathr + '198001_processed_day.nc').squeeze()
weights_1d = np.cos(np.deg2rad(weight1.latitude)).values
# expand to 2D
weights = weights_1d[:, np.newaxis] * np.ones(weight1.longitude.size)
for y in range(2010, 2020):
    for m in range(1, 13):  # 遍历月
        print('正在处理:', y, '年', m, '月')
        vapor = xr.open_dataset(pathv + str(y) + str(m).zfill(2) + '_processed_day.nc')
        rainfall = xr.open_dataset(pathr + str(y) + str(m).zfill(2) + '_processed_day.nc')
        for i in range(1, len(wetday_bins) + 1):
            print('bins:', i)
            select_region = wetday_indices == i
            vapor_region = np.squeeze(vapor.to_array().values)[:, select_region]
            rainfall_region = np.squeeze(rainfall.to_array().values)[:, select_region]
            # weights_region = weights[select_region]

            select_region_1mm = rainfall_region > 1
            rain_area_vapor = vapor_region[select_region_1mm]
            rain_area = rainfall_region[select_region_1mm]
            # weights_region = weights_region[, :
            # weights_3d = weights_region[np.newaxis, :] * np.ones((vapor.time.size, 1))
            # weights_region_1mm = weights_3d[select_region_1mm]

            # rain_vapor_indices = np.digitize(rain_area_vapor, vapor_bins)
            # rain_rainfall_sum = [np.dot(rain_area[rain_vapor_indices == j], weights_region_1mm[rain_vapor_indices == j]) for j in range(1, len(vapor_bins) + 1)]
            # rain_weight_sum = [np.sum(weights_region_1mm[rain_vapor_indices == j]) for j in range(1, len(vapor_bins) + 1)]
            # rain_results[i - 1, :] = rain_results[i - 1, :] + rain_rainfall_sum
            # rain_count[i - 1, :] = rain_count[i - 1, :] + rain_weight_sum

            vapor_indices = np.digitize(rain_area_vapor, vapor_bins)
            rainfall_sum = [np.sum(rain_area[vapor_indices == j]) for j in range(1, len(vapor_bins) + 1)]
            rainfall_p4_sum = [np.sum(rain_area[vapor_indices == j] ** 4) for j in range(1, len(vapor_bins) + 1)]
            rainfall_p2_sum = [np.sum(rain_area[vapor_indices == j] ** 2) for j in range(1, len(vapor_bins) + 1)]
            count_sum = [np.sum(vapor_indices == j) for j in range(1, len(vapor_bins) + 1)]
            results[i - 1, :] = results[i - 1, :] + rainfall_sum
            count[i - 1, :] = count[i - 1, :] + count_sum
            results_p2[i - 1, :] = results_p2[i - 1, :] + rainfall_p2_sum
            results_p4[i - 1, :] = results_p4[i - 1, :] + rainfall_p4_sum
results_final = results / count
Binder_4th_order_Cumulant = 1 - (results_p4 / count) / (3 * (results_p2 / count) ** 2)
# rain_results_final = rain_results / rain_count
np.save('power_10years10_noweight.npy', results_final)
np.save('count_10years10_noweight.npy', count)
np.save('Binder_4th_order_Cumulant10.npy', Binder_4th_order_Cumulant)
# np.save('p2_10years00_noweight.npy', results_p2)
# np.save('weight_power_10years10_1mm.npy', rain_results_final)
end_time = time.time()
print('程序运行时间: ', end_time - start_time)
