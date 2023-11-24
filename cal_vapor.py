import numpy as np
import xarray as xr
from config_hour import *

pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"

wetday = xr.open_dataset('era5_frequency.nc')['tp'].sel(latitude=slice(flat, -flat))
wetday = wetday.coarsen(longitude=int(deg / 0.25), latitude=int(deg / 0.25), boundary='trim').mean()
# 数据初始化
wetday_bins = np.arange(0, np.max(wetday), wetday_gap)  # num是你想要的分组数量
wetday_indices = np.digitize(wetday, wetday_bins)
results = np.zeros((len(wetday_bins), vapor_bins_count))
count = np.zeros((len(wetday_bins), vapor_bins_count))
results_p4 = np.zeros((len(wetday_bins), vapor_bins_count))
results_p2 = np.zeros((len(wetday_bins), vapor_bins_count))
vapor_bins = np.linspace(vapor_strat, vapor_end, num=vapor_bins_count)  # num是你想要的分组数量

# Calculate the area weights
# weight1 = xr.open_dataset(pathr + '198001_processed_day_0.5.nc').squeeze()
# weights_1d = np.cos(np.deg2rad(wetday.latitude)).values
# expand to 2D
# weights = weights_1d[:, np.newaxis] * np.ones(wetday.longitude.size)
for y in range(start_year, end_year):
    for m in range(1, 13):  # 遍历月
        print('正在处理:', y, '年', m, '月')
        # vapor = xr.open_dataset(f'{pathv}{y}{str(m).zfill(2)}_{time_gap}_{deg}.nc').sel(latitude=slice(flat, -flat))
        # rainfall = xr.open_dataset(f'{pathr}{y}{str(m).zfill(2)}_{time_gap}_{deg}.nc').sel(latitude=slice(flat, -flat))
        vapor = xr.open_dataset(f'{pathv}{y}{str(m).zfill(2)}hour_{deg}.nc').sel(latitude=slice(flat, -flat))
        rainfall = xr.open_dataset(f'{pathr}{y}{str(m).zfill(2)}hour_{deg}.nc').sel(latitude=slice(flat, -flat)) * 1000
        # vapor = xr.open_dataset(pathv + str(y) + str(m).zfill(2) + '_processed_day_' + str(deg) + '.nc').sel(latitude=slice(flat, -flat))
        # rainfall = xr.open_dataset(pathr + str(y) + str(m).zfill(2) + '_processed_day_' + str(deg) + '.nc').sel(latitude=slice(flat, -flat))
        for i in range(1, len(wetday_bins) + 1):
            print('bins:', i)

            select_region = wetday_indices == i
            vapor_region = np.squeeze(vapor.to_array().values)[:, select_region]
            rainfall_region = np.squeeze(rainfall.to_array().values)[:, select_region]
            # 假设你的数据是 data，长度是24的整数倍
            # reshaped_data = np.reshape(rainfall_region, (31, 24, rainfall_region.shape[-1]))
            # summed_data = np.sum(reshaped_data, axis=1).squeeze()
            select_region_1mm = rainfall_region > rain_threshold
            rain_area_vapor = vapor_region[select_region_1mm]
            rain_area = rainfall_region[select_region_1mm]

            vapor_indices = np.digitize(rain_area_vapor, vapor_bins)
            rainfall_sum = [np.sum(rain_area[vapor_indices == j]) for j in range(1, len(vapor_bins) + 1)]
            rainfall_p4_sum = [np.sum(rain_area[vapor_indices == j] ** 4) for j in range(1, len(vapor_bins) + 1)]
            rainfall_p2_sum = [np.sum(rain_area[vapor_indices == j] ** 2) for j in range(1, len(vapor_bins) + 1)]
            count_sum = [np.sum(vapor_indices == j) for j in range(1, len(vapor_bins) + 1)]

            results[i - 1, :] = results[i - 1, :] + rainfall_sum
            count[i - 1, :] = count[i - 1, :] + count_sum
            results_p2[i - 1, :] = results_p2[i - 1, :] + rainfall_p2_sum
            results_p4[i - 1, :] = results_p4[i - 1, :] + rainfall_p4_sum
avg_rain = results / count
second_moment = results_p2 / count
binder_4th_order_cumulant = 1 - (results_p4 / count) / (3 * second_moment ** 2)
var = second_moment - avg_rain ** 2
# rain_results_final = rain_results / rain_count
np.save(f'.\\temp_data\\power{start_year}-{end_year}_{deg}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}_avg_rain.npy', avg_rain)
np.save(f'.\\temp_data\\power{start_year}-{end_year}_{deg}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}_count.npy', count)
np.save(f'.\\temp_data\\power{start_year}-{end_year}_{deg}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}_binder.npy', binder_4th_order_cumulant)
np.save(f'.\\temp_data\\power{start_year}-{end_year}_{deg}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}_var.npy', var)
# np.save(
#     '.\\temp_data\\' + 'power' + str(start_year) + '-' + str(end_year) + '_' + str(deg) + '_' + str(vapor_bins_count) + '_' + str(wetday_gap) + '_' + str(flat) + '_avg_rain.npy',
#     avg_rain)
# np.save('.\\temp_data\\' + 'power' + str(start_year) + '-' + str(end_year) + '_' + str(deg) + '_' + str(vapor_bins_count) + '_' + str(wetday_gap) + '_' + str(flat) + '_count.npy',
#         count)
# np.save('.\\temp_data\\' + 'power' + str(start_year) + '-' + str(end_year) + '_' + str(deg) + '_' + str(vapor_bins_count) + '_' + str(wetday_gap) + '_' + str(flat) + 'binder.npy',
#         binder_4th_order_cumulant)
# np.save('.\\temp_data\\' + 'power' + str(start_year) + '-' + str(end_year) + '_' + str(deg) + '_' + str(vapor_bins_count) + '_' + str(wetday_gap) + '_' + str(flat) + 'var.npy',
#         var)

# np.save('count_10years80_1mm_0.5.npy', count)
# np.save('Binder_4th_order_Cumulant_10years80_0.5.npy', Binder_4th_order_Cumulant)
# np.save('p2_10years00_noweight.npy', results_p2)
# np.save('weight_power_10years10_1mm.npy', rain_results_final)
# end_time = time.time()
# elapsed_time = end_time - start_time  # 计算程序运行时间
# minutes, seconds = divmod(elapsed_time, 60)  # 转换为分和秒
# print(f'程序运行时间: {int(minutes)}分{int(seconds)}秒')
