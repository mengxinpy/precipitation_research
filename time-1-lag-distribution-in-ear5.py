import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

pathv = "F:\\liusch\\ERA5\\1980-2019\\total_column_water_vapour\\"
pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"
# 假设已经通过某种方式定义了pathv和pathr
vapor_month = xr.open_mfdataset(pathv + '198*_processed_day_0.25.nc')
rain_month = xr.open_mfdataset(pathr + '198*_processed_day_0.25.nc')

# 假设数据集中有一个名为'precipitation'的DataArray
precipitation = rain_month['tp']

# 计算百分位数阈值
percentiles = [33, 66]
percentile_values = np.load('era5_percentile_area.npy')
err5_frenquency=xr.open_dataset('era5_frequency.nc')
# percentile_values = np.percentile(precipitation, percentiles)

# 创建一个掩码数组，用于标记每个百分位数范围内的元素
mask_low = precipitation <= percentile_values[0]
mask_mid = (precipitation > percentile_values[0]) & (precipitation <= percentile_values[1])
mask_high = precipitation > percentile_values[1]


# 计算后继时间元素的分布
def calculate_lagged_distribution(precipitation, mask, time_lag):
    print('one_part')
    # 获取时间维度的长度
    time_length = precipitation.sizes['time']

    # 计算有效的时间索引，排除最后time_lag个时间点
    valid_times = np.arange(time_length - time_lag)

    # 使用掩码选择元素，并且排除最后time_lag个时间点
    valid_mask = mask.isel(time=valid_times)

    # 获取掩码为True的位置的索引
    idx = np.where(valid_mask)
    # 获取原始的经度和纬度索引
    longitude_idx = valid_mask.longitude[idx[1]].values
    latitude_idx = valid_mask.latitude[idx[2]].values
    # 根据索引和时间滞后找到对应的后继元素
    lagged_values = precipitation.isel(time=(valid_times[idx[0]] + time_lag), longitude=longitude_idx, latitude=latitude_idx)

    # 移除矩阵中的nan值
    matrix_nonan = lagged_values[~np.isnan(lagged_values)]

    # 计算直方图和bin边界
    hist, bins = np.histogram(matrix_nonan, bins='auto', density=True)

    # 绘制直方图
    plt.hist(matrix_nonan, bins=bins, density=True, alpha=0.5)

    # 如果需要计算其他分布参数，您可以在这里添加代码
    # 例如，计算标准差、偏度等

    return 1


# 设置时间滞后
time_lag = 1

# 计算每个百分位数范围内元素的后继时间点的分布
distribution_low = calculate_lagged_distribution(precipitation, mask_low, time_lag)
# distribution_mid = calculate_lagged_distribution(precipitation, mask_mid, time_lag)
distribution_high = calculate_lagged_distribution(precipitation, mask_high, time_lag)
plt.title('Probability Density Function')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
# 输出分布参数
# print("Distribution mean for low percentile range:", distribution_low)
# print("Distribution mean for mid percentile range:", distribution_mid)
# print("Distribution mean for high percentile range:", distribution_high)
