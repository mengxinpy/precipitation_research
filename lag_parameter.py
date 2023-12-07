import numpy as np
import xarray as xr

# 定义范围和点的数量
start = 1
end = 180
num_points = 30

# 计算对数底数
base = end ** (1 / (num_points - 1))
# 生成对数点并取最近的整数
log_points = np.unique([int(round(start * base ** n)) for n in range(num_points)])
rainfall_frequency = xr.open_dataset('era5_frequency_1.nc').to_array().values.squeeze()
bins = np.arange(0, np.max(rainfall_frequency), 0.17)  # 参数
indices = np.digitize(rainfall_frequency, bins)
area6_percentile = np.load('ear5_percentile_area_1deg_6area.npy')
# 选择特定的列创建一个新的二维数组
selected_columns = [49, 59, 69, 79]
area_top_per_all = area6_percentile[:, selected_columns].T
# 转换为排序列表
