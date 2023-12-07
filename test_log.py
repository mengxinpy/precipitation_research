import numpy as np

# 定义范围和点的数量
start = 1
end = 180
num_points = 30

# 计算对数底数
base = end ** (1 / (num_points - 1))
# 生成对数点并取最近的整数
log_points = np.unique([int(round(start * base ** n)) for n in range(num_points)])
# 转换为排序列表
# unique_log_points = sorted(list(log_points))
# print(np.unique(log_points))
# print(len(np.unique(log_points)))
