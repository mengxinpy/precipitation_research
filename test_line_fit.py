import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# 假设 log_points 是 x 轴的数据点，memory 是 y 轴的数据点。
# 这里我们使用一个简化的例子来展示如何进行操作。
log_points = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
memory = np.random.rand(11)  # 这里用随机数代替实际数据

# 分割点
split_point = 40

# 找到分割点的索引
split_index = np.where(log_points == split_point)[0][0]

# 分割数据
x1, y1 = log_points[:split_index+1], memory[:split_index+1]
x2, y2 = log_points[split_index:], memory[split_index:]

# 对数据进行对数转换
log_x1, log_y1 = np.log(x1), np.log(y1)
log_x2, log_y2 = np.log(x2), np.log(y2)

# 进行线性拟合
slope1, intercept1, _, _, _ = linregress(log_x1, log_y1)
slope2, intercept2, _, _, _ = linregress(log_x2, log_y2)

# 将拟合的线转换回非对数形式
fit_y1 = np.exp(intercept1) * x1**slope1
fit_y2 = np.exp(intercept2) * x2**slope2

# 绘制原始数据和拟合线
plt.plot(log_points, memory, 'o', label='Original Data')
plt.plot(x1, fit_y1, 'r-', label=f'Fit Before x={split_point}')
plt.plot(x2, fit_y2, 'b-', label=f'Fit After x={split_point}')

# 设置图例和对数坐标轴
plt.legend()
plt.xscale('log')
plt.yscale('log')

# 显示图表
plt.show()

# 输出斜率
print(f'Slope before x={split_point}:', slope1)
print(f'Slope after x={split_point}:', slope2)
