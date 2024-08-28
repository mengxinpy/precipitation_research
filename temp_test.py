import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 模拟降水时间序列数据
np.random.seed(42)
time_series_length = 1000
precipitation_data = np.random.gamma(shape=2.0, scale=1.0, size=time_series_length)

# 计算降水变异性
mean_precipitation = np.mean(precipitation_data)
std_precipitation = np.std(precipitation_data)
variability = std_precipitation / mean_precipitation

# 计算降水持续时间
threshold = 0.5  # 降水事件的阈值
events = precipitation_data > threshold
event_durations = []
current_duration = 0

for event in events:
    if event:
        current_duration += 1
    else:
        if current_duration > 0:
            event_durations.append(current_duration)
            current_duration = 0

if current_duration > 0:
    event_durations.append(current_duration)

mean_duration = np.mean(event_durations)

# 输出结果
print(f"降水变异性: {variability:.2f}")
print(f"平均降水持续时间: {mean_duration:.2f}")

# 绘制时间序列图
plt.figure(figsize=(12, 6))
plt.plot(precipitation_data, label='Precipitation')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.legend()
plt.show()
