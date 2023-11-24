import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from config import *
from functools import partial


# 定义模型函数
# 定义模型函数，不包括 wc 作为参数
def power_law(w, a, b):
    return a * ((w - wc) / wc) ** b


# 示例数据（请替换为您的实际数据）
w = np.linspace(vapor_strat, vapor_end, vapor_bins_count)[0:130]
y = np.load(f".\\temp_data\\amsr2_power_{deg}_{vapor_bins_count}_{flat}_avg_rain.npy").squeeze()[0:130]

# 初始临界点
wc_initial = 71

# 迭代拟合
wc = wc_initial
last_error = float('inf')
for i in range(100):  # 限制最大迭代次数为 10
    # 筛选大于 wc 的数据
    print(f"第{i}次迭代")
    w_fit = w[w > wc]
    y_fit = y[w > wc]

    # partial_func = partial(power_law, wc=wc)
    # 检查是否有 NaN 或无穷大值
    # if np.any(np.isnan(y_fit)) or np.any(np.isinf(y_fit)):
    #     print("y_fit 中检测到非有限值。退出循环。")
    #     break
    lower_bounds = [0, 0]  # a, wc, and b all have lower bounds of 0
    upper_bounds = [np.inf, 1]  # a has no upper bound, wc has an upper bound of wc_initial + 5, b has an upper bound of 1

    # 拟合
    popt, pcov = curve_fit(power_law, w_fit, y_fit, bounds=(lower_bounds, upper_bounds))
    print(popt)
    # 计算拟合误差
    residuals = y_fit - power_law(w_fit, *popt)
    current_error = np.sum(residuals ** 2) / residuals.size

    # 检查误差是否减少
    if current_error < last_error:
        last_error = current_error
        # 根据实际情况调整 wc
        wc -= 0.5  # 例如，每次减少1，您需要根据实际情况调整这个值
    else:
        break  # 如果误差没有改善，停止迭代

# 绘图展示拟合效果
plt.scatter(w, y, label='原始数据')
# 从临界点 wc 到原始数据的最大值创建一个新的 w 数组
w_new = np.linspace(wc, max(w), 500)  # 500是点的数量，可以根据需要调整

# 使用拟合参数计算模型的预测值
y_pred = power_law(w_new, *popt)
plt.plot(w_new, y_pred, label='拟合曲线', color='red')
plt.axvline(x=wc, color='green', linestyle='--', label=f'临界点: wc = {wc:.2f}')
plt.legend()
plt.show()
