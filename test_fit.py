import numpy as np
import matplotlib
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 设置 Matplotlib 使用的字体，这里以 'Microsoft YaHei' 为例
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['font.size'] = 10  # 可以调整字体大小

def power_law(x, a, b):
    return a * x ** b


# 生成 x 数据
x = np.linspace(1, 10, 100)

# 模拟的真实参数
a_true = 2
b_true = 3

# 生成 y 数据并加入噪声
y = power_law(x, a_true, b_true) + np.random.normal(0, 0.5, x.shape)


# 拟合
popt, pcov = curve_fit(power_law, x, y)

# 打印拟合得到的参数
print("拟合参数:", popt)

# 绘制原始数据和拟合曲线
plt.scatter(x, y, label='原始数据')
plt.plot(x, power_law(x, *popt), label='拟合曲线', color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
