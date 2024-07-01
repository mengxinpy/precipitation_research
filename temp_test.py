import matplotlib.pyplot as plt
import numpy as np

#生成数据
x = np.logspace(0.1, 1, 100)
y = x**2

#进行直线拟合
logx = np.log10(x)
logy = np.log10(y)
coeffs = np.polyfit(logx, logy, 1)
poly = np.poly1d(coeffs)

#绘制双对数坐标下的分布图
plt.loglog(x, y, label='Original data')
plt.loglog(x, 10**poly(logx), label='Fitted line')

#设置图例和标题
plt.legend(loc='best')
plt.title('Log-Log plot with fitted line')
plt.xlabel('x')
plt.ylabel('y')
# plt.grid(True)
plt.show()
