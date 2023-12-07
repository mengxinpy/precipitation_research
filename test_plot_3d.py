import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个3D矩阵，每行代表一条线，每条线有两个点，每个点有三个坐标
matrix = np.random.random((10, 2, 3))

# 创建一个3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 对于每一条线，绘制线条
for line in matrix:
    ax.plot(line[:, 0], line[:, 1], line[:, 2])

# 显示图形
plt.show()
