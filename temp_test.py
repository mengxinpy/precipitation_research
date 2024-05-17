import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 2 * np.pi, 10)
y = np.sin(x)

# 颜色列表
colors = ['b', 'g', 'r', 'y']

# 创建茎叶图，为每个点指定颜色
for i in range(len(x)):
    plt.stem([x[i]], [y[i]], linefmt=colors[i % len(colors)], markerfmt='o'+colors[i % len(colors)], basefmt=" ")

plt.show()