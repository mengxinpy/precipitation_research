import matplotlib.pyplot as plt

# 生成数据
x_values = [1, 2, 3, 4, 5]
y_values = [1, 4, 9, 16, 25]

# 绘制线图
plt.plot(x_values, y_values, marker='o')

# 添加注释
plt.annotate('这里是(3, 9)', xy=(3, 9), xytext=(3.5, 10),
             arrowprops=dict(facecolor='black', shrink=0.05))

# 显示图形
plt.show()
