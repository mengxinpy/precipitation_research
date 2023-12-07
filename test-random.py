import numpy as np

# 生成一个 (3, 3, 3) 的 1-10 整数随机矩阵
random_matrix_3x3x3 = np.random.randint(1, 11, size=(3, 3, 3))

# 使用 roll 在第一个维度上移动 2 位
rolled_matrix_3x3x3 = np.roll(random_matrix_3x3x3, shift=2)

# random_matrix_3x3x3, rolled_matrix_3x3x3
