import numpy as np
import matplotlib.pyplot as plt
import nolds

# 生成一些示例数据
np.random.seed(0)
data = np.cumsum(np.random.randn(1000))

# 步骤1：绘制原始时间序列
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(data)
plt.title("Original Time Series")
plt.xlabel("Time")
plt.ylabel("Value")

# 步骤2：计算去趋势波动分析
# 使用 nolds.dfa 进行 DFA 分析
dfa = nolds.dfa(data, order=1)


# 手动计算去趋势波动分析的步骤
def detrended_fluctuation_analysis(data, nvals):
    N = len(data)
    F_n = []
    for n in nvals:
        segments = N // n
        F_n_vals = []
        for i in range(segments):
            segment = data[i * n:(i + 1) * n]
            x = np.arange(n)
            p = np.polyfit(x, segment, 1)  # 一阶多项式拟合
            trend = np.polyval(p, x)
            fluctuation = segment - trend
            F_n_vals.append(np.sqrt(np.mean(fluctuation ** 2)))
        F_n.append(np.mean(F_n_vals))
    return F_n


# 窗口大小
nvals = np.logspace(1, 2, num=20, dtype=int)
F_n = detrended_fluctuation_analysis(data, nvals)

# 步骤3：绘制去趋势后的时间序列
plt.subplot(2, 2, 2)
detrended_data = data - np.polyval(np.polyfit(np.arange(len(data)), data, 1), np.arange(len(data)))
plt.plot(detrended_data)
plt.title("Detrended Time Series")
plt.xlabel("Time")
plt.ylabel("Value")

# 步骤4：绘制波动函数F(n)随窗口大小n的变化
plt.subplot(2, 2, 3)
plt.plot(nvals, F_n, 'bo-', label="F(n)")
plt.xscale('log')
plt.yscale('log')
plt.title("Fluctuation Function F(n)")
plt.xlabel("Window size (n)")
plt.ylabel("Fluctuation F(n)")

# 步骤5：对数对数图及拟合直线
log_n = np.log(nvals)
log_F_n = np.log(F_n)
slope, intercept = np.polyfit(log_n, log_F_n, 1)

plt.subplot(2, 2, 4)
plt.plot(log_n, log_F_n, 'bo-', label="Log-Log plot")
plt.plot(log_n, slope * log_n + intercept, 'r-', label=f'Fit: slope={slope:.2f}')
plt.title("Log-Log Plot of F(n) vs n")
plt.xlabel("log(n)")
plt.ylabel("log(F(n))")
plt.legend()

plt.tight_layout()
plt.show()

print(f"DFA exponent (slope of the log-log plot): {slope:.2f}")
