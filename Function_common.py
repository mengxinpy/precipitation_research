import numpy as np
import numpy.fft as fft


def condition_above_percentile(data, percentile=30, time_axis=0):
    data = data.chunk({'time': -1})
    thresholds = data.quantile(1 - percentile / 100, dim='time')
    # 初始化标记矩阵，其形状与原始数据相同
    marked_matrix = np.zeros_like(data)

    # 对于每个(lat, lon)点，标记大于阈值的时间点为1，其他为0
    marked_matrix = data > thresholds

    return marked_matrix, thresholds


def get_hist(dur):
    bins = np.unique(np.logspace(np.log10(min(dur)), np.log10(max(dur)), 30).round())
    # 计算直方图数据，density=True 以获取频率
    hist, bin_edges = np.histogram(dur, bins=bins, density=True)
    # 计算 bin 中心点，用于作为 x 轴数据
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, hist


def nan_digitize(data, bins):
    nan_mask = np.isnan(data)

    # 创建一个与 data 大小相同的数组，并初始化为 NaN
    indices = np.full(data.shape, np.nan, dtype=float)

    # 对非 NaN 值进行 digitize 操作
    non_nan_indices = np.digitize(data.values[~nan_mask], bins)

    # 将非 NaN 值的 digitize 结果填充回去
    indices[~nan_mask] = non_nan_indices

    return indices


def get_fft_values(y_values, N, f_s):
    f_values = np.linspace(0.0, f_s, N)
    fft_values_ = fft.fft(y_values, N)
    # power spectrum
    fft_values = np.abs(fft_values_) ** 2
    # 归一化
    fft_values = fft_values / np.sum(fft_values[0:N // 2])

    return f_values[0:N // 2], fft_values[0:N // 2]


def just_spectrum(x):
    fft_poiint_num = 16384
    M = 8180
    # Vh
    ''' 计算频谱 '''
    # f_s = 1 ， 采样频率为 1
    freq, X = get_fft_values(x, fft_poiint_num, 1)
    period = 1 / freq
    period = period[len(period)::-1]
    print('len(period):  ', len(period))
    period = period[0:M]
    inverse_X = X[len(X)::-1]
    inverse_X = inverse_X[0:M]
    return period[0:M] / 365.25, inverse_X[0:M]
