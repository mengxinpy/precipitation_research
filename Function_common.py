import numpy as np
import numpy.fft as fft
import xarray as xr
from numpy.core.numeric import newaxis


def calculate_gradient(data_array):
    if not isinstance(data_array, xr.DataArray):
        raise ValueError("Input must be an xarray DataArray.")

    # 获取纬度和经度
    lat = data_array.latitude
    lon = data_array.longitude

    # 计算纬度和经度的间隔（假设均匀间隔）
    # 不翻转纬度，直接计算梯度
    dlat = np.gradient(lat)
    dlon = np.gradient(lon)

    # 计算 u 和 v 分量
    u = data_array.differentiate('longitude') / dlon
    v = data_array.differentiate('latitude') / dlat[:,np.newaxis]

    # 创建一个新的 Dataset
    gradient_ds = xr.Dataset(
        {
            'u': u,
            'v': v
        },
        coords={
            'latitude': data_array.latitude,
            'longitude': data_array.longitude
        }
    )

    return gradient_ds


def gradient_direction_comparison(da1, da2):
    # 计算梯度
    grad_y1, grad_x1 = np.gradient(da1)
    grad_y2, grad_x2 = np.gradient(da2)

    # 计算梯度方向的内积
    dot_product = grad_x1 * grad_x2 + grad_y1 * grad_y2

    # 计算梯度模
    magnitude1 = np.sqrt(grad_x1 ** 2 + grad_y1 ** 2)
    magnitude2 = np.sqrt(grad_x2 ** 2 + grad_y2 ** 2)

    # 避免除以零
    epsilon = 1e-10
    magnitude1 = np.where(magnitude1 == 0, epsilon, magnitude1)
    magnitude2 = np.where(magnitude2 == 0, epsilon, magnitude2)

    # 归一化内积
    normalized_dot_product = dot_product / (magnitude1 * magnitude2)

    # 创建新的 DataArray，复制 da1 的属性和坐标
    result = xr.DataArray(
        normalized_dot_product,
        dims=da1.dims,
        coords=da1.coords,
        attrs=da1.attrs
    )

    return result


def get_da_list(key_list):
    da_list = []
    for key in key_list:
        file_name = f'.\\internal_data\\{key}_era5\\{key}_frequency.nc'
        file_data = xr.open_dataarray(file_name)
        if key in ['duration', 'quiet', 'intensity']:
            file_data = np.log(file_data)
        da_list.append(file_data)
    return da_list


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
