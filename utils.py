import numpy as np
from scipy.stats import t, chi2
import numpy.fft as fft
import xarray as xr
from numpy.core.numeric import newaxis
from config import intern_data_path


# def percentile_th_all(dr, top_bins=(40, 30, 20, 10)):
#     condition_all_top = np.zeros((len(top_bins), dr.sizes['latitude'], dr.sizes['longitude']), dtype=bool)
#     # Assume `top_bins` and `dr` are defined as in your context
#     latitude = dr.sizes['latitude']
#     longitude = dr.sizes['longitude']
#
#     # Create a DataArray filled with zeros
#     all_th = xr.DataArray(
#         np.zeros((len(top_bins), latitude, longitude)),
#         dims=['top_bins', 'latitude', 'longitude'],
#         coords={'top_bins': list(top_bins), 'latitude': dr.coords['latitude'], 'longitude': dr.coords['longitude']}
#     )
#     for p, per in enumerate(top_bins):
#         condition_all_top[p], all_th.loc[dict(top_bins=per)] = condition_above_percentile(dr, percentile=per)  # todo:可能会导致赋值错误
#     np.save(f'{intern_data_path}condition_all_top', condition_all_top)
#     all_th.to_netcdf(f'{intern_data_path}all_th.nc')
#     return condition_all_top, all_th

def percentile_th_all(dr, top_bins=(40, 30, 20, 10)):
    latitude = dr.sizes['latitude']
    longitude = dr.sizes['longitude']

    # Create a DataArray filled with zeros
    all_th = xr.DataArray(
        np.zeros((len(top_bins), latitude, longitude)),
        dims=['top_bins', 'latitude', 'longitude'],
        coords={'top_bins': list(top_bins), 'latitude': dr.coords['latitude'], 'longitude': dr.coords['longitude']}
    )
    for p, per in enumerate(top_bins):
        _, all_th.loc[dict(top_bins=per)] = condition_above_percentile(dr, percentile=per)  # todo:可能会导致赋值错误
    all_th.to_netcdf(f'{intern_data_path}all_th.nc')
    return all_th


def depart_ml_lat(data_list):
    low_lat_list = []
    mid_lat_list = []

    for data in data_list:
        low_lat = data.sel(latitude=slice(30, -30))
        south_hemisphere = data.sel(latitude=slice(-30, -60))
        north_hemisphere = data.sel(latitude=slice(60, 30))
        combined = xr.concat([north_hemisphere, south_hemisphere], dim="latitude")
        mid_lat = combined.sortby('latitude')

        low_lat_list.append(low_lat)
        mid_lat_list.append(mid_lat)

    return low_lat_list, mid_lat_list

def point_path_data(var, dataset='era5', lat=60):
    if dataset == 'era5':
        data_path = '/Volumes/DiskShared/ERA5/1980-2019/total_precipitation/'
        data_point = xr.open_mfdataset(data_path + '*processed_day_1.nc') \
            .sel(latitude=slice(lat, -lat))['tp']
    elif dataset == 'mswep':
        data_path = '/Volumes/DiskShared/MSWEP_Daily/processed_combined/'
        varname = ''.join(word[0] for word in var.split('_') if word)
        data_point = xr.open_mfdataset(data_path + 'MSWEP_processed_combined_align.nc') \
            [varname].sel(latitude=slice(lat, -lat))
    elif dataset == 'persiann':
        data_path = '/Volumes/DiskShared/Download_Persiann_Daily/processed_combined/'
        varname = ''.join(word[0] for word in var.split('_') if word)
        data_point = xr.open_mfdataset(data_path + 'PERSIANN_processed_combined.nc') \
            [varname].sel(latitude=slice(lat, -lat))
    elif dataset == 'yossi':
        path_all = '/Users/kustai/PycharmProjects/ERA5/1980-2019/total_precipitation_origin/'
        varname = ''.join(word[0] for word in var.split('_') if word)
        data_point = xr.open_mfdataset(path_all + '*.nc') \
            [varname].sel(latitude=slice(lat, -lat))
    elif dataset == 'hour':
        path_all = 'F:/ZZBO/TP/'
        varname = ''.join(word[0] for word in var.split('_') if word)
        data_point = (xr.open_mfdataset(path_all + '*.nc')[varname]
                      .sel(latitude=slice(lat, -lat),
                           time=slice('2010-01-01', '2010-01-31'))
                      .coarsen(latitude=4, boundary='trim')
                      .mean())
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            "Please choose one of: 'ear5', 'mswep', 'persiann', 'yossi', 'hour'."
        )

    return data_point


def decode_list(key_list,dataset='era5'):
    files_names = []
    for key in key_list:
        if key in ['season', 'wet', 'duration', 'power', 'intensity']:
            files_names.append(f'./internal_data/{key}_{dataset}/{key}_frequency.nc')
        else:
            files_names.append(f'./internal_data/wet_{dataset}/{key}_frequency.nc')

    return files_names



def convert_longitude(lon):
    if lon < 0:
        return 360 + lon
    else:
        return lon

def depart_ml_lat(data_list):
    low_lat_list = []
    mid_lat_list = []

    for data in data_list:
        low_lat = data.sel(latitude=slice(30, -30))
        south_hemisphere = data.sel(latitude=slice(-30, -60))
        north_hemisphere = data.sel(latitude=slice(60, 30))
        combined = xr.concat([north_hemisphere, south_hemisphere], dim="latitude")
        mid_lat = combined.sortby('latitude')

        low_lat_list.append(low_lat)
        mid_lat_list.append(mid_lat)

    return low_lat_list, mid_lat_list

def get_list_form_onat(onat_list, dr, percentile_th, top_bins=30):
    dr_list = []
    th_list = []
    # 提取所有的lon和lat
    # lons, lats = zip(*onat_list)
    # 将lon映射到转换后的经度
    # lons = [convert_longitude(lon) for lon in lons]
    lons = [convert_longitude(lon) for lon, lat in onat_list]
    lats = [lat for lon, lat in onat_list]

    # 使用xarray的向量化选择功能进行批量选择
    lon_dim = xr.DataArray(lons, dims="points")
    lat_dim = xr.DataArray(lats, dims="points")
    # 批量选择dr和percentile_th
    percentile_th = percentile_th.sel(top_bins=top_bins, method='nearest')
    drs = dr.sel(longitude=lon_dim, latitude=lat_dim, method='nearest')
    th = percentile_th.sel(longitude=lon_dim, latitude=lat_dim, method='nearest')
    # 将选择后的数据添加到列表中
    # 将选择后的数据展开为单独的时间序列列表
    for i in range(len(drs.points)):
        dr_list.append(drs.isel(points=i))
        th_list.append(th.isel(points=i).values)

    return dr_list, th_list


def convert_onat2drt(onat_list, dr, all_th):
    dr_list = []
    th_list = []

    for lon, lat in onat_list:
        # Select the time series for the specific longitude and latitude
        time_series = dr.sel(longitude=lon, latitude=lat, method="nearest")
        dr_list.append(time_series)

        # Select the threshold values for the specific longitude and latitude
        thresholds = all_th.sel(longitude=lon, latitude=lat, method="nearest").values
        th_list.append(thresholds.tolist())

    return dr_list, th_list

def get_dataset_duration_all(key_list, dataset='ear5', log_duration_etc=True, unify=True, wet_th=False):
    da_list = []
    intensity_condition = xr.open_dataarray(f'./internal_data/intensity_era5/intensity_frequency.nc') > 1
    file_names=[f'./internal_data/wet_{dataset}/{key}_frequency.nc' for key in key_list]
    for file_name in file_names:
        file_data = xr.open_dataarray(file_name)
        if file_data.name in ['duration', 'quiet', 'intensity']:
            if log_duration_etc:
                file_data = np.log10(file_data)
        if unify:
            # 归一化
            file_data = (file_data - file_data.min()) / (file_data.max() - file_data.min())
        if wet_th:
            file_data = file_data.where(intensity_condition, np.nan)
        if file_data.name == 'wet':
            # 假设 file_data 是一个 xarray DataArray
            file_data = file_data.rename('Wet-day frequency')
        if file_data.name == 'duration':
            # 假设 file_data 是一个 xarray DataArray
            file_data = file_data.rename(r'Duration (day)')
            if log_duration_etc:
                file_data = file_data.rename(r'Duration $\log_{10}\,(\mathrm{day})$')
        da_list.append(file_data)

    return da_list


def get_refine_da_list(key_list, dataset='era5', log_duration_etc=True, unify=True, wet_th=False):
    da_list = []
    intensity = xr.open_dataarray(f'./internal_data/intensity_era5/intensity_frequency.nc')
    intensity_condition = intensity > 1
    for file_name in decode_list(key_list,dataset):
        file_data = xr.open_dataarray(file_name)
        if file_data.name in ['duration', 'quiet', 'intensity']:
            if log_duration_etc:
                file_data = np.log10(file_data)
        if unify:
            # 归一化
            file_data = (file_data - file_data.min()) / (file_data.max() - file_data.min())
        if wet_th:
            file_data = file_data.assign_coords({
                'latitude':  intensity.latitude,
                'longitude': intensity.longitude
            })
            file_data = file_data.where(intensity > 1, np.nan)
        if file_data.name == 'wet':
            # 假设 file_data 是一个 xarray DataArray
            file_data = file_data.rename('Wet-day frequency')
        if file_data.name == 'duration':
            # 假设 file_data 是一个 xarray DataArray
            file_data = file_data.rename(r'Duration (day)')
            if log_duration_etc:
                file_data = file_data.rename(r'Duration $\log_{10}\,(\mathrm{day})$')
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
    """
    输入：
        dur: 一维数组或列表，表示降水持续时间数据

    输出：
        bin_centers: 直方图的 bin 中心
        hist: 直方图的频率(密度)值
        mean_dur: dur 的样本均值
        std_dur: dur 的样本标准差(ddof=1)
        ci_mean: 均值的95%置信区间 (下限, 上限)
        ci_std: 标准差的95%置信区间 (下限, 上限)
    """

    # 1) 生成对数刻度的 bins，注意取 unique 以防重复
    bins = np.unique(np.logspace(np.log10(min(dur)), np.log10(max(dur)), 30).round())

    # 2) 计算直方图 (以密度形式 density=True)
    hist, bin_edges = np.histogram(dur, bins=bins, density=True)

    # 3) 计算 bin 的中心位置
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # ========== 计算均值、标准差 ==========
    mean_dur = np.mean(dur)
    # ddof=1 是无偏估计（样本标准差）
    std_dur = np.std(dur, ddof=1)
    n = len(dur)  # 样本大小
    df = n - 1  # 自由度

    # ========== 均值的95%置信区间 (基于 t 分布) ==========
    # alpha = 0.05 => 95% 置信区间
    alpha = 0.05
    # t 分布在右尾的临界值（单尾 0.025）
    t_crit = t.ppf(1 - alpha / 2, df)
    # 标准误 (SE) = std_dur / sqrt(n)
    se = std_dur / np.sqrt(n)
    ci_mean_lower = mean_dur - t_crit * se
    ci_mean_upper = mean_dur + t_crit * se
    ci_mean = (ci_mean_lower, ci_mean_upper)

    # ========== 标准差的95%置信区间 (基于卡方分布) ==========
    # chi2_{0.975} 和 chi2_{0.025}，分别是右尾0.025和0.975的临界值
    chi2_lower = chi2.ppf(1 - alpha / 2, df)  # 0.975分位, 值较大
    chi2_upper = chi2.ppf(alpha / 2, df)  # 0.025分位, 值较小

    # 注意：标准差区间的公式
    #   ( (n-1)*s^2 / chi2_{1-alpha/2} )^(1/2) <= sigma <= ( (n-1)*s^2 / chi2_{alpha/2} )^(1/2)
    ci_std_lower = np.sqrt((df * std_dur ** 2) / chi2_lower)
    ci_std_upper = np.sqrt((df * std_dur ** 2) / chi2_upper)
    ci_std = (ci_std_lower, ci_std_upper)

    return bin_centers, hist, mean_dur, std_dur, ci_mean, ci_std


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
