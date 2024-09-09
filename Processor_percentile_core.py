import numpy as np
import xarray as xr
import numpy.fft as fft
from numpy.f2py.auxfuncs import throw_error

from Function_common import condition_above_percentile
from Function_calculate_event_durations import calculate_every_event_durations, calculate_event_one_all_durations
from Function_calculate_event_durations import calculate_event_dfa


def get_lspf_frequency(lspf):
    all_time_len = 3600 * 24 * lspf.shape[0]
    lspf = lspf.sum(dim='time')
    lspf_frequency = lspf / all_time_len
    return lspf_frequency


def get_lsprf_frequency(dr, lsp):
    # lspf频率
    lsp_st = lsp.sum(dim='time')
    dr_st = dr.sum(dim='time')
    lsprf_frequency = lsp_st / dr_st
    return lsprf_frequency


def get_cp_frequency(cp, dr):
    cp_st = cp.sum(dim='time')
    dr_st = dr.sum(dim='time')
    cp_frequency = cp_st / dr_st
    return cp_frequency


def get_fft_values(y_values, N, f_s):
    f_values = np.linspace(0.0, f_s, N)
    fft_values_ = fft.fft(y_values, N)
    # power spectrum
    fft_values = np.abs(fft_values_) ** 2
    # 归一化
    fft_values = fft_values / np.sum(fft_values[0:N // 2])

    return f_values[0:N // 2], fft_values[0:N // 2]


def just_spectrum_ratio(x):
    fft_poiint_num = 16384
    M = 8180
    # Vh
    ''' 计算频谱 '''
    # f_s = 1 ， 采样频率为 1
    freq, X = get_fft_values(x, fft_poiint_num, 1)
    inverse_X = X[len(X)::-1]
    a = np.sum(inverse_X[14:50])
    # b = np.sum(inverse_X[0:366])
    # rat = a / b
    # print(f'a:{a} b:{b}')
    return a


def get_power_frequency(dr):
    result_da = xr.apply_ufunc(
        just_spectrum_ratio,  # 应用的自定义函数
        dr,  # 应用函数的DataArray
        input_core_dims=[['time']],  # 指定哪个维度是“核心维度”，即函数应用的维度
        vectorize=True,  # 启用向量化以自动广播和循环
        dask="parallelized",  # 如果使用Dask，请启用并行计算
        output_dtypes=[float],  # 指定输出数据类型
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
    return result_da


def get_season_frequency(dr):
    # 将日数据转换为月数据
    monthly_data = dr.resample(time='M').sum()

    # 计算每年的季节性指数
    def calculate_seasonality_index(group):
        p_total = group.sum(dim='time')
        p_monthly_mean = p_total / 12
        si = (1 / p_total) * np.sum(np.abs(group - p_monthly_mean), axis=0)
        return si

    # 按年分组并计算每年的季节性指数
    seasonality_indices = monthly_data.groupby('time.year').apply(calculate_seasonality_index)

    # 求多年的季节性指数的平均
    mean_seasonality_index = seasonality_indices.mean(dim='year')

    return mean_seasonality_index


def get_wet_frequency(dr):
    condition_down = dr.where(dr > 1, 0)
    condition_ud = condition_down.where(dr < 1, 1)

    # 计算频率
    frequency = condition_ud.sum(dim='time') / condition_ud.shape[0]

    # 返回结果
    return frequency


def get_k_frequency(dr):
    result_da = xr.apply_ufunc(
        calculate_distribution_slope,  # 应用的自定义函数
        dr,  # 应用函数的DataArray
        input_core_dims=[['time']],  # 指定哪个维度是“核心维度”，即函数应用的维度
        vectorize=True,  # 启用向量化以自动广播和循环
        dask="parallelized",  # 如果使用Dask，请启用并行计算
        output_dtypes=[float],  # 指定输出数据类型
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
    return result_da


def get_intensity_frequency(dr):
    if not isinstance(dr, xr.DataArray):
        raise TypeError("输入必须是xr.DataArray类型")

    # 检查输入是否包含time维度
    if 'time' not in dr.dims:
        raise ValueError("输入的xr.DataArray对象必须包含'time'维度")

    # 在time维度上求平均
    avg_dr = dr.mean(dim='time')

    return avg_dr


def quiet_percentile(dr, sp_frequency, sp_percentile):
    quiet_frequency = get_quiet_frequency(dr)

    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=quiet_frequency)

    quiet_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def get_quiet_frequency(dr):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    _, quiet_frequency = calculate_every_event_durations(dr.values, percentile_th=percentile_th)
    quiet_frequency = xr.DataArray(quiet_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])
    return quiet_frequency


def duration_percentile(dr, sp_frequency, sp_percentile):
    duration_frequency = get_duration_frequency(dr)

    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=duration_frequency)

    duration_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def get_duration_frequency(dr):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    duration_frequency, _ = calculate_every_event_durations(dr.values, percentile_th=percentile_th)
    duration_frequency = xr.DataArray(duration_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])
    return duration_frequency


def qk_percentile(dr, sp_frequency, sp_percentile):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    _, k_frequency = calculate_event_one_all_durations(dr.values, percentile_th=percentile_th)
    k_frequency = xr.DataArray(k_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=k_frequency)

    k_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def k_percentile(dr, sp_frequency, sp_percentile):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    k_frequency, _ = calculate_event_one_all_durations(dr.values, percentile_th=percentile_th)
    k_frequency = xr.DataArray(k_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])

    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=k_frequency)

    k_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def dfas_percentile(dr, sp_frequency, sp_percentile):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    core_frequency, _ = calculate_event_dfa(dr.values, percentile_th=percentile_th)
    core_frequency = xr.DataArray(core_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])

    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=core_frequency)

    core_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def dfa_percentile(dr, sp_frequency, sp_percentile):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    core_frequency, _ = calculate_event_dfa(dr.values, percentile_th=percentile_th)
    core_frequency = xr.DataArray(core_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])

    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=core_frequency)

    core_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def power_percentile(dr, sp_frequency, sp_percentile):
    power_frequency = get_power_frequency(dr)
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=power_frequency)

    power_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def intensity_percentile(dr, sp_frequency, sp_percentile):
    core_frequency = get_intensity_frequency(dr)
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=core_frequency)

    core_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def season_percentile(dr, sp_frequency, sp_percentile):
    core_frequency = get_season_frequency(dr)
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=core_frequency)
    core_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def wet_percentile(dr, sp_frequency, sp_percentile):
    wet_frequency = get_wet_frequency(dr)
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=wet_frequency)
    wet_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def lspf_percentile(dr, lspf, sp_frequency, sp_percentile):
    core_frequency = get_lspf_frequency(lspf)
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=core_frequency)
    core_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def lsprf_percentile(dr, lsp, sp_frequency, sp_percentile):
    lsprf_frequency = get_lsprf_frequency(dr, lsp)
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=lsprf_frequency)
    lsprf_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def cp_percentile(dr, cp, sp_frequency, sp_percentile):
    # 获取频率
    cp_frequency = get_cp_frequency(dr, cp)
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=cp_frequency)
    # 输出和保存文件
    cp_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile


def get_percentile_core(dr, raw_frequency):
    raw_data = dr.values
    frequency = raw_frequency.values
    bins = np.linspace(np.nanmin(frequency), np.nanmax(frequency), 6, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
    indices = np.digitize(frequency, bins)
    print(f'max_frequency:{np.nanmax(frequency)}')
    print(f'min_frequency:{np.nanmin(frequency)}')

    result_percentile = np.zeros((len(bins), 100))

    assert len(bins) == indices.max() == 6
    # assert indices.min() == 1

    for area_num in range(len(bins)):
        print(f'area_num: {area_num}')
        condition_area = area_num + 1 == indices
        condition_wetday = raw_data > 1
        wetday_condition_area = condition_wetday & condition_area
        result_percentile[area_num, :] = np.nanpercentile(raw_data[wetday_condition_area], np.arange(1, 101))

    return bins, indices, result_percentile


def process_percentile(key, dr, data_path):
    frequency_functions = {
        'wet': get_wet_frequency,
        'season': get_season_frequency,
        'duration': get_duration_frequency,
        'quiet': get_quiet_frequency
    }

    # Retrieve the function based on the key
    frequency_function = frequency_functions.get(key)

    if frequency_function is None:
        print(f"Warning: Invalid key '{key}'. No action taken.")
        return None  # Optionally return None or raise an exception

    # Call the function to get the frequency
    frequency = frequency_function(dr)
    frequency.name = key
    frequency.to_netcdf(f'{data_path}{key}_frequency.nc')

    # Proceed with the rest of the processing
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=frequency)
    np.savez(f'{data_path}{key}_data', percentile=result_percentile, bins=bins, indices=indices)
