import numpy as np
import xarray as xr
import numpy.fft as fft
from numpy.f2py.auxfuncs import throw_error

from utils import condition_above_percentile
from utils_event import calculate_every_event_durations


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


def just_spectrum_ratio(time_series):
    # 对时间序列进行傅里叶变换
    fft_result = np.fft.fft(time_series)

    # 计算频率
    frequencies = np.fft.fftfreq(len(time_series))

    # 计算幅值
    magnitude = np.abs(fft_result)

    # 归一化幅值
    normalized_magnitude = magnitude / np.sum(magnitude)

    # 找到最接近365.25天的频率索引
    target_frequency = 1 / 365.25
    closest_index = np.argmin(np.abs(frequencies - target_frequency))

    # 返回这些频率的归一化幅值占比总和
    return np.sum(normalized_magnitude[closest_index])
    # 返回这些频率的归一化幅值占比总和


def get_power_frequency(dr):
    result_da = xr.apply_ufunc(
        just_spectrum_ratio,  # 应用的自定义函数
        dr,  # 应用函数的DataArray
        input_core_dims=[['time']],  # 指定哪个维度是"核心维度"，即函数应用的维度
        vectorize=True,  # 启用向量化以自动广播和循环
        dask="parallelized",  # 如果使用Dask，请启用并行计算
        output_dtypes=[float],  # 指定输出数据类型
        dask_gufunc_kwargs={'allow_rechunk': True}
    )
    return result_da


def get_wet_frequency(dr):
    condition_down = dr.where(dr > 1, 0)
    condition_ud = condition_down.where(dr < 1, 1)

    # 计算频率
    frequency = condition_ud.sum(dim='time') / condition_ud.shape[0]

    # 返回结果
    return frequency


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


def get_duration_frequency(dr, percentile=30):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=percentile)
    duration_frequency, _ = calculate_every_event_durations(dr.values, percentile_th=percentile_th)
    duration_frequency = xr.DataArray(duration_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])
    file_name = f'./internal_data/top{percentile:.0f}%_frequency.nc'
    duration_frequency.name = f'top{percentile}'
    duration_frequency.to_netcdf(file_name)
    return duration_frequency


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
    bins_num = 6
    bins = np.linspace(np.nanmin(frequency), np.nanmax(frequency), bins_num, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
    indices = np.digitize(frequency, bins)
    print(f'max_frequency:{np.nanmax(frequency)}')
    print(f'min_frequency:{np.nanmin(frequency)}')

    result_percentile = np.zeros((len(bins), 100))

    # assert indices.min() == 1

    for area_num in range(len(bins)):
        print(f'area_num: {area_num}')
        condition_area = area_num + 1 == indices
        condition_wetday = raw_data > 1
        wetday_condition_area = condition_wetday & condition_area
        result_percentile[area_num, :] = np.nanpercentile(raw_data[wetday_condition_area], np.arange(1, 101))

    return bins, indices, result_percentile


def process_percentile(key, dr, data_path):
    print(f"传入的key: {key}")
    print(f"key的类型: {type(key)}")
    
    frequency_functions = {
        'wet': get_wet_frequency,
        'duration': get_duration_frequency,
        'quiet': get_quiet_frequency,
        'power': get_power_frequency,
        'intensity': get_intensity_frequency,
    }
    
    print(f"可用的函数: {list(frequency_functions.keys())}")
    
    # Retrieve the function based on the key
    frequency_function = frequency_functions.get(key)
    print(f"获取到的函数: {frequency_function}")
    
    if frequency_function is None:
        print(f"Warning: Invalid key '{key}'. No action taken.")
        return None  # Optionally return None or raise an exception
    
    # Call the function to get the frequency
    print("开始调用频率计算函数...")
    frequency = frequency_function(dr)
    print("频率计算完成")
    
    frequency.name = key
    frequency.to_netcdf(f'{data_path}{key}_frequency.nc')
    
    # Proceed with the rest of the processing
    bins, indices, result_percentile = get_percentile_core(dr=dr, raw_frequency=frequency)
    np.savez(f'{data_path}{key}_data', percentile=result_percentile, bins=bins, indices=indices)
