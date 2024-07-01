import numpy as np
import xarray as xr
import numpy.fft as fft
from plt_temp import condition_above_percentile
from calculate_event_durations import calculate_every_event_durations, calculate_event_one_all_durations


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
    a = np.sum(inverse_X[0:14])
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


import xarray as xr


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


def percentile_sums(matrices, percentiles):
    sum_below = np.zeros((len(percentiles)))
    for ind, percentile in enumerate(percentiles):
        sum_below[ind] = np.sum(matrices[matrices < percentile])

    return sum_below


def get_percentile_dur(dr, lsp, cp, raw_frequency):
    raw_data = dr.values
    frequency = raw_frequency.values
    bins = np.logspace(np.log10(np.min(frequency)), np.log10(6), 6,
                       endpoint=False)  # 注意频率99-------------------------------------------------------------------------
    indices = np.digitize(frequency, bins)
    print(f'max_frequency:{np.max(frequency)}')
    print(f'min_frequency:{np.min(frequency)}')

    result_percentile = np.zeros((len(bins), 100))
    cp_percentile = np.zeros((len(bins), 100))
    lsp_percentile = np.zeros((len(bins), 100))
    lsp_fraction_percentile = np.zeros((len(bins), 100))
    valid_data_count = np.zeros((len(bins)))
    lsp_data = lsp.values
    cp_data = cp.values
    assert len(bins) == indices.max() == 6
    assert indices.min() == 1
    for area_num in range(len(bins)):
        print(f'area_num: {area_num}')
        condition_area = area_num + 1 == indices
        condition_wetday = raw_data > 1
        condition_wetday_cp = cp_data > 1
        condition_wetday_lsp = lsp_data > 1
        wetday_condition_area = condition_wetday & condition_area
        cp_conditioned_data = cp_data[condition_wetday_cp & condition_area]
        lsp_conditioned_data = lsp_data[condition_wetday_lsp & condition_area]
        valid_data_count[area_num] = np.sum(wetday_condition_area)
        result_percentile[area_num, :] = np.nanpercentile(raw_data[wetday_condition_area], np.arange(1, 101))
        cp_percentile[area_num, :] = np.nanpercentile(cp_conditioned_data, np.arange(1, 101))
        lsp_percentile[area_num, :] = np.nanpercentile(lsp_conditioned_data, np.arange(1, 101))
        lsp_fraction_percentile[area_num, :] = percentile_sums(lsp_conditioned_data, result_percentile[area_num, :]) / percentile_sums(raw_data[wetday_condition_area],
                                                                                                                                       result_percentile[area_num, :])

    return bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile


def get_percentile(dr, lsp, cp, raw_frequency):
    raw_data = dr.values
    frequency = raw_frequency.values
    bins = np.linspace(np.min(frequency), np.max(frequency), 6, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
    indices = np.digitize(frequency, bins)
    print(f'max_frequency:{np.max(frequency)}')
    print(f'min_frequency:{np.min(frequency)}')

    result_percentile = np.zeros((len(bins), 100))
    cp_percentile = np.zeros((len(bins), 100))
    lsp_percentile = np.zeros((len(bins), 100))
    lsp_fraction_percentile = np.zeros((len(bins), 100))
    valid_data_count = np.zeros((len(bins)))
    lsp_data = lsp.values
    cp_data = cp.values
    assert len(bins) == indices.max() == 6
    assert indices.min() == 1
    for area_num in range(len(bins)):
        print(f'area_num: {area_num}')
        condition_area = area_num + 1 == indices
        condition_wetday = raw_data > 1
        condition_wetday_cp = cp_data > 1
        condition_wetday_lsp = lsp_data > 1
        wetday_condition_area = condition_wetday & condition_area
        cp_conditioned_data = cp_data[condition_wetday_cp & condition_area]
        lsp_conditioned_data = lsp_data[condition_wetday_lsp & condition_area]
        valid_data_count[area_num] = np.sum(wetday_condition_area)
        result_percentile[area_num, :] = np.nanpercentile(raw_data[wetday_condition_area], np.arange(1, 101))
        cp_percentile[area_num, :] = np.nanpercentile(cp_conditioned_data, np.arange(1, 101))
        lsp_percentile[area_num, :] = np.nanpercentile(lsp_conditioned_data, np.arange(1, 101))
        lsp_fraction_percentile[area_num, :] = percentile_sums(lsp_conditioned_data, result_percentile[area_num, :]) / percentile_sums(raw_data[wetday_condition_area],
                                                                                                                                       result_percentile[area_num, :])

    return bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile


def lspf_percentile(dr, lspf, lsp, cp, sp_frequency, sp_percentile):
    lspf_frequency = get_lspf_frequency(lspf)
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile = get_percentile(dr=dr, cp=cp, lsp=lsp, raw_frequency=lspf_frequency)
    lspf_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, lspf_frequency, valid_data_count, lsp_fraction_percentile

    # plt_duration(calculate_event_durations(raw_dr.values, percentile_th, data_frequency_v)[0], title='Duration', vbins=bins, fig_name=fig_path + f'quiet_all.png')
    # plt_duration(calculate_event_durations(raw_dr.values, percentile_th, data_frequency_v)[1], title='Quiet', vbins=bins, fig_name=fig_path + f'quiet_all.png')
    # result_ltp = era5_narea_ptop_klag_1deg(log_points=log_points, dr=raw_dr, bins=bins, indices=indices,


def quiet_percentile(dr, lsp, cp, sp_frequency, sp_percentile):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    _, quiet_frequency = calculate_every_event_durations(dr.values, percentile_th=percentile_th)
    quiet_frequency = xr.DataArray(quiet_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile = get_percentile_dur(dr=dr, cp=cp, lsp=lsp,
                                                                                                                                    raw_frequency=quiet_frequency)
    # quiet_frequency = xr.Dataset({'tp': quiet_frequency})
    quiet_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, quiet_frequency, valid_data_count, lsp_fraction_percentile


def duration_percentile(dr, lsp, cp, sp_frequency, sp_percentile):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    duration_frequency, _ = calculate_every_event_durations(dr.values, percentile_th=percentile_th)
    duration_frequency = xr.DataArray(duration_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile = get_percentile_dur(dr=dr, cp=cp, lsp=lsp,
                                                                                                                                    raw_frequency=duration_frequency)
    # duration_frequency = xr.Dataset({'tp': duration_frequency})
    duration_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, duration_frequency, valid_data_count, lsp_fraction_percentile


def power_percentile(dr, lsp, cp, sp_frequency, sp_percentile):
    power_frequency = get_power_frequency(dr)
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile = get_percentile(dr=dr, cp=cp, lsp=lsp,
                                                                                                                                raw_frequency=power_frequency)
    power_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, power_frequency, valid_data_count, lsp_fraction_percentile


def qk_percentile(dr, lsp, cp, sp_frequency, sp_percentile):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    _, k_frequency = calculate_event_one_all_durations(dr.values, percentile_th=percentile_th)
    k_frequency = xr.DataArray(k_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile = get_percentile(dr=dr, cp=cp, lsp=lsp,
                                                                                                                                raw_frequency=k_frequency)
    # k_frequency = xr.Dataset({'tp': k_frequency})
    k_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, k_frequency, valid_data_count, lsp_fraction_percentile


def k_percentile(dr, lsp, cp, sp_frequency, sp_percentile):
    condition_top, percentile_th = condition_above_percentile(dr, percentile=30)
    k_frequency, _ = calculate_event_one_all_durations(dr.values, percentile_th=percentile_th)
    k_frequency = xr.DataArray(k_frequency, coords=[('latitude', dr.latitude.data), ('longitude', dr.longitude.data)])
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile = get_percentile(dr=dr, cp=cp, lsp=lsp,
                                                                                                                                raw_frequency=k_frequency)
    # k_frequency = xr.Dataset({'tp': k_frequency})
    k_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, k_frequency, valid_data_count, lsp_fraction_percentile


def wet_percentile(dr, lsp, cp, sp_frequency, sp_percentile):
    wet_frequency = get_wet_frequency(dr)
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile = get_percentile(dr=dr, cp=cp, lsp=lsp,
                                                                                                                                raw_frequency=wet_frequency)
    wet_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, wet_frequency, valid_data_count, lsp_fraction_percentile


def lsprf_percentile(dr, lsp, cp, sp_frequency, sp_percentile):
    lsprf_frequency = get_lsprf_frequency(dr, lsp)
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile = get_percentile(dr=dr, cp=cp, lsp=lsp,
                                                                                                                                raw_frequency=lsprf_frequency)
    lsprf_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, lsprf_frequency, valid_data_count, lsp_fraction_percentile


def cp_percentile(dr, cp, lsp, sp_frequency, sp_percentile):
    # 获取频率
    cp_frequency = get_cp_frequency(dr, cp)
    bins, indices, result_percentile, cp_percentile, lsp_percentile, valid_data_count = get_percentile(dr=dr, cp=cp, lsp=lsp, raw_frequency=cp_frequency)
    # 输出和保存文件
    cp_frequency.to_netcdf(sp_frequency)
    np.save(sp_percentile, result_percentile)
    return bins, indices, result_percentile, cp_percentile, lsp_percentile, cp_frequency, valid_data_count


# def wetday_percentile(dr,wetday_frequency, sp_frequency, sp_percentile):

def random_percentile(dr, original_dataarray):
    # 获取频率
    raw_data = dr.values
    random_frequency = original_dataarray.copy(data=np.random.uniform(size=original_dataarray.shape))
    frequency = random_frequency.values
    bins = np.linspace(0, np.max(frequency), 6, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
    indices = np.digitize(frequency, bins)
    print(f'max_frequency:{np.max(frequency)}')

    result_percentile = np.zeros((len(bins), 100))
    assert len(bins) == indices.max() == 6
    assert indices.min() == 1

    for area_num in range(len(bins)):
        print(area_num)
        condition_area = area_num + 1 == indices
        condition_wetday = raw_data > 1
        result_percentile[area_num, :] = np.nanpercentile(raw_data[condition_wetday & condition_area], np.arange(1, 101))

    # 输出和保存文件
    random_percentile = result_percentile
    return bins, indices, random_percentile, random_frequency
