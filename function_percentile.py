import numpy as np


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


def percentile_sums(matrices, percentiles):
    sum_below = np.zeros((len(percentiles)))
    for ind, percentile in enumerate(percentiles):
        sum_below[ind] = np.sum(matrices[matrices < percentile])

    return sum_below


def get_percentile(dr, lsp, cp, raw_frequency):
    raw_data = dr.values
    frequency = raw_frequency.values
    bins = np.linspace(0, np.max(frequency), 6, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
    indices = np.digitize(frequency, bins)
    print(f'max_frequency:{np.max(frequency)}')

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
        print(area_num)
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
