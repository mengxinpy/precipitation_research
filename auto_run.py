import importlib
import os
import pandas as pd
import numpy as np
import xarray as xr
from function_wdp import wdp_era5_lfp
from function_wet50 import perform, perform_month
from lag_path_parameter import log_points, path_out, global_png_path
from plt_temp import draw_hist_dq, draw_hist_dq_fit2, draw_hist_data_collapse
from plt_temp import plt_duration
from plt_temp import draw_all_era5_area


# 获取制定变量的数据

# draw_all_era5_area(point_path_data('total_precipitation', lat=lat_range))


def main_month(var, percentile_name, colorbar_title='Frequency (%)', module_name='function_percentile_core', dec=None, rd=False, renew='000', data_set='era5', wet=False):
    # 获取参数
    module = importlib.import_module(module_name)
    func = getattr(module, percentile_name)
    era5_frequency, era5_frequency_np, fig_path, figure_title_font, func_percentile, lat_range, raw_dr, sp_frequency, sp_percentile, var = parm_set(data_set, dec, module,
                                                                                                                                                    percentile_name, rd, var)
    # draw_all_era5_area(raw_dr, sp=global_png_path)
    # 获取 frequency
    if renew[1] == '1':
        # 初始化 func_kwargs 字典，包含所有必要的参数
        func_kwargs = {
            'dr': raw_dr,
            'sp_frequency': sp_frequency,
            'sp_percentile': sp_percentile
        }
        bins, indices, data_percentile = func_percentile(**func_kwargs)
        save_function(bins, indices, data_percentile, sp=path_out + var)
    else:
        bins, indices, data_percentile = load_function(sp=path_out + var + '.npz')

    data_frequency = xr.open_dataarray(sp_frequency)

    # 修饰结果
    if var.split('_')[-1] in ['duration', 'quiet', 'intensity']:
        data_frequency = np.log10(data_frequency)
        colorbar_title_lfp = var.split('_')[-1] + '  (day)log10'
    else:
        colorbar_title_lfp = var.split('_')[-1]

    # 展示初步结果
    # data_frequency_lfp = data_frequency
    # data_frequency_lfp = xr.where(era5_frequency_np > 0.3, data_frequency, np.nan)
    data_frequency_lfp = xr.where(era5_frequency_np > 0.3, data_frequency, np.nan)
    wdp_era5_lfp(data_frequency=data_frequency_lfp,
                 data_percentile=data_percentile,
                 sp_fp=fig_path,
                 colorbar_title=colorbar_title_lfp)

    # 记忆性处理过程
    ltp_out = f'{path_out}{var}'
    if rd:
        raw_dr = raw_dr.sel(time=np.random.choice(raw_dr.time.values, size=len(raw_dr.time), replace=False))
    if renew[2] == '1':
        result_ltp, duration_hist, quiet_hist = perform_month(era5_frequency=era5_frequency_np,
                                                              log_points=log_points,
                                                              dr=raw_dr,
                                                              bins=bins, indices=indices,
                                                              sp_out=ltp_out, sp_test=fig_path)
    else:
        result_ltp = np.load(ltp_out + 'ltp.npy')
        duration_hist = np.load(ltp_out + 'duration.npy', allow_pickle=True)
        quiet_hist = np.load(ltp_out + 'quiet.npy', allow_pickle=True)

    # 使用 zip 函数同时遍历两个数据集
    bins_dp = bins
    for ind, (p_duration, p_quiet) in enumerate(zip(duration_hist, quiet_hist)):
        for sub_ind, (duration, quiet) in enumerate(zip(p_duration, p_quiet)):
            plt_duration(duration, title='Duration', vbins=bins_dp, fig_name=fig_path + f'duration_{ind}%_{sub_ind}per.png')
            plt_duration(quiet, title='Quiet', vbins=bins_dp, fig_name=fig_path + f'quiet_{ind}%_{sub_ind}per.png')


def main_process(var, percentile_name, colorbar_title='Frequency (%)', module_name='function_percentile_core', dec=None, rd=False, renew='000', data_set='era5', wet=False):
    # 获取参数
    module = importlib.import_module(module_name)
    func = getattr(module, percentile_name)
    era5_frequency, era5_frequency_np, fig_path, figure_title_font, func_percentile, lat_range, raw_dr, sp_frequency, sp_percentile, var = parm_set(data_set, dec, module,
                                                                                                                                                    percentile_name, rd, var)
    # draw_all_era5_area(raw_dr, sp=global_png_path)
    # 获取 frequency
    if renew[1] == '1':
        # 初始化 func_kwargs 字典，包含所有必要的参数
        func_kwargs = {
            'dr': raw_dr,
            'sp_frequency': sp_frequency,
            'sp_percentile': sp_percentile
        }
        bins, indices, data_percentile = func_percentile(**func_kwargs)
        save_function(bins, indices, data_percentile, sp=path_out + var)
    else:
        bins, indices, data_percentile = load_function(sp=path_out + var + '.npz')

    data_frequency = xr.open_dataarray(sp_frequency)

    # 修饰结果
    if var.split('_')[-1] in ['duration', 'quiet', 'intensity']:
        data_frequency = np.log10(data_frequency)
        colorbar_title_lfp = var.split('_')[-1] + '  (day)log10'
    else:
        colorbar_title_lfp = var.split('_')[-1]

    # 展示初步结果 
    # data_frequency_lfp = data_frequency
    # data_frequency_lfp = xr.where(era5_frequency_np > 0.3, data_frequency, np.nan)
    data_frequency_lfp = xr.where(era5_frequency_np > 0.3, data_frequency, np.nan)
    wdp_era5_lfp(data_frequency=data_frequency_lfp,
                 data_percentile=data_percentile,
                 sp_fp=fig_path,
                 colorbar_title=colorbar_title_lfp)

    # 记忆性处理过程
    ltp_out = f'{path_out}{var}'
    if rd:
        raw_dr = raw_dr.sel(time=np.random.choice(raw_dr.time.values, size=len(raw_dr.time), replace=False))
    if renew[2] == '1':
        result_ltp, duration_hist, quiet_hist = perform(era5_frequency=era5_frequency_np,
                                                        log_points=log_points,
                                                        dr=raw_dr,
                                                        bins=bins, indices=indices,
                                                        sp_out=ltp_out, sp_test=fig_path)
    else:
        result_ltp = np.load(ltp_out + 'ltp.npy')
        duration_hist = np.load(ltp_out + 'duration.npy', allow_pickle=True)
        quiet_hist = np.load(ltp_out + 'quiet.npy', allow_pickle=True)

    # 画duration或者quiet的分布图
    # draw_hist_dq_fit2(duration_hist, title='Duration', vbins=bins, fig_name=fig_path + f'p_dur_fit1.png')
    # draw_hist_dq_fit2(quiet_hist, title='Quiet', vbins=bins, fig_name=fig_path + f'p_quiet_fit1.png')
    data_frequency_dq = xr.where(era5_frequency_np > 0.3, era5_frequency, np.nan)
    bins_dp = np.linspace(np.min(data_frequency_dq), np.max(data_frequency_dq), 6, endpoint=False)
    draw_hist_dq(duration_hist, title='Duration', vbins=bins_dp, fig_name=fig_path + f'p_dur.png')
    draw_hist_dq(quiet_hist, title='Quiet', vbins=bins_dp, fig_name=fig_path + f'p_quiet.png')
    draw_hist_data_collapse(duration_hist, title='Duration', vbins=bins_dp, fig_name=fig_path + f'p_dur_data_collapse.png')
    draw_hist_data_collapse(quiet_hist, title='Quiet', vbins=bins_dp, fig_name=fig_path + f'p_quiet_data_collapse.png')


# 备用语句
# data_frequency_lfp = data_frequency.where((era5_frequency >= 0.3), np.nan)
def point_path_data_hour(var, lat):
    path_all = f'C:\\ERA5\\1980-2019\\{var}_1year\\'
    data_point = xr.open_mfdataset(path_all + '*_processed_hour_1year_1deg.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


def point_path_data(var, lat):
    path_all = f'C:\\ERA5\\1980-2019\\{var}\\'
    data_point = xr.open_mfdataset(path_all + '*processed_day_1.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


def remove_nan_time_slices(data_point, nan_threshold=0.5):
    nan_count = data_point.isnull().sum(dim=['latitude', 'longitude'])

    # 计算每个时间切片中元素的总数
    total_count = np.prod(data_point.shape[1:])

    # 计算每个时间切片中的NaN比例
    nan_ratio = nan_count / total_count

    # 找出NaN比例低于阈值的时间切片
    valid_time_indices = nan_ratio < nan_threshold

    # 过滤掉包含大量NaN值的时间切片
    filtered_data_point = data_point.sel(time=valid_time_indices)

    return filtered_data_point


def point_path_data_amsr2(lat):
    path_all = f'F:\\liusch\\amsr2\\processed_amsr\\'
    # RSS_AMSR2_ocean_L3_daily_2013-01-01_v08.2_processed
    # data_point = xr.open_mfdataset(path_all + '*processed_amsr2.nc', combine='nested', concat_dim='time')
    data_point = xr.open_mfdataset(path_all + '*processed_amsr2.nc', combine='nested', concat_dim='time')['tp']
    time_values = pd.date_range(start='2013-01-01', periods=data_point.sizes['time'], freq='D')
    data_point = data_point.assign_coords(time=time_values)
    # 检查时间维度中是否有 NaN 值
    data_point = remove_nan_time_slices(data_point, nan_threshold=0.7)
    nan_area_indices = data_point.isnull().any(dim=['time'])

    # 使用 where 方法将整个时间维度的数据设为 NaN
    # data_point = data_point.where(~nan_area_indices, other=np.nan)
    return data_point


def parm_set(data_set, dec, module, percentile_name, rd, var):
    # 基本的参数信息
    if rd:
        var += '_random'
    if data_set != 'era5':
        var += f'_{data_set}'
    figure_title_font, lat_range, sp_frequency, sp_percentile, fig_path = some_parameter(dec, var)
    era5_frequency, era5_frequency_np, era5_frequency_d3 = era5_parm()
    if data_set == 'amsr2':
        raw_dr = point_path_data_amsr2(lat_range)
    else:
        raw_dr = point_path_data('total_precipitation', lat=lat_range)
    func_percentile = getattr(module, percentile_name)
    return era5_frequency, era5_frequency_np, fig_path, figure_title_font, func_percentile, lat_range, raw_dr, sp_frequency, sp_percentile, var


def era5_parm():
    era5_frequency = xr.open_dataset('era5_frequency_processed.nc').to_array().squeeze()
    era5_frequency_np = era5_frequency.values
    era5_frequency_d3 = np.where(era5_frequency_np < 0.3, np.nan, era5_frequency_np)
    return era5_frequency, era5_frequency_np, era5_frequency_d3


def area_top(data_percentile):
    selected_columns = [59, 69, 79, 89]
    area_top_per_all = data_percentile[:, selected_columns].T
    return area_top_per_all, selected_columns


# 定义一些参数
def some_parameter(dec, var):
    figure_title_font = 24
    lat_range = 60
    sp_frequency = f'{path_out}{var}_frequency_lat{lat_range}.nc'
    sp_percentile = f'{path_out}{var}_percentile_lat{lat_range}.npy'
    fig_path = f'.\\temp_fig\\{var}\\'
    os.makedirs(fig_path, exist_ok=True)
    return figure_title_font, lat_range, sp_frequency, sp_percentile, fig_path


def save_function(bins, indices, data_percentile, sp):
    np.savez_compressed(sp, bins=bins, indices=indices, data_percentile=data_percentile)


def load_function(sp):
    with np.load(sp) as data:
        bins = data['bins']
        indices = data['indices']
        data_percentile = data['data_percentile']

    return bins, indices, data_percentile


if __name__ == '__main__':
    start_key = 'wet'
    if start_key == 'wet30':
        percentile_key = 'lsprf'
    else:
        percentile_key = start_key
    data_set = 'era5'

    main_month(f'wetday_month_vt_{start_key}', percentile_name=f'{percentile_key}_percentile', renew='001', data_set=data_set)
    # main_process(f'wetday_vt_{start_key}', percentile_name=f'{percentile_key}_percentile', renew='001', data_set=data_set)
