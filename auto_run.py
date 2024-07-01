import importlib
import os
import numpy as np
import xarray as xr
from calculate_event_durations import calculate_event_durations
from function_draw_distribution import draw_distribution
from function_draw_memory_topper_1deg_6area import dm_area_top
from function_wdp import wdp_era5_lfp
from function_wet50 import era5_wet50, era5_hour1year
from lag_path_parameter import log_points, path_out
from plt_temp import scatter_plot, draw_hist_dq, draw_hist_dq_fit2
from plt_temp import draw_all_era5_area
from plt_temp import era5_draw_area_dataArray
from plt_temp import plt_duration
from plt_temp import condition_above_percentile


# 获取制定变量的数据
def point_path_data_hour(var, lat):
    path_all = f'C:\\ERA5\\1980-2019\\{var}_1year\\'
    data_point = xr.open_mfdataset(path_all + '*_processed_hour_1year_1deg.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


def point_path_data(var, lat):
    path_all = f'C:\\ERA5\\1980-2019\\{var}\\'
    data_point = xr.open_mfdataset(path_all + '*processed_day_1.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


def main_process(var, percentile_name, colorbar_title='Frequency (%)', module_name='function_percentile', dec=None, rd=False, renew=0, wet=False, **func_kwargs):
    # draw_all_era5_area(point_path_data('total_precipitation', lat=lat_range).where(era5_frequency_np > 0.3, np.nan))
    # random_int_matrix = np.random.randint(0, 10, data_frequency.values.squeeze().shape)
    # cov_matrix = np.ones(data_frequency.values.squeeze().shape)
    # scatter_plot(data_frequency.values.squeeze(), era5_frequency_np, var.split('_')[-1])
    # scatter_plot(data_frequency.values.squeeze(), cov_matrix, var.split('_')[-1])
    # if var.split('_')[-1] == 'wet30':
    #     data_frequency = era5_frequency
    #     bins = np.linspace(0.3, np.max(era5_frequency_np), 6, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
    #     indices = np.digitize(era5_frequency_np, bins)
    #     indices[indices == 0] = 1
    # 动态导入指定模块
    module = importlib.import_module(module_name)
    func = getattr(module, percentile_name)

    # 获得一些修饰参数
    if rd:
        var += '_random'

    figure_title_font, lat_range, sp_frequency, sp_percentile, fig_path = some_parameter(dec, var)
    era5_frequency, era5_frequency_np, era5_frequency_d3 = era5_parm()

    # 获得不同区域的降水统计信息
    func_percentile = getattr(module, percentile_name)
    if renew == 1:
        # 初始化 func_kwargs 字典，包含所有必要的参数
        func_kwargs = {
            'dr': point_path_data('total_precipitation', lat=lat_range),
            'cp': point_path_data('convective_precipitation', lat=lat_range),
            'lsp': point_path_data('large_scale_precipitation', lat=lat_range),
            'sp_frequency': sp_frequency,
            'sp_percentile': sp_percentile
        }
        (bins,
         indices,
         data_percentile,
         cp_percentile,
         lsp_percentile,
         data_frequency,
         valid_data_count,
         lsp_fraction_percentile) = func_percentile(**func_kwargs)
        if isinstance(data_frequency, xr.Dataset):
            data_frequency = data_frequency['tp']

        save_function(bins,
                      indices,
                      data_percentile,
                      cp_percentile,
                      lsp_percentile,
                      valid_data_count,
                      lsp_fraction_percentile,
                      sp=path_out + var)
    else:
        (bins,
         indices,
         data_percentile,
         cp_percentile,
         lsp_percentile,
         valid_data_count,
         lsp_fraction_percentile) = load_function(sp=path_out + var + '.npz')
        data_frequency = xr.open_dataarray(sp_frequency)

    if var.split('_')[-1] in ['duration', 'quiet']:

        data_frequency = np.log10(data_frequency)
        colorbar_title_lfp = var.split('_')[-1] + '  (day)log10'
    else:
        colorbar_title_lfp = var.split('_')[-1]

    data_frequency_lfp = data_frequency.where((era5_frequency >= 0.3), np.nan)
    wdp_era5_lfp(data_frequency=data_frequency_lfp,
                 data_percentile=data_percentile,
                 lfp=lsp_fraction_percentile,
                 sp_fp=fig_path,
                 colorbar_title=colorbar_title_lfp)
    era5_draw_area_dataArray(data_frequency - era5_frequency, name='wet-varification')
    # 主要计算过程
    area_top_per_all, selected_columns = area_top(data_percentile)

    ltp_out = f'{path_out}{var}'
    raw_dr = point_path_data('total_precipitation', lat=lat_range)
    if rd:
        raw_dr = raw_dr.sel(time=np.random.choice(raw_dr.time.values, size=len(raw_dr.time), replace=False))

    if renew == 2:
        result_ltp, duration_hist, quiet_hist = era5_wet50(era5_frequency=era5_frequency_np,
                                                           log_points=log_points,
                                                           dr=raw_dr,
                                                           bins=bins, indices=indices,
                                                           sp_out=ltp_out, sp_test=fig_path)
    else:
        result_ltp = np.load(ltp_out + 'ltp.npy')
        duration_hist = np.load(ltp_out + 'duration.npy', allow_pickle=True)
        quiet_hist = np.load(ltp_out + 'quiet.npy', allow_pickle=True)

    # 画duration或者quiet的分布图
    draw_hist_dq_fit2(duration_hist, title='Duration', vbins=bins, fig_name=fig_path + f'p_dur_fit1.png')
    draw_hist_dq_fit2(quiet_hist, title='Quiet', vbins=bins, fig_name=fig_path + f'p_quiet_fit1.png')
    draw_hist_dq(duration_hist, title='Duration', vbins=bins, fig_name=fig_path + f'p_dur.png')
    draw_hist_dq(quiet_hist, title='Quiet', vbins=bins, fig_name=fig_path + f'p_quiet.png')
    # 画示意的分布图
    draw_distribution(bins=bins, log_points=log_points, ltp=result_ltp.transpose((0, 2, 1, 3)), var=var, fig_path=fig_path, sample=5)

    # 画记忆性的图
    dm_area_top(bins=bins, log_points=log_points, area_top_per_all=area_top_per_all, selected_columns=selected_columns, dm_in=result_ltp,
                fig_path=fig_path,
                figure_title='', figure_title_font=figure_title_font, colorbar_title=colorbar_title,
                dec=dec)


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


def save_function(bins, indices, data_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile, sp):
    np.savez_compressed(sp, bins=bins, indices=indices, data_percentile=data_percentile, cp_percentile=cp_percentile,
                        lsp_percentile=lsp_percentile, valid_data_count=valid_data_count,
                        lsp_fraction_percentile=lsp_fraction_percentile)


def load_function(sp):
    with np.load(sp) as data:
        bins = data['bins']
        indices = data['indices']
        data_percentile = data['data_percentile']
        cp_percentile = data['cp_percentile']
        lsp_percentile = data['lsp_percentile']
        valid_data_count = data['valid_data_count']
        lsp_fraction_percentile = data['lsp_fraction_percentile']

    return bins, indices, data_percentile, cp_percentile, lsp_percentile, valid_data_count, lsp_fraction_percentile


if __name__ == '__main__':
    start_key = 'wet30'
    if start_key == 'wet30':
        percentile_key = 'lsprf'
    else:
        percentile_key = start_key
    # filename = os.path.splitext(os.path.basename(__file__))[0]
    # figure_title = f'day-in-40years_{var}_180day_lag'
    # colorbar_title = f'{var} {dec} (%)'
    # main_process_random_time('lsp_fraction_random_time', percentile_name='lsprf_percentile', colorbar_title='Frequency (%)')
    # main_process('lsp_fraction_cover_v1', percentile_name='lspf_percentile', lspf=point_path_data('large_scale_precipitation_fraction', lat=60))
    # main_process('lsp_fraction_v2_wd2', percentile_name='lsprf_percentile', rd=True)
    # main_process('lsp_fraction_v2_wt-w', percentile_name='lsprf_percentile', rd=True)
    # main_process('lsp_fraction_v2_wet30', percentile_name='lsprf_percentile')
    # main_process('lsp_fraction_v2_wet50', percentile_name='lsprf_percentile', rd=True)
    # main_process('wetday_vt_quiet', percentile_name='quiet_percentile', renew=False)
    # main_process('wetday_vt_duration', percentile_name='duration_percentile', renew=False)
    main_process(f'wetday_vt_{start_key}', percentile_name=f'{percentile_key}_percentile', renew=0)
    # main_process('wetday_vt_power_1year', percentile_name='power_percentile', renew=False)
    # main_process('wetday_vt_power_1year', percentile_name='power_percentile', renew=True, rd=True)
    # main_process('wetday_vt_wet-day', percentile_name='lsprf_percentile', renew=True)
    # main_process('lsp_fraction_vt_wet30', percentile_name='lsprf_percentile')
    # main_process('lsp_fraction_v2_wd1', percentile_name='lsprf_percentile', rd=True)
    # main_process('lsp_fraction', percentile_name='lsprf_percentile', rd=True)
    # main_process('lsp_fraction_order_sample', percentile_name='lsprf_percentile')
    # main_process('lsp_fraction_order', percentile_name='lsprf_percentile')
    # main_process_random_time('large_scale_precipitation_fraction_random_time', percentile_name='lspf_percentile', dec='cover time',
    #                          figure_title=, colorbar_title=
    #                          lspf = point_path_data('large_scale_precipitation_fraction', lat=60))
    # main_process_random_time('large_scale_precipitation_fraction_random_time', function_name='lsprf_percentile')
    # main_process('large_scale_precipitation_random', function_name='lsprf_percentile')
    # main_process('convective_precipitation', function_name='cp_percentile')
