import importlib
import os

import numpy as np
import xarray as xr

from function_draw_distribution import draw_distribution
from function_draw_memory_topper_1deg_6area import dm_area_top
from function_wdp import wdp_era5_lfp
from function_wet50 import era5_wet50
from lag_path_parameter import log_points, path_out


# 获取制定变量的数据
def point_path_data(var, lat):
    path_all = f'C:\\ERA5\\1980-2019\\{var}\\'
    data_point = xr.open_mfdataset(path_all + '*processed_day_1.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


def main_process(var, percentile_name, colorbar_title='Frequency (%)', module_name='function_percentile', dec=None, rd=False, renew=True, **func_kwargs):
    # 动态导入指定模块
    module = importlib.import_module(module_name)
    func = getattr(module, percentile_name)

    # 获得一些修饰参数
    if rd:
        var += '_random'
    figure_title_font, lat_range, sp_frequency, sp_percentile, fig_path = some_parameter(dec, var)
    # era5_frequency = xr.open_dataset('era5_frequency.nc').sel(latitude=slice(lat_range, -lat_range)).coarsen(longitude=4, latitude=4, boundary='trim').reduce(
    #     lambda x, **kwargs: x[:, 0, :, 0])
    # era5_frequency.to_netcdf('era5_frequency_processed.nc')
    era5_frequency = xr.open_dataset('era5_frequency_processed.nc').to_array()
    era5_frequency_np = era5_frequency.values.squeeze()
    # 获得不同区域的降水统计信息
    func_percentile = getattr(module, percentile_name)
    if renew:
        bins, indices, data_percentile, cp_percentile, lsp_percentile, data_frequency, valid_data_count, lsp_fraction_percentile = \
            func_percentile(dr=point_path_data('total_precipitation', lat=lat_range), cp=point_path_data('convective_precipitation', lat=lat_range),
                            lsp=point_path_data('large_scale_precipitation', lat=lat_range), sp_frequency=sp_frequency, sp_percentile=sp_percentile, **func_kwargs)  # 输出参数
        save_function(bins, indices, data_percentile, cp_percentile, lsp_percentile, data_frequency, valid_data_count, lsp_fraction_percentile, sp=path_out + 'variables')
    else:
        bins, indices, data_percentile, cp_percentile, lsp_percentile, data_frequency, valid_data_count, lsp_fraction_percentile = load_function(sp=path_out + 'variables.npz')
    # 画分布图和频率图
    wdp_era5_lfp(data_frequency=era5_frequency.where((era5_frequency >= 0.3), np.nan), data_percentile=data_percentile, lfp=lsp_fraction_percentile,
                 sp_fp=fig_path,
                 colorbar_title='Frequency (%)')

    # 主要计算过程
    area_top_per_all, selected_columns = area_top(data_percentile)  # todo:加阈值

    ltp_out = f'{path_out}ltp_all_{var}_lat60.npy'
    # result_ltp = np.load(ltp_out)
    #
    raw_dr = point_path_data('total_precipitation', lat=lat_range)
    if rd:
        raw_dr = raw_dr.sel(time=np.random.choice(raw_dr.time.values, size=len(raw_dr.time), replace=False))
    # result_ltp = era5_narea_ptop_klag_1deg(log_points=log_points, dr=raw_dr, bins=bins, indices=indices,
    #                                        sp_out=ltp_out)
    #

    bins = np.linspace(0.3, np.max(era5_frequency_np), 6, endpoint=False)  # 注意频率99-------------------------------------------------------------------------
    indices = np.digitize(era5_frequency_np, bins)
    indices[indices == 0] = 1
    result_ltp = era5_wet50(era5_frequency=era5_frequency_np, log_points=log_points, dr=raw_dr, bins=bins, indices=indices,
                            sp_out=ltp_out, sp_test=fig_path)
    # result_ltp = era5_wet50(era5_frequency=era5_frequency_np, log_points=log_points, dr=raw_dr, bins=bins, indices=indices,
    #                         sp_out=ltp_out, sp_test=fig_path)

    # 画示意的分布图
    # draw_distribution_test(bins=bins, log_points=log_points, ltp=result_ltp[1].transpose(1, 0, 2), var=var, toparea_percentile=toparea_percentile)
    draw_distribution(bins=bins, log_points=log_points, ltp=result_ltp.transpose((0, 2, 1, 3)), var=var, fig_path=fig_path, sample=5)

    # 画记忆性的图
    # dm_lagtime = f'{fig_path}'
    # dm_frequency = f'{fig_path}'
    dm_area_top(bins=bins, log_points=log_points, area_top_per_all=area_top_per_all, selected_columns=selected_columns, dm_in=result_ltp,
                fig_path=fig_path,
                figure_title='', figure_title_font=figure_title_font, colorbar_title=colorbar_title,
                dec=dec)


def area_top(data_percentile):
    selected_columns = [69, 79]
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


def save_function(bins, indices, data_percentile, cp_percentile, lsp_percentile, data_frequency, valid_data_count, lsp_fraction_percentile, sp):
    np.savez_compressed(sp, bins=bins, indices=indices, data_percentile=data_percentile, cp_percentile=cp_percentile,
                        lsp_percentile=lsp_percentile, data_frequency=data_frequency, valid_data_count=valid_data_count,
                        lsp_fraction_percentile=lsp_fraction_percentile)


def load_function(sp):
    with np.load(sp) as data:
        bins = data['bins']
        indices = data['indices']
        data_percentile = data['data_percentile']
        cp_percentile = data['cp_percentile']
        lsp_percentile = data['lsp_percentile']
        data_frequency = data['data_frequency']
        valid_data_count = data['valid_data_count']
        lsp_fraction_percentile = data['lsp_fraction_percentile']

    return bins, indices, data_percentile, cp_percentile, lsp_percentile, data_frequency, valid_data_count, lsp_fraction_percentile


if __name__ == '__main__':
    # filename = os.path.splitext(os.path.basename(__file__))[0]
    # figure_title = f'day-in-40years_{var}_180day_lag'
    # colorbar_title = f'{var} {dec} (%)'
    # main_process_random_time('lsp_fraction_random_time', percentile_name='lsprf_percentile', colorbar_title='Frequency (%)')
    # main_process('lsp_fraction_cover_v1', percentile_name='lspf_percentile', lspf=point_path_data('large_scale_precipitation_fraction', lat=60))
    # main_process('lsp_fraction_v2_wd2', percentile_name='lsprf_percentile', rd=True)
    # main_process('lsp_fraction_v2_wt-w', percentile_name='lsprf_percentile', rd=True)
    # main_process('lsp_fraction_v2_wet30', percentile_name='lsprf_percentile')
    # main_process('lsp_fraction_v2_wet50', percentile_name='lsprf_percentile', rd=True)
    main_process('wetday_vt_wet30', percentile_name='lsprf_percentile', renew=False)
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
