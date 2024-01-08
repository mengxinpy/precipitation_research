import xarray as xr
from function_percentile import lspf_percentile
from function_era5_narea_ptop_klag_1deg import era5_narea_ptop_klag_1deg
from function_draw_memory_topper_1deg_6area import dm_area_top
from function_wdp import wdp_era5, wdp_era5_lfp
from lag_path_parameter import log_points, path_all, path_out, path_png
import numpy as np
import os
import importlib


# 获取制定变量的数据
def point_path_data(var, lat):
    path_all = f'E:\\ERA5\\1980-2019\\{var}\\'
    data_point = xr.open_mfdataset(path_all + '*processed_day_1.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


def main_process(var, function_name, module_name='function_percentile', dec=None, **func_kwargs):
    # 动态导入指定模块
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)

    # 获得一些修饰参数
    colorbar_title, figure_title, figure_title_font, lat_range, sp_frequency, sp_percentile = some_parameter(dec, var)

    # 获得不同区域的降水统计信息
    func_percentile = getattr(module, function_name)
    bins, indices, data_percentile, cp_percentile, lsp_percentile, data_frequency, valid_data_count, lsp_fraction_percentile = \
        func_percentile(dr=point_path_data('total_precipitation', lat=lat_range), cp=point_path_data('convective_precipitation', lat=lat_range),
                        lsp=point_path_data('large_scale_precipitation', lat=lat_range), sp_frequency=sp_frequency, sp_percentile=sp_percentile, **func_kwargs)  # 输出参数
    # 画分布图和频率图
    wdp_era5(data_frequency=data_frequency, data_percentile=data_percentile, cp_percentile=cp_percentile, lsp_percentile=lsp_percentile, sp_fp=var, colorbar_title=colorbar_title)

    # 主要计算过程
    selected_columns = [49, 59, 69, 79]
    area_top_per_all = data_percentile[:, selected_columns].T
    ltp_out = f'{path_out}ltp_all_{var}_lat60.npy'
    raw_dr = point_path_data('total_precipitation', lat=lat_range)
    # 打乱时间序列
    shuffled_da = raw_dr.sel(time=np.random.choice(raw_dr.time.values, size=len(raw_dr.time), replace=False))
    # 按时间重新排列
    sorted_da = shuffled_da.sortby('time')
    result_ltp = era5_narea_ptop_klag_1deg(log_points=log_points, dr=sorted_da, bins=bins, indices=indices,
                                           area_top_per_all=area_top_per_all,
                                           sp_out=ltp_out)
    # result_ltp = era5_narea_ptop_klag_1deg(log_points=log_points, dr=point_path_data('total_precipitation', lat=lat_range), bins=bins, indices=indices,
    #                                        area_top_per_all=area_top_per_all,
    #                                        sp_out=ltp_out)

    # 画记忆性的图
    dm_lagtime = f'{path_png}tmp_{var}_{lat_range}.png'
    dm_frequency = f'{path_png}amt_{var}_{lat_range}.png'
    dm_area_top(bins=bins, log_points=log_points, area_top_per_all=area_top_per_all, selected_columns=selected_columns, dm_in=result_ltp,
                sp_dm_frequency=dm_frequency, sp_dm_lagtime=dm_lagtime,
                figure_title=figure_title, figure_title_font=figure_title_font, colorbar_title=colorbar_title,
                dec=dec)


def main_process_random_time(var, function_name, module_name='function_percentile', dec=None, **func_kwargs):
    # 动态导入指定模块
    module = importlib.import_module(module_name)
    # 从模块中获取指定函数
    func = getattr(module, function_name)

    # 获得一些修饰参数
    colorbar_title, figure_title, figure_title_font, lat_range, sp_frequency, sp_percentile = some_parameter(dec, var)

    # 获得不同区域的降水统计信息
    func_percentile = getattr(module, function_name)
    bins, indices, data_percentile, cp_percentile, lsp_percentile, data_frequency, valid_data_count, lsp_fraction_percentile = \
        func_percentile(dr=point_path_data('total_precipitation', lat=lat_range), cp=point_path_data('convective_precipitation', lat=lat_range),
                        lsp=point_path_data('large_scale_precipitation', lat=lat_range), sp_frequency=sp_frequency, sp_percentile=sp_percentile, **func_kwargs)  # 输出参数
    # 画分布图和频率图
    wdp_era5_lfp(data_frequency=data_frequency, data_percentile=data_percentile, lfp=lsp_fraction_percentile, sp_fp=var,
                 colorbar_title=colorbar_title)

    # 主要计算过程
    area_top_per_all, selected_columns = area_top(data_percentile)

    raw_dr = point_path_data('total_precipitation', lat=lat_range)
    shuffled_da = raw_dr.sel(time=np.random.choice(raw_dr.time.values, size=len(raw_dr.time), replace=False))

    ltp_out = f'{path_out}ltp_all_{var}_lat60.npy'
    result_ltp = era5_narea_ptop_klag_1deg(log_points=log_points, dr=shuffled_da, bins=bins, indices=indices,
                                           area_top_per_all=area_top_per_all,
                                           sp_out=ltp_out)

    # 画记忆性的图
    dm_lagtime = f'{path_png}tmp_{var}_{lat_range}.png'
    dm_frequency = f'{path_png}amt_{var}_{lat_range}.png'
    dm_area_top(bins=bins, log_points=log_points, area_top_per_all=area_top_per_all, selected_columns=selected_columns, dm_in=result_ltp,
                sp_dm_frequency=dm_frequency, sp_dm_lagtime=dm_lagtime,
                figure_title=figure_title, figure_title_font=figure_title_font, colorbar_title=colorbar_title,
                dec=dec)
    # 画记忆性分布的图



def area_top(data_percentile):
    selected_columns = [49, 59, 69, 79]
    area_top_per_all = data_percentile[:, selected_columns].T
    return area_top_per_all, selected_columns


# 定义一些参数
def some_parameter(dec, var):
    figure_title = f'day-in-40years_{var}_180day_lag'
    colorbar_title = f'{var} {dec} (%)'
    figure_title_font = 24
    lat_range = 60
    path_var = path_all + var
    sp_frequency = f'{path_out}{var}_frequency_lat{lat_range}.nc'
    sp_percentile = f'{path_out}{var}_percentile_lat{lat_range}.npy'
    return colorbar_title, figure_title, figure_title_font, lat_range, sp_frequency, sp_percentile


if __name__ == '__main__':
    # filename = os.path.splitext(os.path.basename(__file__))[0]
    main_process_random_time('large_scale_precipitation_fraction_random_time', function_name='lspf_percentile', dec='cover time',
                             lspf=point_path_data('large_scale_precipitation_fraction', lat=60))
    # main_process_random_time('large_scale_precipitation_fraction_random_time', function_name='lsprf_percentile')
    # main_process('large_scale_precipitation_random', function_name='lsprf_percentile')
    # main_process('convective_precipitation', function_name='cp_percentile')
