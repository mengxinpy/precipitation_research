import xarray as xr
from function_percentile import lspf_percentile
from function_era5_narea_ptop_klag_1deg import era5_narea_ptop_klag_1deg
from function_draw_memory_topper_1deg_6area import dm_area_top
# from function_wdp import wdp_era5
from lag_path_parameter import log_points, path_all, path_out, path_png
import numpy as np
import os
import importlib


def point_path_data(var, lat):
    path_all = f'C:\\ERA5\\1980-2019\\{var}\\'
    data_point = xr.open_mfdataset(path_all + '*processed_day_1.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
    return data_point


def main_process(var, function_name, module_name='function_percentile', dec=None, **func_kwargs):
    # 动态导入指定模块
    module = importlib.import_module(module_name)
    # 从模块中获取指定函数
    func = getattr(module, function_name)

    figure_title = f'day-in-40years_{var}_180day_lag'
    colorbar_title = f'{var} {dec} (%)'
    figure_title_font = 24
    lat_range = 60
    path_var = path_all + var
    sp_frequency = f'{path_out}{var}_frequency_lat{lat_range}.nc'
    sp_percentile = f'{path_out}{var}_percentile_lat{lat_range}.npy'
    # 获得不同区域的降水统计信息
    func_percentile = getattr(module, function_name)
    bins, indices, data_percentile, cp_percentile, lsp_percentile, data_frequency, valid_data_count = \
        func_percentile(dr=point_path_data('total_precipitation', lat=lat_range), cp=point_path_data('convective_precipitation', lat=lat_range),
                        lsp=point_path_data('large_scale_precipitation', lat=lat_range), sp_frequency=sp_frequency, sp_percentile=sp_percentile, **func_kwargs)  # 输出参数

    # wdp_era5(data_frequency=data_frequency, data_percentile=data_percentile, cp_percentile=cp_percentile, lsp_percentile=lsp_percentile, sp_fp=var, colorbar_title=colorbar_title)


if __name__ == '__main__':
    # filename = os.path.splitext(os.path.basename(__file__))[0]
    # main_process('large_scale_precipitation', function_name='lspf_percentile', dec='cover time', lspf=point_path_data('large_scale_precipitation_fraction', lat=60))
    main_process('convective_precipitation', function_name='cp_percentile')
