import os
from config import onat_list

import numpy as np
import pandas as pd
import xarray as xr
from IPython.core.display_functions import display
from IPython.display import FileLinks
from utils import point_path_data
from utils import nan_digitize
from config import log_points, path_out
from plot_lib import draw_hist_dq, draw_hist_data_collapse, plt_duration_hist
from plot_lib import draw_hist_dq_dataarray
from map_lib import wdp_era5_lfp, combined_plot, plot_map, plot_distribution
from processor_main import perform
from processor_map import process_percentile
from config import intern_data_path



def main_process(var, data_set='era5', renew='000', return_point='0', rd=False, hide_low_30=False):
    # 获取参数
    era5_frequency, era5_frequency_np, fig_path, internal_data_path, raw_dr, var = parm_set(data_set, rd, var)

    # %% percentile part
    percentile_key = var.split('_')[0]
    if renew[1] == '1':
        # 初始化 func_kwargs 字典，包含所有必要的参数
        func_kwargs = {
            'key': percentile_key,
            'dr': raw_dr,
            'data_path': internal_data_path,
        }
        process_percentile(**func_kwargs)
    data_percentile, bins, indices = load_data(f'{internal_data_path}{percentile_key}_data.npz')
    data_frequency = xr.open_dataarray(f'{internal_data_path}{percentile_key}_frequency.nc')
    if return_point == 'percentile':
        return
    # %%  show percentile part result
    # 为初步检查的画图修饰数据
    if percentile_key in ['duration', 'quiet', 'intensity']:
        data_frequency = np.log10(data_frequency)
        colorbar_title_lfp = percentile_key + '  (day)log10'
    else:
        colorbar_title_lfp = percentile_key

    # 展示初步结果 
    # data_frequency_lfp = data_frequency
    # data_frequency_lfp = xr.where(era5_frequency_np > 0.3, data_frequency, np.nan)
    if hide_low_30:
        data_frequency_lfp = xr.where(era5_frequency_np > 0.3, data_frequency, np.nan)
    else:
        data_frequency_lfp = data_frequency
    wdp_era5_lfp(data_frequency=data_frequency_lfp,
                 data_percentile=data_percentile,
                 sp_fp=fig_path,
                 colorbar_title=colorbar_title_lfp)
    # display(FileLinks(fig_path))
    if return_point == 'percentile_show':
        return
    # %% process_divide memory and duration

    if rd:
        raw_dr = raw_dr.sel(time=np.random.choice(raw_dr.time.values, size=len(raw_dr.time), replace=False))
    if renew[2] == '1':
        perform(dr=raw_dr, bins=bins, indices=indices, ltp_out=internal_data_path, ltp_fig=fig_path)
    duration_hist, quiet_hist = load_ltp(internal_data_path)

    if return_point == 'perform':
        return
    # %% show the result of process_divide
    # 画duration或者quiet的分布图
    # data_frequency_dq = xr.where(era5_frequency_np, era5_frequency, np.nan)
    # bins_dp = np.linspace(np.min(data_frequency_dq), np.max(data_frequency_dq), 6, endpoint=False)
    # draw_hist_dq(duration_hist, title='Duration', vbins=bins, fig_name=fig_path + f'p_dur.png')
    # print(bins)
    # draw_hist_dq(quiet_hist, title='Quiet', vbins=bins, fig_name=fig_path + f'p_quiet.png')
    # draw_hist_data_collapse(duration_hist, title='Duration', vbins=bins, fig_name=fig_path + f'p_dur_data_collapse.png')
    # draw_hist_data_collapse(quiet_hist, title='Quiet', vbins=bins, fig_name=fig_path + f'p_quiet_data_collapse.png')


def load_ltp(ltp_out):
    duration_hist = np.load(ltp_out + 'ltp_duration_newest.npy', allow_pickle=True)
    quiet_hist = np.load(ltp_out + 'ltp_quiet_newest.npy', allow_pickle=True)
    return duration_hist, quiet_hist


# 备用语句
# data_frequency_lfp = data_frequency.where((era5_frequency >= 0.3), np.nan)
def point_path_data_hour(var, lat):
    path_all = f'C:/ERA5/1980-2019/{var}_1year/'
    data_point = xr.open_mfdataset(path_all + '*_processed_hour_1year_1deg.nc')[''.join(word[0] for word in var.split('_') if word)].sel(latitude=slice(lat, -lat))
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


def point_path_data_amsr2(variable, lat=60):
    path_all = f'F:/liusch/amsr2/processed_amsr/'
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


def parm_set(data_set, rd, var):
    # 基本的参数信息
    if rd:
        var += '_random'
    var += f'_{data_set}'
    internal_data_path, fig_path = path_parameter(var)
    era5_frequency, era5_frequency_np, era5_frequency_d3 = era5_parm()
    if data_set == 'amsr2':
        raw_dr = point_path_data_amsr2()
    else:
        raw_dr = point_path_data('total_precipitation',dataset=data_set)
    return era5_frequency, era5_frequency_np, fig_path, internal_data_path, raw_dr, var


def load_data(file_path):
    loaded_data = np.load(file_path, allow_pickle=True)

    result_percentile = loaded_data['percentile']
    bins = loaded_data['bins']
    indices = loaded_data['indices']

    return result_percentile, bins, indices


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
def path_parameter(var):
    internal_data_path = f'./internal_data/{var}/'
    fig_path = f'./fig/{var}/'
    create_directory_for_file(fig_path)
    create_directory_for_file(internal_data_path)
    return internal_data_path, fig_path


def create_directory_for_file(file_path):
    # 提取文件的目录部分
    directory = os.path.dirname(file_path)

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)


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
    data_set_list = ['mswep', 'persiann','era5']

    main_process(f'{start_key}', renew='001', data_set='era5')
