import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import xarray as xr
from scipy.stats import percentileofscore

from utils_event import calculate_event_durations, calculate_latitude_weights, calculate_weighted_event_durations
from utils import get_hist
from config import onat_list, onat_list_one
from config import intern_data_path
from plot_lib import pt_6


def perform(dr, bins, indices, ltp_out, ltp_fig, top_bins=(40, 30, 20, 10,8,6,4,2), era5_frequency=1):
    condition_frequency, condition_wetday, duration_hist, quiet_hist, raw_data = ini_data(dr, bins, top_bins, era5_frequency)
    
    # 获取纬度信息并计算权重
    latitudes = dr.latitude.values
    lat_weights = calculate_latitude_weights(latitudes)
    # 扩展权重维度以匹配数据维度
    lat_weights = lat_weights[:, np.newaxis]  # 添加经度维度

    for p, per in enumerate(top_bins):
        condition_top, percentile_th = condition_above_percentile(dr, percentile=per)

        for area_num in range(len(bins)):
            condition_area = (indices == area_num + 1)
            condition_af = condition_area & condition_frequency

            # 使用新的加权计算函数
            event_durations, quiet_durations = calculate_weighted_event_durations(
                raw_data, percentile_th=percentile_th, mask_array=condition_af, lat_weights=lat_weights
            )

            # 分别对两个结果进行处理
            duration_hist[p, area_num] = get_hist(event_durations)
            quiet_hist[p, area_num] = get_hist(quiet_durations)
            print(f'per:{p} area:{area_num}')

    np.save(ltp_out + f'ltp_duration_newest', duration_hist)
    np.save(ltp_out + f'ltp_quiet_newest', quiet_hist)


def ini_data(dr, bins, top_bins, era5_frequency):
    # 数据初始化
    raw_data = dr.values.squeeze()
    condition_frequency = era5_frequency > 0.3
    condition_wetday = raw_data > 1
    duration_hist = np.empty((len(top_bins), len(bins)), dtype=object)
    quiet_hist = np.empty((len(top_bins), len(bins)), dtype=object)
    return condition_frequency, condition_wetday, duration_hist, quiet_hist, raw_data

def percentile_function(data, percentile, axis):
    return np.percentile(data, 99 - percentile, axis=axis)


def apply_percentile(dataarray, percentile, time_axis):
    return xr.apply_ufunc(
        np.percentile,  # 直接使用np.percentile函数
        dataarray,  # 输入的DataArray
        99 - percentile,  # 百分位数的参数
        input_core_dims=[['time']],  # 时间轴作为核心维度
        kwargs={'axis': -1},  # 指定沿着最后一个轴计算
        dask='parallelized',  # 允许并行化
        output_dtypes=[float],  # 输出数据类型
        vectorize=True  # 向量化处理
    )


def condition_above_percentile(data, percentile=30, time_axis=0):

    # 重新分块，使时间维度只有一个块
    data = data.chunk({'time': -1})
    thresholds = data.quantile(1 - percentile / 100, dim='time')
    marked_matrix = np.zeros_like(data)

    # 对于每个(lat, lon)点，标记大于阈值的时间点为1，其他为0
    marked_matrix = data > thresholds

    return marked_matrix, thresholds
