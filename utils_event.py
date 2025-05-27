import numpy as np
from numba import jit
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import nolds
from plot_lib import pt_6
from config import onat_list, path_test_png
import os


def calculate_every_event_durations(precipitation_array, percentile_th):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations = get_duration_all(end_events, precipitation_array, start_events)
    durations_qt = get_duration_all(end_events_qt, precipitation_array, start_events_qt)
    return durations, durations_qt


def calculate_event_durations(precipitation_array, percentile_th, mask_array):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations = get_duration(end_events, mask_array, precipitation_array, start_events)
    durations_qt = get_duration(end_events_qt, mask_array, precipitation_array, start_events_qt)
    return durations, durations_qt


@jit(nopython=True)
def get_duration(end_events, mask_array, precipitation_array, start_events):
    # 将 mask_array 展开成一维数组
    mask_flat = mask_array.flatten()

    # 将 start_events 和 end_events 展开成二维数组，形状为 (时间, 纬度 * 经度)
    start_flat = start_events.reshape(start_events.shape[0], -1)
    end_flat = end_events.reshape(end_events.shape[0], -1)

    # 使用布尔索引选择被掩码选中的位置
    start_flat = start_flat[:, mask_flat]
    end_flat = end_flat[:, mask_flat]

    # 初始化一个空列表来存储所有的事件持续时间
    durations = []

    # 对每个被掩码选中的位置计算持续时间
    for i in range(start_flat.shape[1]):
        start_indices = np.where(start_flat[:, i])[0]
        end_indices = np.where(end_flat[:, i])[0]
        event_durations = end_indices - start_indices + 1
        durations.extend(event_durations)

    # 将 durations 转换为 NumPy 数组
    durations = np.array(durations)
    return durations


def get_duration_all(end_events, precipitation_array, start_events):
    avg_duration = np.zeros((precipitation_array.shape[1:]))
    for lat in range(precipitation_array.shape[1]):
        for lon in range(precipitation_array.shape[2]):
            start_indices = np.where(start_events[:, lat, lon])[0]
            end_indices = np.where(end_events[:, lat, lon])[0]
            event_durations = end_indices - start_indices + 1
            avg_duration[lat, lon] = np.mean(event_durations)
    return avg_duration


# 自定义函数，将NaN和小于等于0的值替换为0.0001
def replace_values(x):
    if pd.isna(x) or x <= 0:
        return 0.0001
    else:
        return x


def calculate_latitude_weights(latitudes):
    """计算纬度权重（基于余弦值）"""
    return np.cos(np.deg2rad(latitudes))


def calculate_weighted_event_durations(precipitation_array, percentile_th, mask_array, lat_weights):
    """
    计算加权的事件持续时间
    在区域内根据纬度权重计算加权平均的事件持续时间
    """
    # 获取事件开始和结束标记
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    
    # 计算加权的持续时间
    weighted_durations = get_weighted_duration(end_events, mask_array, precipitation_array, start_events, lat_weights)
    weighted_durations_qt = get_weighted_duration(end_events_qt, mask_array, precipitation_array, start_events_qt, lat_weights)
    
    return weighted_durations, weighted_durations_qt


def get_weighted_duration(end_events, mask_array, precipitation_array, start_events, lat_weights):
    """
    根据纬度权重计算加权的持续时间
    """
    # 展开数组
    mask_flat = mask_array.flatten()
    
    # 确保lat_weights的维度与mask_array匹配
    lat_weights_expanded = np.tile(lat_weights, (1, mask_array.shape[1]))
    lat_weights_flat = lat_weights_expanded.flatten()
    
    # 展开时间序列数据
    start_flat = start_events.reshape(start_events.shape[0], -1)
    end_flat = end_events.reshape(end_events.shape[0], -1)
    
    # 选择被掩码选中的位置
    start_selected = start_flat[:, mask_flat]
    end_selected = end_flat[:, mask_flat]
    weights_selected = lat_weights_flat[mask_flat]
    
    # 计算每个格点的持续时间并应用权重
    weighted_durations = []
    
    for i in range(start_selected.shape[1]):
        start_indices = np.where(start_selected[:, i])[0]
        end_indices = np.where(end_selected[:, i])[0]
        
        if len(start_indices) > 0 and len(end_indices) > 0:
            # 确保开始和结束事件配对
            min_len = min(len(start_indices), len(end_indices))
            event_durations = end_indices[:min_len] - start_indices[:min_len] + 1
            
            # 对每个事件应用该格点的权重
            weight = weights_selected[i]
            weighted_event_durations = event_durations * weight
            weighted_durations.extend(weighted_event_durations)
    
    return np.array(weighted_durations)
