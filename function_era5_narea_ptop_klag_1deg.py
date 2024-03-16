import numpy as np
from plt_temp import test_plot
import matplotlib.pyplot as plt
from matplotlib import colors as clr
from plt_temp import draw_area_heap, draw_area_heap_cover
from scipy.stats import percentileofscore


def percentile_value(matrix, value):
    # 继续使用上面的matrix和value
    flattened = matrix.flatten()

    # 计算百分位数
    percentile = percentileofscore(flattened, value, kind='rank')

    print(f"The value {value} is in the {percentile}th percentile of the matrix data.")


def era5_narea_ptop_klag_1deg(log_points, dr, bins, indices, sp_out, top_bins=(30, 20)):
    # 数据初始化
    result_klag = np.zeros((len(top_bins), len(log_points) + 2, len(bins), 100))
    raw_data = dr.values.squeeze()
    condition_wetday = raw_data > 1

    # 数据维度检验
    assert len(bins) == indices.max()
    assert indices.min() == 1
    assert result_klag.shape == (2, 27, 6, 100)
    assert len(log_points) == 25

    for p, per in enumerate(top_bins):
        # condition_top = True
        condition_top = condition_above_percentile(raw_data, percentile=per)  # todo: debug and test
        condition_top_wetday = condition_top & condition_wetday
        for ind, k in enumerate(log_points):
            condition_top_lag = np.roll(condition_top_wetday, shift=k, axis=0)
            condition_top_lag = condition_top_lag & condition_wetday
            condition_top_lag[0:k, :, :] = False
            for area_num in range(len(bins)):
                condition_area = (indices == area_num + 1)
                if ind == 0:
                    result_klag[p, 0, area_num, :] = np.nanpercentile(raw_data[condition_wetday & condition_area & np.roll(condition_wetday, shift=-k, axis=0)], np.arange(1, 101))
                    # result_klag[p, 0, area_num, :] = np.nanpercentile(raw_data[condition_wetday & condition_area], np.arange(1, 101))
                    result_klag[p, 1, area_num, :] = np.nanpercentile(raw_data[condition_top_wetday & condition_area], np.arange(1, 101))
                result_klag[p, ind + 2, area_num, :] = np.nanpercentile(raw_data[condition_top_lag & condition_area], np.arange(1, 101))
                if p == 0 and area_num == 3 and ind == 1:
                    for n in range(1, 31):
                        draw_area_heap_cover(raw_data[n], (condition_area & condition_wetday[n], condition_top_lag[n] & condition_area), name=f'self_top{n}')
                    # percentile_value(raw_data[:, condition_area], 1)
                print(f'per:{p} time:{k} area:{area_num}')

    np.save(sp_out, result_klag)
    return result_klag


def condition_above_percentile(data, percentile=30, time_axis=0):
    """
    标记矩阵中沿给定时间维度大于某个百分位数的元素。

    参数:
    data -- 输入的三维数据矩阵，维度为 (time, lat, lon)
    percentile -- 要计算的百分位数，默认为30
    time_axis -- 时间维度在矩阵中的索引，默认为0

    返回:
    一个标记矩阵，其中大于各自阈值的元素被标记为1，其他被标记为0
    """
    # 计算每个(lat, lon)点在时间维度上的指定百分位数的阈值
    thresholds = np.percentile(data, 99 - percentile, axis=time_axis)

    # 初始化标记矩阵，其形状与原始数据相同
    marked_matrix = np.zeros_like(data)

    # 对于每个(lat, lon)点，标记大于阈值的时间点为1，其他为0
    marked_matrix = data > thresholds

    return marked_matrix


def get_toparea(condition_topper_area, condition_wetday, raw_data):
    condition_topper_wetday = condition_topper_area & condition_wetday
    topper_area = np.sum(condition_topper_wetday, axis=0)
    topper_area = topper_area / topper_area.max()
    topper_area[topper_area < (np.nanpercentile(topper_area[topper_area > 0], 70))] = 0
    rain_toparea = raw_data[:, topper_area > 0]
    toparea_percentile = np.nanpercentile(rain_toparea[rain_toparea > 1], np.arange(1, 101))
    draw_area_heap(topper_area, 'topper_area')
    return toparea_percentile, topper_area

# np.save(f'{path_out}\\result_klag_1deg_6area_topper', result_klag)
