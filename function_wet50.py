import numpy as np
from plt_temp import test_plot, plt_duration, era5_draw_area_dataArray
import matplotlib.pyplot as plt
from matplotlib import colors as clr
from plt_temp import draw_area_heap, draw_area_heap_cover, show_spectrum, show_all_spectrum, pt, get_hist, draw_hist_dq
from plt_temp import plot_precipitation_distribution
from scipy.stats import percentileofscore
from calculate_event_durations import calculate_event_durations
import plotly.graph_objs as go
import plotly.express as px
from lag_path_parameter import onat_list, onat_list_one
import seaborn as sns
import xarray as xr


def percentile_value(matrix, value):
    # 继续使用上面的matrix和value
    flattened = matrix.flatten()

    # 计算百分位数
    percentile = percentileofscore(flattened, value, kind='rank')

    print(f"The value {value} is in the {percentile}th percentile of the matrix data.")


def perform_month(era5_frequency, log_points, dr, bins, indices, sp_out, sp_test, top_bins=(40, 30, 20, 10)):
    # 数据初始化
    result_klag = np.zeros((len(top_bins), 4, len(log_points) + 2, len(bins), 100))
    duration_hist = np.empty((4, 4, 6), dtype=object)
    quiet_hist = np.empty((4, 4, 6), dtype=object)
    seasonal_data = seasonal_split(dr)
    condition_frequency = era5_frequency > 0.3
    # 数据维度检验
    # assert_para(bins, indices, log_points, result_klag)
    for p, per in enumerate(top_bins):
        condition_top, percentile_th = condition_above_percentile(dr, percentile=per)
        for season, one_season_data in seasonal_data.items():
            raw_data = one_season_data.values.squeeze()
            condition_wetday = raw_data > 1
            # 时间序列检查
            # plot_tap(onat_list, onat_list_one, one_season_data, percentile_th, f'{sp_test}{season}{per}%_', bins)

            for area_num in range(len(bins)):
                condition_area = (indices == area_num + 1)
                condition_af = condition_frequency & condition_area
                print(f'true value of area:{area_num} seanson:{season} top:{per} is {np.sum(condition_af)}')
                duration_hist[p, 4, area_num] = get_hist(calculate_event_durations(raw_data, percentile_th=percentile_th, mask_array=condition_af)[0])
                quiet_hist[p, 4, area_num] = get_hist(calculate_event_durations(raw_data, percentile_th=percentile_th, mask_array=condition_af)[1])

                print(f'per:{p} season:{season} area:{area_num}')

    np.save(sp_out + 'ltp', result_klag)
    np.save(sp_out + 'duration', duration_hist)
    np.save(sp_out + 'quiet', quiet_hist)
    return result_klag, duration_hist, quiet_hist


def perform(era5_frequency, log_points, dr, bins, indices, sp_out, sp_test, top_bins=(40, 30, 20, 10)):
    # 数据初始化
    result_klag = np.zeros((len(top_bins), len(log_points) + 2, len(bins), 100))
    raw_data = dr.values.squeeze()
    condition_frequency = era5_frequency > 0.3
    condition_wetday = raw_data > 1
    duration_hist = np.empty((4, 6), dtype=object)
    quiet_hist = np.empty((4, 6), dtype=object)

    # 数据维度检验
    assert_para(bins, indices, log_points, result_klag)

    for p, per in enumerate(top_bins):
        condition_top, percentile_th = condition_above_percentile(dr, percentile=per)

        # 时间序列检查
        plot_tap(onat_list, onat_list_one, dr, percentile_th, f'{sp_test}{per}%_', bins)

        for ind, k in enumerate(log_points):
            condition_top_lag = np.roll(condition_top, shift=k, axis=0)
            condition_top_lag = condition_top_lag & condition_wetday
            condition_top_lag[0:k, :, :] = False

            for area_num in range(len(bins)):
                condition_area = (indices == area_num + 1)
                condition_af = condition_area & condition_frequency

                if ind == 0:
                    result_klag[p, 0, area_num, :] = np.nanpercentile(raw_data[condition_wetday & condition_af], np.arange(1, 101))
                    result_klag[p, 1, area_num, :] = np.nanpercentile(raw_data[condition_top & condition_af], np.arange(1, 101))
                    duration_hist[p, area_num] = get_hist(calculate_event_durations(raw_data, percentile_th=percentile_th, mask_array=condition_af)[0])
                    quiet_hist[p, area_num] = get_hist(calculate_event_durations(raw_data, percentile_th=percentile_th, mask_array=condition_af)[1])

                result_klag[p, ind + 2, area_num, :] = np.nanpercentile(raw_data[condition_top_lag & condition_af], np.arange(1, 101))
                result_klag[p, ind + 2, area_num, :] = np.nanpercentile(raw_data[condition_top_lag & condition_af], np.arange(1, 101))

                print(f'per:{p} time:{k} area:{area_num}')

    np.save(sp_out + 'ltp', result_klag)
    np.save(sp_out + 'duration', duration_hist)
    np.save(sp_out + 'quiet', quiet_hist)
    return result_klag, duration_hist, quiet_hist


def assert_para(bins, indices, log_points, result_klag):
    assert len(bins) == indices.max()
    assert indices.min() == 1
    assert result_klag.shape == (4, 27, 6, 100)
    assert len(log_points) == 25


def plot_tap(onat_list, onat_list_one, dr, percentile_th, sp_test, bins):
    plt.close()
    dr_list = []
    th_list = []

    # 处理 onat_list_one 并进行绘图
    for lon, lat in onat_list_one:
        lon = convert_longitude(lon)
        drs = dr.sel(longitude=lon, latitude=lat, method='nearest')
        dr_list.append(drs)
        th_list.append(percentile_th.sel(longitude=lon, latitude=lat, method='nearest').values)

    pt(onat_list_one, th_list, dr_list, bins=bins, sp=f'{sp_test}time series one')
    show_all_spectrum(dr_list, bins=bins, sp=f'{sp_test}power_spectrum one')
    # plot_precipitation_distribution(dr_list, output_path=f'{sp_test}distribution')

    # 清空列表以便处理 onat_list
    dr_list.clear()
    th_list.clear()

    # 处理 onat_list 并进行绘图
    for lon, lat in onat_list:
        lon = convert_longitude(lon)
        drs = dr.sel(longitude=lon, latitude=lat, method='nearest')
        dr_list.append(drs)
        th_list.append(percentile_th.sel(longitude=lon, latitude=lat, method='nearest').values)

    pt(onat_list, th_list, dr_list, bins=bins, sp=f'{sp_test}time series')
    show_all_spectrum(dr_list, bins=bins, sp=f'{sp_test}power_spectrum')


def convert_longitude(lon):
    if lon < 0:
        return 360 + lon
    else:
        return lon


def plot_time_series_with_threshold(df, threshold, fig_name):
    print(f'th:{threshold}')
    df_2011 = df.sel(time=slice('2011-01-01', '2011-12-31'))
    df = df_2011.to_dataframe().reset_index()
    # df = df.head(100)
    df['tp'] = df['tp'].where((df['tp'] > 0.001) & (~np.isnan(df['tp'])), 0.001)
    # 假设DataFrame的第一列是日期，第二列是值
    dates = df.iloc[:, 0]
    values = df.iloc[:, 3]

    # 创建时间序列曲线
    trace = go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name=f'{fig_name}',
        line=dict(color='black')
    )

    # 创建阈值线
    threshold_line = go.Scatter(
        x=[dates.min(), dates.max()],
        y=[threshold, threshold],
        mode='lines',
        name='Threshold',
        line=dict(color='red', dash='dash')
    )

    # 创建填充区域 - 值高于阈值的部分
    fill_above = go.Scatter(
        x=np.concatenate((dates, dates[::-1])),  # x坐标正向和反向
        y=np.concatenate((np.maximum(values, threshold), np.full_like(values, threshold)[::-1])),  # y坐标为值和阈值的较大者和阈值
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.3)',  # 红色填充，透明度为0.3
        line=dict(color='rgba(255, 255, 255, 0)'),  # 设置线的颜色为透明
        showlegend=False
    )

    # 创建填充区域 - 值低于阈值的部分
    fill_below = go.Scatter(
        x=np.concatenate((dates, dates[::-1])),  # x坐标正向和反向
        y=np.concatenate((np.minimum(values, threshold), np.full_like(values, threshold)[::-1])),  # y坐标为值和阈值的较小者和阈值
        fill='toself',
        fillcolor='rgba(0, 0, 255, 0.3)',  # 蓝色填充，透明度为0.3
        line=dict(color='rgba(255, 255, 255, 0)'),  # 设置线的颜色为透明
        showlegend=False
    )

    # 将图层添加到绘图布局中
    data = [fill_below, fill_above, trace, threshold_line]

    layout = go.Layout(
        title='Time Series with Threshold',
        xaxis=dict(title='Date'),
        yaxis=dict(title='total Precipitation')
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(yaxis_type="log")
    fig.write_html(fig_name)

    # # 显示图表
    # fig.show()


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
    # 重新分块，使时间维度只有一个块
    data = data.chunk({'time': -1})
    thresholds = data.quantile(1 - percentile / 100, dim='time')
    # thresholds = apply_percentile(data, percentile, time_axis)
    # thresholds = xr.apply_ufunc(np.percentile, data, 99 - percentile, axis=time_axis)  # 计算
    # 初始化标记矩阵，其形状与原始数据相同
    marked_matrix = np.zeros_like(data)

    # 对于每个(lat, lon)点，标记大于阈值的时间点为1，其他为0
    marked_matrix = data > thresholds

    return marked_matrix, thresholds


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
def seasonal_split(ds):
    # Extract time, lon, lat dimensions
    time = ds['time']

    # Create a mask for each season
    winter_mask = (time.dt.month == 12) | (time.dt.month <= 2)
    spring_mask = (time.dt.month >= 3) & (time.dt.month <= 5)
    summer_mask = (time.dt.month >= 6) & (time.dt.month <= 8)
    autumn_mask = (time.dt.month >= 9) & (time.dt.month <= 11)

    # Use the masks to select data for each season
    seasons = {
        'winter': ds.sel(time=winter_mask),
        'spring': ds.sel(time=spring_mask),
        'summer': ds.sel(time=summer_mask),
        'autumn': ds.sel(time=autumn_mask)
    }

    return seasons
