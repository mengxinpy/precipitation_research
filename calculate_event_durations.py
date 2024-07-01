import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from plt_temp import pt
from lag_path_parameter import onat_list, path_test_png


def calculate_and_draw_distribution_slope(data, save_path=None):
    # 计算直方图
    if np.isnan(data).any():
        return np.nan

    base = 30 ** (1 / (15 - 1))  # 生成对数点并取最近的整数
    log_points = np.unique([int(round(1 * base ** n)) for n in range(15)])
    bins = log_points
    # bins = np.logspace(np.log10(2), np.log10(30), num=50)
    filtered_data = data[(data > bins.min()) & (data < bins.max())]
    hist, bin_edges = np.histogram(filtered_data, bins=bins, density=True)

    # 计算每个bin的中心点
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 过滤掉直方图中的0值，因为对数0是未定义的
    non_zero_mask = hist > 0
    hist_filtered = hist[non_zero_mask]
    bin_centers_filtered = bin_centers[non_zero_mask]

    # 对非零的bin中心和频率取对数
    log_bin_centers = np.log(bin_centers_filtered)
    log_hist = np.log(hist_filtered)

    # 使用线性回归拟合直方图的双对数
    coefficients = np.polyfit(log_bin_centers, log_hist, 1)
    slope = coefficients[0]

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(log_bin_centers, log_hist, marker='o', color='b', label='Data Points')
    plt.plot(log_bin_centers, np.polyval(coefficients, log_bin_centers), color='r', label='Linear Fit')

    # 设置图形标题和标签
    plt.title('Log-log Plot of Histogram Data')
    plt.xlabel('Log of Bin Centers')
    plt.ylabel('Log of Histogram Density')

    # 添加图例
    plt.legend()

    # 如果提供了保存路径，则保存图形
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")
    plt.close()

    # 显示图形
    # plt.show()

    return slope


def calculate_distribution_slope(data):
    # 计算直方图
    if np.isnan(data).any():
        return np.nan

    base = 30 ** (1 / (15 - 1))  # 生成对数点并取最近的整数
    log_points = np.unique([int(round(1 * base ** n)) for n in range(15)])
    bins = log_points
    # bins = np.logspace(np.log10(2), np.log10(30), num=50)
    filtered_data = data[(data > bins.min()) & (data < bins.max())]
    hist, bin_edges = np.histogram(filtered_data, bins=bins, density=True)

    # 计算每个bin的中心点
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 过滤掉直方图中的0值，因为对数0是未定义的
    non_zero_mask = hist > 0
    hist_filtered = hist[non_zero_mask]
    bin_centers_filtered = bin_centers[non_zero_mask]

    # 对非零的bin中心和频率取对数
    log_bin_centers = np.log(bin_centers_filtered)
    log_hist = np.log(hist_filtered)

    # 使用线性回归拟合直方图的双对数
    coefficients = np.polyfit(log_bin_centers, log_hist, 1)
    slope = coefficients[0]

    return slope


def plot_timeseries(dataarray, th, use_plotly=False):
    if 'time' not in dataarray.dims:
        raise ValueError("The provided DataArray does not contain a 'time' dimension.")

    # Convert the DataArray to a DataFrame
    df = dataarray.to_dataframe().reset_index()

    # Plot using seaborn or plotly
    if use_plotly:
        df['time'] = pd.to_datetime(df['time'])
        fig = px.line(df, x='time', y=dataarray.name)
        fig.add_hline(y=th.values, line=dict(color="red", width=3, opacity=0.5))
        fig.update_layout(yaxis_type="log")
        fig.write_html("F:/liusch/remote_project/climate_new/temp_fig/durations_time/plotly_timeseries.html")

    else:
        # Plot using Seaborn
        sns.lineplot(data=df, x='time', y=dataarray.name)
        plt.savefig("F:/liusch/remote_project/climate_new/temp_fig/durations_time/seaborn_timeseries.png")
        plt.close()


# Example usage:
# Assume 'da' is an xarray DataArray with a 'time' dimension
# plot_timeseries(da, use_plotly=True)  # Plot using Plotly
# plot_timeseries(da, use_plotly=False) # Plot using Seaborn


def calculate_every_event_durations(precipitation_array, percentile_th):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations = method_name_all(end_events, precipitation_array, start_events)
    durations_qt = method_name_all(end_events_qt, precipitation_array, start_events_qt)
    # durations, start_indices, end_indices = method_name(end_events, mask_array, precipitation_array, start_events)
    # durations_qt, _, _ = method_name(end_events_qt, mask_array, precipitation_array, start_events_qt)
    # dr_list = [precipitation_array[:, 0, 10], start_events[:, 0, 10], end_events[:, 0, 10], start_events_qt[:, 0, 10], end_events_qt[:, 0, 10], end_events_qt[:, 0, 10]]
    # pt(onat_list=onat_list, th_list=[percentile_th[0, 10].values] * 6, dr_list=dr_list, sp=f'{path_test_png}test time series')

    # plot_timeseries(dr.sel(longitude=lon, latitude=lat, method='nearest'), percentile_th.sel(longitude=lon, latitude=lat, method='nearest'), use_plotly=True)
    return durations, durations_qt


def calculate_event_durations(precipitation_array, percentile_th, mask_array):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations = method_name(end_events, mask_array, precipitation_array, start_events)
    durations_qt = method_name(end_events_qt, mask_array, precipitation_array, start_events_qt)
    # durations, start_indices, end_indices = method_name(end_events, mask_array, precipitation_array, start_events)
    # durations_qt, _, _ = method_name(end_events_qt, mask_array, precipitation_array, start_events_qt)
    # dr_list = [precipitation_array[:, 0, 10], start_events[:, 0, 10], end_events[:, 0, 10], start_events_qt[:, 0, 10], end_events_qt[:, 0, 10], end_events_qt[:, 0, 10]]
    # pt(onat_list=onat_list, th_list=[percentile_th[0, 10].values] * 6, dr_list=dr_list, sp=f'{path_test_png}test time series')

    # plot_timeseries(dr.sel(longitude=lon, latitude=lat, method='nearest'), percentile_th.sel(longitude=lon, latitude=lat, method='nearest'), use_plotly=True)
    return durations, durations_qt


def calculate_event_one_all_durations(precipitation_array, percentile_th):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations_k = method_name_one_all(end_events, precipitation_array, start_events)
    durations_qt_k = method_name_one_all(end_events_qt, precipitation_array, start_events_qt)
    # durations, start_indices, end_indices = method_name(end_events, mask_array, precipitation_array, start_events)
    # durations_qt, _, _ = method_name(end_events_qt, mask_array, precipitation_array, start_events_qt)
    # dr_list = [precipitation_array[:, 0, 10], start_events[:, 0, 10], end_events[:, 0, 10], start_events_qt[:, 0, 10], end_events_qt[:, 0, 10], end_events_qt[:, 0, 10]]
    # pt(onat_list=onat_list, th_list=[percentile_th[0, 10].values] * 6, dr_list=dr_list, sp=f'{path_test_png}test time series')

    # plot_timeseries(dr.sel(longitude=lon, latitude=lat, method='nearest'), percentile_th.sel(longitude=lon, latitude=lat, method='nearest'), use_plotly=True)
    return durations_k, durations_qt_k


def method_name(end_events, mask_array, precipitation_array, start_events):
    durations = []
    for lat in range(precipitation_array.shape[1]):
        for lon in range(precipitation_array.shape[2]):
            print(f'lat:{lat} lon:{lon}')
            if not mask_array[lat, lon]:
                continue
            start_indices = np.where(start_events[:, lat, lon])[0]
            end_indices = np.where(end_events[:, lat, lon])[0]
            event_durations = end_indices - start_indices + 1
            durations.extend(event_durations)
    durations = np.array(durations)
    return durations


def method_name_one_all(end_events, precipitation_array, start_events):
    all_duration = np.zeros((precipitation_array.shape[1:]))
    ind = 1
    for lat in range(precipitation_array.shape[1]):
        for lon in range(precipitation_array.shape[2]):
            start_indices = np.where(start_events[:, lat, lon])[0]
            end_indices = np.where(end_events[:, lat, lon])[0]
            event_durations = end_indices - start_indices + 1
            if ind < 30:
                all_duration[lat, lon] = calculate_and_draw_distribution_slope(event_durations,
                                                                               save_path=f'F:\\liusch\\remote_project\\climate_new\\temp_fig\\distribution_varification\\{ind}')
                ind += 1
            all_duration[lat, lon] = calculate_distribution_slope(event_durations)
    return all_duration


def method_name_all(end_events, precipitation_array, start_events):
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
