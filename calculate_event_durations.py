import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import nolds
from plt_temp import pt
from lag_path_parameter import onat_list, path_test_png
import os


def dfa_brev(data, nvals=None, overlap=True, order=1, debug_plot=False, show=False):
    N = len(data)
    Y = np.cumsum(data - np.mean(data))

    if nvals is None:
        nvals = np.logspace(1, np.log10(N / 4), num=20).astype(int)

    F_n = []

    for n in nvals:
        rms_list = []
        for start in range(0, N, n if not overlap else n // 2):
            if start + n > N:
                break
            segment = Y[start:start + n]
            x = np.arange(n)
            poly = np.polyfit(x, segment, order)
            trend = np.polyval(poly, x)
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_list.append(rms)

        F_n.append(np.mean(rms_list))

    F_n = np.array(F_n)

    log_n = np.log(nvals)
    log_F_n = np.log(F_n)

    coeffs = np.polyfit(log_n, log_F_n, 1)

    if debug_plot or show:
        import matplotlib.pyplot as plt
        plt.plot(log_n, log_F_n, 'o', label='data')
        plt.plot(log_n, np.polyval(coeffs, log_n), '-', label='fit')
        plt.xlabel('log(n)')
        plt.ylabel('log(F(n))')
        plt.legend()
        if show:
            plt.show()

    return coeffs[0]


def perform_dfa(data):
    nvals = np.logspace(1, np.log10(len(data) // 4), num=20, dtype=int)
    # Step 1: 去平均
    data = data - np.mean(data)

    # Step 2: 计算累积和
    Y = np.cumsum(data)

    N = len(Y)
    F_n = []
    fluctuation_list = np.zeros((len(nvals), len(Y)))
    for n_ind, n in enumerate(nvals):
        if n > N:
            continue
        segments = N // n
        F_n_vals = []

        for i in range(segments):
            segment = Y[i * n:(i + 1) * n]
            x = np.arange(n)
            p = np.polyfit(x, segment, 1)  # 一阶多项式拟合
            trend = np.polyval(p, x)
            fluctuation = segment - trend
            fluctuation_list[n_ind, i * n:(i + 1) * n] = fluctuation
            F_n_vals.append(np.sqrt(np.mean(fluctuation ** 2)))

        F_n.append(np.mean(F_n_vals))

    log_n = np.log(nvals)
    log_F_n = np.log(F_n)
    slope, intercept = np.polyfit(log_n, log_F_n, 1)
    print(f"DFA exponent (slope of the log-log plot): {slope:.2f}")
    return slope


def perform_dfa_analysis(data, save_path):
    """
    对输入的时间序列数据进行去趋势波动分析 (DFA)，并将结果图表保存到指定路径。

    参数:
    data (numpy.ndarray): 输入的一维时间序列数据。
    save_path (str): 保存图表的路径。
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # 步骤1：绘制原始时间序列
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.plot(data)
    plt.title("Original Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")

    # 步骤2：计算去趋势波动分析
    dfa = nolds.dfa(data, order=1)
    print(f'functional_dfa:{dfa}')

    # 手动计算去趋势波动分析的步骤
    def detrended_fluctuation_analysis(data, nvals):
        # Step 1: 去平均
        data = data - np.mean(data)

        # Step 2: 计算累积和
        Y = np.cumsum(data)

        N = len(Y)
        F_n = []
        fluctuation_list = np.zeros((len(nvals), len(Y)))
        for n_ind, n in enumerate(nvals):
            if n > N:
                continue
            segments = N // n
            F_n_vals = []

            for i in range(segments):
                segment = Y[i * n:(i + 1) * n]
                x = np.arange(n)
                p = np.polyfit(x, segment, 1)  # 一阶多项式拟合
                trend = np.polyval(p, x)
                fluctuation = segment - trend
                fluctuation_list[n_ind, i * n:(i + 1) * n] = fluctuation
                F_n_vals.append(np.sqrt(np.mean(fluctuation ** 2)))

            F_n.append(np.mean(F_n_vals))

        return F_n, Y, fluctuation_list

    # 窗口大小
    nvals = np.logspace(1, np.log10(len(data) // 4), num=20, dtype=int)
    F_n, Y, fluctuation_list = detrended_fluctuation_analysis(data, nvals)

    # 步骤3：绘制累积和时间序列
    plt.subplot(2, 2, 2)
    plt.plot(Y)
    plt.title("Cumulative Sum of Time Series")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Sum")

    # 步骤4：绘制波动函数F(n)随窗口大小n的变化
    plt.subplot(2, 2, 3)
    for ind, fluctuation in enumerate(fluctuation_list[::10]):
        plt.plot(fluctuation, label=f'n:{nvals[::10][ind]}')
    plt.title("Fluctuation Function F(n)")
    plt.xlabel("Window size (n)")
    plt.ylabel("Fluctuation F(n)")
    plt.legend()

    # 步骤5：对数对数图及拟合直线
    log_n = np.log(nvals)
    log_F_n = np.log(F_n)
    slope, intercept = np.polyfit(log_n, log_F_n, 1)

    plt.subplot(2, 2, 4)
    plt.plot(log_n, log_F_n, 'bo-', label="Log-Log plot")
    plt.plot(log_n, slope * log_n + intercept, 'r-', label=f'Fit: slope={slope:.2f}')
    plt.title("Log-Log Plot of F(n) vs n")
    plt.xlabel("log(n)")
    plt.ylabel("log(F(n))")
    plt.legend()

    plt.tight_layout()

    # 保存图表
    plt.savefig(save_path)
    plt.close()

    print(f"DFA exponent (slope of the log-log plot): {slope:.2f}")
    return slope


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
    durations = get_duration_all(end_events, precipitation_array, start_events)
    durations_qt = get_duration_all(end_events_qt, precipitation_array, start_events_qt)
    # durations, start_indices, end_indices = get_duration(end_events, mask_array, precipitation_array, start_events)
    # durations_qt, _, _ = get_duration(end_events_qt, mask_array, precipitation_array, start_events_qt)
    # dr_list = [precipitation_array[:, 0, 10], start_events[:, 0, 10], end_events[:, 0, 10], start_events_qt[:, 0, 10], end_events_qt[:, 0, 10], end_events_qt[:, 0, 10]]
    # pt(onat_list=onat_list, th_list=[percentile_th[0, 10].values] * 6, dr_list=dr_list, sp=f'{path_test_png}test time series')

    # plot_timeseries(dr.sel(longitude=lon, latitude=lat, method='nearest'), percentile_th.sel(longitude=lon, latitude=lat, method='nearest'), use_plotly=True)
    return durations, durations_qt


def calculate_event_durations(precipitation_array, percentile_th, mask_array):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations = get_duration(end_events, mask_array, precipitation_array, start_events)
    durations_qt = get_duration(end_events_qt, mask_array, precipitation_array, start_events_qt)
    return durations, durations_qt


def calculate_event_dfa(precipitation_array, percentile_th):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations_k = get_duration_dfa(end_events, precipitation_array, start_events)
    durations_qt_k = get_duration_dfa(end_events_qt, precipitation_array, start_events_qt)
    return durations_k, durations_qt_k


def calculate_event_one_all_durations(precipitation_array, percentile_th):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations_k = get_duration_one_all(end_events, precipitation_array, start_events)
    durations_qt_k = get_duration_one_all(end_events_qt, precipitation_array, start_events_qt)
    return durations_k, durations_qt_k


def get_duration(end_events, mask_array, precipitation_array, start_events):
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


def get_duration_dfa(end_events, precipitation_array, start_events):
    all_duration = np.zeros((precipitation_array.shape[1:]))
    ind = 1
    for lat in range(precipitation_array.shape[1]):
        for lon in range(precipitation_array.shape[2]):
            start_indices = np.where(start_events[:, lat, lon])[0]
            end_indices = np.where(end_events[:, lat, lon])[0]
            event_durations = end_indices - start_indices + 1
            if ind < 3000:
                perform_dfa_analysis(event_durations, save_path=f'F:\\liusch\\remote_project\\climate_new\\temp_fig\\dfa_verification\\{ind}')
                ind += 10
            all_duration[lat, lon] = perform_dfa(event_durations)
            print(f'lat:{lat} lon:{lon} a:{all_duration[lat, lon]}')
    return all_duration


def get_duration_one_all(end_events, precipitation_array, start_events):
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
