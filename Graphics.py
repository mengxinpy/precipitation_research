import os

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from matplotlib.ticker import FuncFormatter

from Function_common import just_spectrum
from GlobalConfig import path_test_png
from GlobalConfig_graphic import setup_colorbar, auto_close_plot, format_tick
from GlobalConfig_graphic import adjust_all_font_sizes


# colors = [colors[1]] * 6


# colors = ['black'] * 6
@auto_close_plot
def scatter_plots_depart(das_low, das_mid, limited_key, save_path):
    if len(das_low) != len(das_low) or len(das_mid) != len(das_low):
        raise ValueError("The number of data arrays and variable names must be the same.")

    fig_size = len(das_low) * 5
    fig, axes = plt.subplots(len(das_low), len(das_low), figsize=(fig_size, fig_size))

    for i in range(len(das_low)):
        for j in range(len(das_low)):
            ax = axes[i, j]

            da_low_1 = das_low[i]
            da_low_2 = das_low[j]
            da_mid_1 = das_mid[i]
            da_mid_2 = das_mid[j]

            if da_low_1.shape != da_low_2.shape or da_mid_1.shape != da_mid_2.shape:
                raise ValueError("Data arrays must have the same shape.")

            flat_array_low_1 = da_low_1.values.flatten()
            flat_array_low_2 = da_low_2.values.flatten()
            flat_array_mid_1 = da_mid_1.values.flatten()
            flat_array_mid_2 = da_mid_2.values.flatten()

            mask_low = ~np.isnan(flat_array_low_1) & ~np.isnan(flat_array_low_2)
            mask_mid = ~np.isnan(flat_array_mid_1) & ~np.isnan(flat_array_mid_2)

            filtered_array_low_1 = flat_array_low_1[mask_low]
            filtered_array_low_2 = flat_array_low_2[mask_low]
            filtered_array_mid_1 = flat_array_mid_1[mask_mid]
            filtered_array_mid_2 = flat_array_mid_2[mask_mid]

            correlation_coefficient_low = np.corrcoef(filtered_array_low_1, filtered_array_low_2)[0, 1]
            correlation_coefficient_mid = np.corrcoef(filtered_array_mid_1, filtered_array_mid_2)[0, 1]

            if len(filtered_array_low_1) > 1 and len(filtered_array_low_2) > 1:
                filtered_array_low_2_for_fit = filtered_array_low_2
                filtered_array_low_1_for_fit = filtered_array_low_1

                if da_low_1.name == 'wet' and da_low_2.name in limited_key:
                    x_min, x_max = limited_key[da_low_2.name]
                    y_min, y_max = limited_key['wet']
                    mask = (filtered_array_low_2 >= x_min) & (filtered_array_low_2 <= x_max) & (filtered_array_low_1 >= y_min) & (filtered_array_low_1 <= y_max)
                    filtered_array_low_1 = filtered_array_low_1[mask]
                    filtered_array_low_2 = filtered_array_low_2[mask]
                    correlation_coefficient_low = np.corrcoef(filtered_array_low_1, filtered_array_low_2)[0, 1]
                    sns.scatterplot(x=filtered_array_low_2, y=filtered_array_low_1, ax=ax, color='yellow', alpha=0.5, s=2)

                if da_low_2.name == 'wet' and da_low_1.name in limited_key:
                    x_min, x_max = limited_key['wet']
                    y_min, y_max = limited_key[da_low_1.name]
                    mask = (filtered_array_low_2 >= x_min) & (filtered_array_low_2 <= x_max) & (filtered_array_low_1 >= y_min) & (filtered_array_low_1 <= y_max)
                    filtered_array_low_1 = filtered_array_low_1[mask]
                    filtered_array_low_2 = filtered_array_low_2[mask]
                    correlation_coefficient_low = np.corrcoef(filtered_array_low_1, filtered_array_low_2)[0, 1]
                    sns.scatterplot(x=filtered_array_low_2, y=filtered_array_low_1, ax=ax, color='yellow', alpha=0.5, s=2)

                slope_low, intercept_low = np.polyfit(filtered_array_low_2, filtered_array_low_1, 1)
                fit_line_low = np.polyval([slope_low, intercept_low], filtered_array_low_2_for_fit)
                ax.plot(filtered_array_low_2_for_fit, fit_line_low, color='blue', linestyle='--')

                sns.scatterplot(x=filtered_array_low_2_for_fit, y=filtered_array_low_1_for_fit, ax=ax, color='blue', alpha=0.5, s=2, label=f'Low k:{slope_low:.2f}')
                if not np.array_equal(filtered_array_low_2_for_fit, filtered_array_low_2):
                    sns.scatterplot(x=filtered_array_low_2, y=filtered_array_low_1, ax=ax, color='yellow', alpha=0.5, s=2)

            if len(filtered_array_mid_1) > 1 and len(filtered_array_mid_2) > 1:
                slope_mid, intercept_mid = np.polyfit(filtered_array_mid_2, filtered_array_mid_1, 1)
                fit_line_mid = np.polyval([slope_mid, intercept_mid], filtered_array_mid_2)
                ax.plot(filtered_array_mid_2, fit_line_mid, color='red', linestyle='--')

                sns.scatterplot(x=filtered_array_mid_2, y=filtered_array_mid_1, ax=ax, color='red', alpha=0.5, s=2, label=f'Mid k:{slope_mid:.2f}')
            if i == len(das_low) - 1:
                ax.set_xlabel(da_low_2.name, fontsize=32)
            if j == 0:
                ax.set_ylabel(da_low_1.name, fontsize=32)
            ax.set_title(f'Low r={correlation_coefficient_low:.2f}, Mid r={correlation_coefficient_mid:.2f}', fontsize=24)
            # 设置 x 轴和 y 轴的起始值为零
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right',markerscale=10)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    adjust_all_font_sizes(fig, scale_factor=1.5, label_scale_factor=1.7,title_scale_factor=1)
    plt.savefig(f'{save_path}_correlation')
    plt.show()


@auto_close_plot
def scatter_plots(matrices, var_names, figure_name, save_path=path_test_png):
    # 检查输入是否正确
    if len(matrices) != len(var_names):
        raise ValueError("矩阵和变量名称的数量必须相同。")

    # 创建一个图形和子图
    fig, axes = plt.subplots(len(matrices), len(matrices), figsize=(6 * len(matrices), 6 * len(matrices)))

    # 遍历每个矩阵
    for i in range(len(matrices)):
        for j in range(len(matrices)):
            # 获取当前的子图
            ax = axes[i, j]

            # 获取当前的两个矩阵
            matrix1 = np.array(matrices[i])
            matrix2 = np.array(matrices[j])

            # 检查两个矩阵是否具有相同的形状
            if matrix1.shape != matrix2.shape:
                raise ValueError("两个输入矩阵的形状必须相同。")

            # 将矩阵展平为一维数组
            flat_array1 = matrix1.flatten()
            flat_array2 = matrix2.flatten()

            # 创建一个掩码来忽略带有NaN值的点
            mask = ~np.isnan(flat_array1) & ~np.isnan(flat_array2)

            # 使用掩码来过滤数据
            filtered_array1 = flat_array1[mask]
            filtered_array2 = flat_array2[mask]

            # 计算相关系数
            correlation_coefficient = np.corrcoef(filtered_array1, filtered_array2)[0, 1]

            # 绘制散点图
            sns.scatterplot(x=filtered_array2, y=filtered_array1, ax=ax, color='blue', alpha=0.5, s=2)

            # 计算并绘制拟合直线
            if len(filtered_array1) > 1 and len(filtered_array2) > 1:  # 检查是否有足够的数据点进行拟合
                slope, intercept = np.polyfit(filtered_array2, filtered_array1, 1)
                fit_line = np.polyval([slope, intercept], filtered_array2)
                ax.plot(filtered_array2, fit_line, color='red', linestyle='--')

            # 设置标题和坐标轴标签
            if i == len(matrices) - 1:
                ax.set_xlabel(var_names[j], fontsize=32)
            if j == 0:
                ax.set_ylabel(var_names[i], fontsize=32)
            ax.set_title(f'r={correlation_coefficient:.2f}', fontsize=32)

    # 调整图形布局
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # 显示图形
    plt.savefig(save_path + figure_name)


@auto_close_plot
def plot_interactive_contour(dataarray, bins):
    # 提取纬度、经度和值
    lat = dataarray['latitude'].values
    lon = dataarray['longitude'].values
    values = dataarray.values.squeeze()

    # 转换颜色格式为 'rgba(255, 255, 255, 1.0)' 并与位置配对
    custom_colorscale = [
        [pos, f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, {c[3]})'] for pos, c in zip(bins, colors)
    ]
    # 创建等高线图
    contour = go.Contour(
        z=values,
        x=lon,
        y=lat,
        colorscale=custom_colorscale,
        contours=dict(
            start=bins[0],
            end=bins[-1],
            size=(bins[-1] - bins[0]) / (len(bins) - 1)
        ),
        hoverinfo='x+y+z'  # 在鼠标悬停时显示坐标和值
    )

    # 创建图表布局
    layout = go.Layout(
        title='Interactive Contour Plot',
        xaxis=dict(title='Longitude'),
        yaxis=dict(title='Latitude')
    )

    # 创建图表对象
    fig = go.Figure(data=[contour], layout=layout)

    # 显示图表
    fig.show()


@auto_close_plot
def pt(onat_list, th_list, dr_list, bins, sp):
    fig, axs = plt.subplots(6, figsize=(40, 30), constrained_layout=True)  # 创建6个子图
    _, _, colors = setup_colorbar(fig=fig, vbins=bins)

    for idx, (dr, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')
        if isinstance(dr, xr.DataArray):
            # 如果dr是xarray.DataArray，使用.sel()方法
            df_2011 = dr.sel(time=slice('2011-01-01', '2014-12-31'))
        elif isinstance(dr, np.ndarray):
            # 如果dr是numpy.ndarray，使用数组切片
            df_2011 = dr[0:365]
        else:
            raise TypeError("dr 必须是 xarray.DataArray 或 numpy.ndarray 类型")

        # 现在 df_2011 包含了2011年的数据

        # 在每个子图上绘制
        axs[idx].plot(df_2011, label=f'LON/LAT {onat}', color=colors[idx])
        axs[idx].axhline(y=th, color=colors[idx], linestyle='--')  # 绘制阈值线
        axs[idx].set_yscale('log')
        # 设置标题和标签
        axs[idx].set_title(f'Area:{idx + 1} Thresholds: {th:.2f}', fontsize=24)
        if idx == 5:
            axs[idx].set_xlabel('Time', fontsize=28)
        else:
            axs[idx].set_xticklabels([])
        axs[idx].set_ylabel('Precipitation', fontsize=24)
        axs[idx].set_xlim(0, 365)
        axs[idx].set_ylim(0.0001, 100)
        axs[idx].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=28)
        axs[idx].tick_params(axis='both', labelsize=18)
        # 保存图表
    plt.savefig(sp, bbox_inches='tight')


@auto_close_plot
def show_all_spectrum(dr_list, bins, sp):
    fig, axs = plt.subplots(6, figsize=(20, 30), constrained_layout=True)  # 创建6个子图
    _, _, colors = setup_colorbar(fig=fig, vbins=bins)

    for idx, drs in enumerate(dr_list):
        period, inverse_X = just_spectrum(drs)
        print(f'power{np.sum(inverse_X[0:14])}')
        axs[idx].plot(period, inverse_X, color=colors[idx])
        if idx == 5:
            axs[idx].set_xlabel('Period (year)', fontsize=28)
        axs[idx].set_ylabel('Intensity', fontsize=24)
        axs[idx].tick_params(axis='y', labelsize=10)
        axs[idx].set_xlim(0, 4)
        axs[idx].set_ylim(0, 0.004)
        axs[idx].tick_params(axis='both', labelsize=18)

    # 保存并显示图表
    plt.savefig(sp, bbox_inches='tight')


@auto_close_plot
def era5_draw_area_dataArray(dataArray, sp):
    print(f'max:{dataArray.max().values}')
    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes(projection=ccrs.PlateCarree())
    # 创建一个LogNorm实例
    norm = mcolors.LogNorm(0.01, vmax=dataArray.max().values)

    # 创建一个等高线图，使用对数刻度
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    c = ax.contourf(dataArray.longitude, dataArray.latitude, dataArray,
                    transform=ccrs.PlateCarree(), levels=20, cmap='rainbow', norm=norm)

    # 添加颜色条
    plt.colorbar(c, ax=ax, orientation='vertical', label='Log Scaled Values')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    plt.savefig(sp)


@auto_close_plot
def draw_all_era5_area(dataarray, sp):
    # 接着，筛选出2001年的数据
    dataarray_2014 = dataarray.sel(time=dataarray['time'].dt.year == 2014)
    # 现在，遍历每一天，并调用era5_draw_area_dataArray函数
    for day in dataarray_2014['time']:
        # 提取当天的数据
        one_day_data = dataarray_2014.sel(time=day)

        # 构造文件名，例如 "data_20010101"
        name = one_day_data['time'].dt.strftime('data_%Y%m%d').item()
        print(name)

        # 调用函数绘制并保存图像
        era5_draw_area_dataArray(one_day_data, f'{sp}all_area_amsr2\\{name}')


@auto_close_plot
def draw_hist_data_collapse(durations, title, vbins, fig_name):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20), constrained_layout=True)
    # 将二维数组转换为一维数组
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=24)

    _, _, colors = setup_colorbar(fig=fig, vbins=vbins)

    for area, p_dur in enumerate(durations.transpose((1, 0))):

        for p, dur in enumerate(p_dur):
            bin_centers, hist = dur
            dur_mean = np.sum(bin_centers * hist)

            print(f'mean:{dur_mean}')
            x = bin_centers
            y = hist
            y[y == 0] = np.nan

            mask = (x > 5) & (x < 40) & ~np.isnan(y)
            x_lim = x[mask]
            y_lim = y[mask]

            scale = dur_mean
            xlim_scale = x_lim / scale
            # 绘制线图表示概率分布
            # axs[area].loglog(bin_centers / scale, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3))
            axs[area].loglog(bin_centers / scale, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3), label=['top40%', 'top30%', 'top20%', 'top10%'][p])
        # 设置标题和标签
        axs[area].set_title(f'Area:{area + 1}', fontsize=24)
        axs[area].set_ylabel('Probability', fontsize=24)
        axs[area].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=24)
        axs[area].tick_params(axis='both', labelsize=18)

    plt.savefig(fig_name)


@auto_close_plot
def draw_hist_dq_dataarray(durations, title, fig_name):
    # 创建目录
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20), constrained_layout=True)
    # 将二维数组转换为一维数组
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=24)

    _, _, colors = setup_colorbar(fig=fig, vbins=4)

    for area_ind, area in enumerate(durations.coords['area'].values):
        for season_ind, season in enumerate(durations.coords['season'].values):
            bin_centers, hist = durations.sel(season=season, area=slice(area, area)).values[0]
            dur_mean = np.sum(bin_centers * hist)
            # 计算二阶矩 (方差)
            variance = np.sum(hist * (bin_centers - dur_mean) ** 2)

            # 计算三阶矩 (偏度)
            skewness = np.sum(hist * (bin_centers - dur_mean) ** 3)
            print(f'mean:{dur_mean}')
            x = bin_centers
            y = hist
            y[y == 0] = np.nan

            mask = (x > 5) & (x < 40) & ~np.isnan(y)
            x_lim = x[mask]
            y_lim = y[mask]

            scale = dur_mean
            xlim_scale = x_lim / scale
            axs[area].loglog(bin_centers, hist, '*', color=colors[season_ind], label=durations.coords['season'].values[season_ind])
        axs[area].set_title(f'Area:{area + 1}', fontsize=24)
        axs[area].set_xlim(1, 1000)
        axs[area].set_ylabel('Probability', fontsize=24)
        axs[area].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=24)
        axs[area].tick_params(axis='both', labelsize=18)

    plt.savefig(fig_name)


@auto_close_plot
def draw_hist_dq(durations, title, vbins, fig_name):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20), constrained_layout=True)
    # 将二维数组转换为一维数组
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=24)

    _, _, colors = setup_colorbar(fig=fig, vbins=vbins)
    for area, p_dur in enumerate(durations.transpose((1, 0))):

        for p, dur in enumerate(p_dur):
            bin_centers, hist = dur
            dur_mean = np.sum(bin_centers * hist)
            # 计算二阶矩 (方差)
            variance = np.sum(hist * (bin_centers - dur_mean) ** 2)

            # 计算三阶矩 (偏度)
            skewness = np.sum(hist * (bin_centers - dur_mean) ** 3)
            print(f'mean:{dur_mean}')
            x = bin_centers
            y = hist
            y[y == 0] = np.nan

            mask = (x > 5) & (x < 40) & ~np.isnan(y)
            x_lim = x[mask]
            y_lim = y[mask]

            scale = dur_mean
            xlim_scale = x_lim / scale
            # 绘制线图表示概率分布
            # axs[area].loglog(bin_centers / scale, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3))
            axs[area].loglog(bin_centers, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3), label=['top40%', 'top30%', 'top20%', 'top10%'][p])
        axs[area].set_title(f'Area:{area + 1}', fontsize=24)
        axs[area].set_ylabel('Probability', fontsize=24)
        axs[area].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=24)
        axs[area].tick_params(axis='both', labelsize=18)

    plt.savefig(fig_name)


@auto_close_plot
def plt_duration(durations, title, vbins, fig_name):
    fig = plt.figure(figsize=(25, 10), constrained_layout=True)
    fig.tight_layout()

    _, _, colors = setup_colorbar(fig=fig, vbins=vbins)
    # 设置 bins 的边界为对数刻度
    avg = np.zeros(6)
    std = np.zeros(6)
    med = np.zeros(6)
    covs = np.zeros(6)
    plt.subplot(1, 2, 1)
    for ind, dur in enumerate(durations):
        bin_centers, hist = dur
        # bin_centers, hist = get_hist(dur)

        avg[ind] = np.mean(dur)
        med[ind] = np.median(dur)
        std[ind] = np.std(dur)

        x = bin_centers
        y = hist
        y[y == 0] = np.nan
        mask = (x > 5) & (x < 50) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        # 绘制线图表示概率分布
        plt.loglog(bin_centers, hist, '*', color=colors[ind])
        log_x = np.log(x)
        log_y = np.log(y)
        # 选择需要拟合的数据部分，例如x值在4到8之间

        # 进行直线拟合，polyfit返回拟合系数，这里1代表一次多项式，即直线
        coefficients = np.polyfit(log_x, log_y, 1)
        covs[ind] = coefficients[0]

        # 使用拟合系数构建拟合直线的y值
        log_y_fit = np.polyval(coefficients, log_x)

        y_fit = np.exp(log_y_fit)

        # 在双对数坐标下绘制拟合的直线
        plt.loglog(x, y_fit, color=colors[ind], label=f'k={coefficients[0]:.2f}')

        # # 绘制拟合的直线
        # plt.plot(log_x, y_line, color=colors[ind])

    # 设置图表的标题和坐标轴标签
    plt.title(f'Probability Distribution of {title}', fontsize=24)
    plt.xlabel(f'{title}(day)', fontsize=24)
    plt.ylabel('Probability', fontsize=24)

    # 添加网格线
    plt.grid(axis='y', alpha=0.75)
    current_xticks = plt.xticks()[0]
    # 设置新的刻度标签和字体大小
    plt.xticks(current_xticks, fontsize=18)
    current_yticks = plt.yticks()[0]
    # 设置新的刻度标签和字体大小
    plt.yticks(current_yticks, fontsize=18)
    # plt.legend(fontsize=18)
    plt.xlim(1, 500)
    plt.ylim(1e-9, 1)

    plt.subplot(1, 2, 2)
    # 使用plot函数绘制三条曲线
    line1 = plt.plot(vbins, std, 'r*-', linewidth=2, markersize=8, label='Standard Deviation', alpha=0.5)[0]
    line2 = plt.plot(vbins, med, 'b*-', linewidth=2, markersize=8, label='Median', alpha=0.5)[0]
    line3 = plt.plot(vbins, avg, 'g*-', linewidth=2, markersize=8, label='Average', alpha=0.5)[0]
    plt.xticks(vbins, fontsize=18)
    plt.xlabel('Frequency', fontsize=24)
    plt.ylabel('Value', fontsize=24)
    current_xticks = plt.xticks()[0]
    # 设置新的刻度标签和字体大小
    plt.xticks(current_xticks, fontsize=18)
    plt.title('parameter', fontsize=24)
    current_yticks = plt.yticks()[0]
    # 设置新的刻度标签和字体大小
    plt.yticks(current_yticks, fontsize=18)
    # plt.legend(fontsize=18)
    # 创建一个共享x轴的新轴对象
    ax2 = plt.twinx()
    line4 = ax2.plot(vbins, covs, color='black', marker='*', linestyle='--', linewidth=2, markersize=8, label='k')[0]  # 使用黄色来区分

    # 设置y轴的标签
    ax2.set_ylabel('Coefficient', fontsize=24)
    # 创建格式化器并应用
    # 设置第二个 y 轴的刻度标签的字体大小
    ax_current_yticks = ax2.get_yticks()
    ax2.set_yticklabels(ax_current_yticks, fontsize=18)
    formatter = FuncFormatter(format_tick)
    ax2.yaxis.set_major_formatter(formatter)
    # ax2.set_ylim(1, 1e-2)

    # 分别为两个y轴添加图例
    # 或者，更简单的方法，直接使用已有的line对象：
    legend_handles = [line1, line2, line3, line4]

    # 创建一个统一的图例
    ax2.legend(handles=legend_handles, loc='upper right', fontsize=18)

    plt.savefig(fig_name)
