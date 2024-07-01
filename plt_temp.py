import os
import seaborn as sns
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from matplotlib import colors as clr
import numpy.fft as fft
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from lag_path_parameter import path_test_png

# 设置全局的字体大小
plt.rcParams['axes.titlesize'] = 20  # 子图标题的字体大小
plt.rcParams['axes.labelsize'] = 16  # x和y轴标签的字体大小
plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签的字体大小
plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签的字体大小
plt.rcParams['legend.fontsize'] = 14  # 图例的字体大小


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


cdict = {
    'red': ((0.0, inter_from_256(64), inter_from_256(64)),
            (1 / 5 * 1, inter_from_256(102), inter_from_256(102)),
            (1 / 5 * 2, inter_from_256(235), inter_from_256(235)),
            (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
            (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
            (1.0, inter_from_256(169), inter_from_256(169))),
    'green': ((0.0, inter_from_256(57), inter_from_256(57)),
              (1 / 5 * 1, inter_from_256(178), inter_from_256(178)),
              (1 / 5 * 2, inter_from_256(240), inter_from_256(240)),
              (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
              (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
              (1 / 5 * 5, inter_from_256(23), inter_from_256(23))),
    'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
             (1 / 5 * 1, inter_from_256(255), inter_from_256(255)),
             (1 / 5 * 2, inter_from_256(185), inter_from_256(185)),
             (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
             (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
             (1.0, inter_from_256(69), inter_from_256(69))),
}

# all_area_num = data_percentile.shape[0]
cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=6)  # todo:未参数化
colors = cmap(np.linspace(0, 1, 6))


# colors = [colors[1]] * 6


# colors = ['black'] * 6


def scatter_plots(matrices, var_names, figure_name, save_path=path_test_png):
    """
    绘制多个矩阵之间的散点图和相关系数。

    参数:
    matrices (list): 包含要绘制的矩阵的列表。
    var_names (list): 每个矩阵对应的变量名称的列表。
    """

    # 检查输入是否正确
    if len(matrices) != len(var_names):
        raise ValueError("矩阵和变量名称的数量必须相同。")

    # 创建一个图形和子图
    fig, axes = plt.subplots(len(matrices), len(matrices), figsize=(24, 24))

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

            # 设置标题和坐标轴标签
            if i == len(matrices) - 1:
                ax.set_xlabel(var_names[j])
            if j == 0:
                ax.set_ylabel(var_names[i])
            ax.set_title(f'r={correlation_coefficient:.2f}')

    # 调整图形布局
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # 显示图形
    plt.savefig(path_test_png + figure_name)
    # plt.show()


def scatter_plot(matrix1, matrix2, var):
    # 将矩阵转化为numpy数组
    array1 = np.array(matrix1)
    array2 = np.array(matrix2)

    # 检查两个矩阵是否具有相同的形状
    if array1.shape != array2.shape:
        raise ValueError("两个输入矩阵的形状必须相同")

    # 将矩阵展平为一维数组
    flat_array1 = array1.flatten()
    flat_array2 = array2.flatten()

    # 创建一个掩码来忽略带有NaN值的点
    mask = ~np.isnan(flat_array1) & ~np.isnan(flat_array2)

    # 使用掩码来过滤数据
    filtered_array1 = flat_array1[mask]
    filtered_array2 = flat_array2[mask]

    # 计算相关系数
    correlation_coefficient = np.corrcoef(filtered_array1, filtered_array2)[0, 1]

    # 创建一个散点图

    plt.scatter(filtered_array1, filtered_array2, color='blue', alpha=0.5, s=2)  # 改变颜色为蓝色，透明度为0.5，大小为10
    plt.xlabel(var)
    plt.ylabel('wet-day')
    # plt.xscale('log')

    # 在图上展示相关系数
    plt.title(f'Correlation Coefficient: {correlation_coefficient:.2f}')
    plt.show()


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


def pt(onat_list, th_list, dr_list, bins, sp):
    plt.close()
    fig, axs = plt.subplots(6, figsize=(40, 30), constrained_layout=True)  # 创建6个子图
    fig.tight_layout()
    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=bins.min(), vmax=bins.max())
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])

    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    plt.subplots_adjust(left=0.1, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)
    # clbar.formatter.set_powerlimits((0, 0))  # 设置为不使用科学计数法
    # clbar.update_ticks()  # 更新刻度

    # 使用FixedFormatter手动设置刻度标签
    # clbar.ax.yaxis.set_major_formatter(mticker.FixedFormatter([f'{x:.1f}' for x in cbar.get_ticks()]))
    clbar.set_label('Area (frequency)', fontsize=24)

    # 绘制时间序列和阈值线
    # colors = plt.cm.get_cmap(cmap, len(onat_list))
    for idx, (dr, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')
        if isinstance(dr, xr.DataArray):
            # 如果dr是xarray.DataArray，使用.sel()方法
            df_2011 = dr.sel(time=slice('2011-01-01', '2011-12-31'))
        elif isinstance(dr, np.ndarray):
            # 如果dr是numpy.ndarray，使用数组切片
            df_2011 = dr[0:366]
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
    plt.close()


def get_fft_values(y_values, N, f_s):
    f_values = np.linspace(0.0, f_s, N)
    fft_values_ = fft.fft(y_values, N)
    # power spectrum
    fft_values = np.abs(fft_values_) ** 2
    # 归一化
    fft_values = fft_values / np.sum(fft_values[0:N // 2])

    return f_values[0:N // 2], fft_values[0:N // 2]


def just_spectrum(x):
    fft_poiint_num = 16384
    M = 8180
    # Vh
    ''' 计算频谱 '''
    # f_s = 1 ， 采样频率为 1
    freq, X = get_fft_values(x, fft_poiint_num, 1)
    period = 1 / freq
    period = period[len(period)::-1]
    print('len(period):  ', len(period))
    period = period[0:M]
    inverse_X = X[len(X)::-1]
    inverse_X = inverse_X[0:M]
    return period[0:M] / 365.25, inverse_X[0:M]


def show_all_spectrum(dr_list, bins, sp):
    plt.close()
    fig, axs = plt.subplots(6, figsize=(20, 30), constrained_layout=True)  # 创建6个子图
    fig.tight_layout()
    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=bins.min(), vmax=bins.max())
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])

    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    plt.subplots_adjust(left=0.1, right=0.87)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=0.02)
    # 自定义Formatter，只显示两位小数
    # clbar.formatter.set_powerlimits((0, 0))  # 设置为不使用科学计数法
    # clbar.update_ticks()  # 更新刻度

    # 使用FixedFormatter手动设置刻度标签
    # clbar.ax.yaxis.set_major_formatter(mticker.FixedFormatter([f'{x:.1f}' for x in cbar.get_ticks()]))
    clbar.set_label('Area (frequency)', fontsize=24)
    # plt.savefig(sp+'test1')

    # 遍历 dr_list 并在每个子图上绘制
    for idx, drs in enumerate(dr_list):
        period, inverse_X = just_spectrum(drs)
        # 使用 axs[idx] 对象来绘制子图
        # markerline, ste14mlines, baseline = axs[idx].stem(period, inverse_X, linefmt='-', markerfmt=' ', basefmt=" ")
        print(f'power{np.sum(inverse_X[0:14])}')
        axs[idx].plot(period, inverse_X, color=colors[idx])
        # axs[idx].annotate('o', xy=(period[13], 0), xytext=(period[13], 0),
        #                   arrowprops=dict(facecolor='black', shrink=0.05))

        # axs[idx].plot(np.linspace(1, 10, 100), np.random.rand(100))

        # plt.savefig(sp+'test2')
        # 设置颜色
        # plt.setp(markerline, 'color', colors[idx])
        # plt.setp(stemlines, 'color', colors[idx])
        # plt.setp(baseline, 'color', colors[idx])

        # 设置子图标题和标签
        # axs[idx].set_title(f'Area:{idx}', fontsize=24)
        if idx == 5:
            axs[idx].set_xlabel('Period (year)', fontsize=28)
        axs[idx].set_ylabel('Intensity', fontsize=24)
        axs[idx].tick_params(axis='y', labelsize=10)
        axs[idx].set_xlim(0, 4)
        axs[idx].set_ylim(0, 0.004)
        axs[idx].tick_params(axis='both', labelsize=18)
        # plt.savefig(sp+'test3')
    # 保存并显示图表
    plt.savefig(sp, bbox_inches='tight')
    plt.close()


def show_spectrum(x, sp):
    plt.close()
    fft_poiint_num = 16384
    # M 是 周期谱中 10年对应的点数
    # M = 124 * 8 - 10
    M = 8180
    fig = plt.figure(figsize=(8, 5))
    # Vh
    ax1 = plt.subplot(111)
    ''' 计算频谱 '''
    # f_s = 1 ， 采样频率为 1
    freq, X = get_fft_values(x, fft_poiint_num, 1)
    period = 1 / freq
    period = period[len(period)::-1]
    print('len(period):  ', len(period))
    period = period[0:M]
    inverse_X = X[len(X)::-1]
    inverse_X = inverse_X[0:M]
    plt.stem(period[0:M] / 365.25, inverse_X[0:M], 'b', markerfmt=" ", basefmt="-b")
    ax1.set_title('Fourier power spectrum', fontsize=20)
    ax1.set_xlabel('Period (year)', fontsize=15)
    plt.yticks(size=15)
    plt.grid()
    plt.savefig(sp)
    plt.close()


# 获取一个颜色映射
# cmap = cm.get_cmap('viridis')


# infile = xarray.open_dataset(data_frequency).squeeze() * 100


def draw_area_heap(matrix, name):
    matrix2 = np.flipud(np.roll(matrix, 180, axis=1))
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    lon = np.linspace(-180, 180, 360)
    lat = np.linspace(-90, 90, 120)
    Lon, Lat = np.meshgrid(lon, lat)
    c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap='Reds')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.savefig(".\\temp_fig\\" + str(name) + '.png')
    plt.close()


def draw_area_heap_cover(matrix, cover, name):
    os.makedirs(name[:name.find('area') + 4], exist_ok=True)
    colors = ['Blues', 'green', 'red']
    matrix2 = np.flipud(np.roll(matrix, 180, axis=1))
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    lon = np.linspace(-180, 180, 360)
    lat = np.linspace(-60, 60, 120)
    Lon, Lat = np.meshgrid(lon, lat)
    c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap=colors[0])
    fig.colorbar(c, ax=ax)
    # 计算每个循环迭代应该使用的颜色映射索引
    # 计算每个循环迭代应该使用的颜色映射索引
    # colors = [cmap(i / len(cover)) for i in range(len(cover))]
    # colors = [cmap(i) for i in colors]
    for ind, cv in enumerate(cover):
        cv = np.where(cv == 0, np.nan, cv)  # 添加这一行，将0值替换为np.nan
        cv = np.flipud(np.roll(cv, 180, axis=1))
        c = ax.contourf(Lon, Lat, cv, transform=ccrs.PlateCarree(), colors=colors[ind + 1])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.savefig(name)
    plt.close()


def era5_draw_area_dataArray(dataArray, name):
    print(f'max:{dataArray.max().values}')
    fig = plt.figure(figsize=(10, 10))
    # 创建对数刻度的等级。例如：从10^0到10^3的20个等级
    # levels = np.logspace(0.0001, np.log10(dataArray.max().values), 20)

    # 创建一个LogNorm实例
    # norm = mcolors.Normalize(vmin=0, vmax=120)
    ax = plt.axes(projection=ccrs.PlateCarree())
    # 创建一个LogNorm实例
    norm = mcolors.LogNorm(0.01, vmax=dataArray.max().values)

    # 创建一个等高线图，使用对数刻度
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    c = ax.contourf(dataArray.longitude, dataArray.latitude, dataArray,
                    transform=ccrs.PlateCarree(), cmap='rainbow', norm=norm)

    # 添加颜色条
    plt.colorbar(c, ax=ax, orientation='vertical', label='Log Scaled Values')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    plt.savefig(".\\temp_fig\\all_area\\" + str(name) + '.png')
    plt.close()


def draw_all_era5_area(dataarray):
    # 接着，筛选出2001年的数据
    dataarray_2001 = dataarray.sel(time=dataarray['time'].dt.year == 2001)
    # 现在，遍历每一天，并调用era5_draw_area_dataArray函数
    for day in dataarray_2001['time']:
        # 提取当天的数据
        one_day_data = dataarray_2001.sel(time=day)

        # 构造文件名，例如 "data_20010101"
        name = one_day_data['time'].dt.strftime('data_%Y%m%d').item()
        print(name)

        # 调用函数绘制并保存图像
        era5_draw_area_dataArray(one_day_data, name)


def draw_area_heap_1deg(matrix, name):
    matrix2 = np.flipud(np.roll(matrix, 720, axis=1))
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    lon = np.linspace(-180, 180, 1440)
    lat = np.linspace(-90, 90, 721)
    Lon, Lat = np.meshgrid(lon, lat)
    c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap='rainbow')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.savefig(".\\temp_fig\\" + str(name) + '.png')


def test_3matrix(raw, area_wetday, lag):
    raw[area_wetday] = -999
    raw[lag] = np.nan
    matrix2 = np.flipud(np.roll(matrix, 720, axis=1))
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    lon = np.linspace(-180, 180, 1440)
    lat = np.linspace(-90, 90, 721)
    Lon, Lat = np.meshgrid(lon, lat)
    c = ax.contourf(Lon, Lat, matrix2, transform=ccrs.PlateCarree(), levels=20, cmap='rainbow')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶部的经度标签
    gl.right_labels = False  # 关闭右侧的纬度标签
    plt.savefig(".\\temp_fig\\" + str(name) + '.png')


def test_plot(data_pairs):
    """
    绘制多组数据的函数。每组数据由一对 x 和 y 组成。
    :param data_pairs: 包含多组 (x, y) 数据的列表。
    """
    plt.close()
    temp_mark = ['o', 's', 'o']
    plt.figure(figsize=(10, 6))
    colors = ['yellow', 'black', 'blue']
    # 为每组数据绘制一条线，并标记
    for i, (x, y) in enumerate(data_pairs):
        if i == 0:
            plt.plot(x, y, label=f'Line {i + 1}', marker=temp_mark[i], color=colors[i], linestyle="none", markerfacecolor="none")
        else:
            plt.plot(x, y, label=f'Line {i + 1}', marker=temp_mark[i], color=colors[i], linestyle="none")
    plt.xscale('log')  # 设置 x 轴为对数刻度
    plt.xlabel('X-axis (log scale)')
    plt.ylabel('Y-axis')
    plt.title('Custom Plot with Multiple Data Sets')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.show()


# 定义一个格式化函数
def format_tick(tick_val, pos):
    return "%.2f" % tick_val


def draw_hist_dq_fit2(durations, title, vbins, fig_name):
    plt.close()
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20), constrained_layout=True)
    # 将二维数组转换为一维数组
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=24)

    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=vbins.min(), vmax=vbins.max())
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])
    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)

    clbar.set_label('Area (frequency)', fontsize='16')
    # 设置 bins 的边界为对数刻度
    # plt.subplot(1, 2, 1)
    for area, p_dur in enumerate(durations.transpose((1, 0))):

        for p, dur in enumerate(p_dur):
            bin_centers, hist = dur
            dur_mean = np.sum(bin_centers * hist)
            # 计算二阶矩 (方差)
            variance = np.sum(hist * (bin_centers - dur_mean) ** 2)

            # 计算三阶矩 (偏度)
            skewness = np.sum(hist * (bin_centers - dur_mean) ** 3)
            x_fit_parm = dur_mean / variance
            y_fit_parm = np.square(variance) / skewness
            print(f'xfit:{x_fit_parm} yfit:{y_fit_parm}')

            print(f'mean:{dur_mean}')
            x = bin_centers
            y = hist
            y[y == 0] = np.nan

            # if title == 'Duration':
            mask = (x > 5) & (x < 40) & ~np.isnan(y)
            # else:
            #     mask = (x > 30) & (x < 80) & ~np.isnan(y)
            x_lim = x[mask]
            y_lim = y[mask]
            # 绘制线图表示概率分布
            axs[area].loglog(bin_centers * x_fit_parm, hist * y_fit_parm, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3))
            # axs[area].loglog(bin_centers, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3), label=['40', '30', '20', '10'][p])
            # log_x = np.log(x_lim)
            # log_y = np.log(y_lim)
            # # 选择需要拟合的数据部分，例如x值在4到8之间
            #
            # # 进行直线拟合，polyfit返回拟合系数，这里1代表一次多项式，即直线
            # coefficients = np.polyfit(log_x, log_y, 1)
            # # covs[ind] = coefficients[0]
            #
            # # 使用拟合系数构建拟合直线的y值
            # log_y_fit = np.polyval(coefficients, x_lim)
            #
            # y_fit = np.exp(log_y_fit)
            #
            # # 在双对数坐标下绘制拟合的直线
            # if title == 'Duration':
            #     axs[area].loglog(x_lim / np.exp(dur_mean), y_fit, color=colors[area], label=f'k={coefficients[0]:.2f}')
        # axs[area].set_yscale('log')
        # 设置标题和标签
        axs[area].set_title(f'Area:{area + 1}', fontsize=24)
        axs[area].set_ylabel('Probability', fontsize=24)
        # axs[area].set_xlim(10 ** -1, 50)
        # axs[area].set_ylim(10 ** -8, 1)
        axs[area].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=24)
        axs[area].tick_params(axis='both', labelsize=18)

    plt.savefig(fig_name)
    plt.close()


def draw_hist_dq(durations, title, vbins, fig_name):
    plt.close()
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20), constrained_layout=True)
    # 将二维数组转换为一维数组
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=24)

    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=vbins.min(), vmax=vbins.max())
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])
    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)

    clbar.set_label('Area (frequency)', fontsize='16')
    # 设置 bins 的边界为对数刻度
    # plt.subplot(1, 2, 1)
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
            axs[area].loglog(bin_centers / scale, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3))
            # axs[area].loglog(bin_centers, hist, '*', color=colors[area], alpha=0.1 + p * (0.9 / 3), label=['40', '30', '20', '10'][p])
            # log_x = np.log10(x_lim) / dur_mean
            # log_y = np.log10(y_lim)
            # 选择需要拟合的数据部分，例如x值在4到8之间

            # 进行直线拟合，polyfit返回拟合系数，这里1代表一次多项式，即直线
            # coefficients = np.polyfit(log_x, log_y, 1)
            # covs[ind] = coefficients[0]

            # 使用拟合系数构建拟合直线的y值
            # log_y_fit = np.polyval(coefficients, xlim_scale)

            # y_fit = 10 ** log_y_fit

            # 在双对数坐标下绘制拟合的直线
            # if title == 'Duration':
            # axs[area].loglog(xlim_scale, y_fit, color=colors[area], label=f'k={coefficients[0]:.2f}')
        # axs[area].set_yscale('log')
        # 设置标题和标签
        axs[area].set_title(f'Area:{area + 1}', fontsize=24)
        axs[area].set_ylabel('Probability', fontsize=24)
        # axs[area].set_xlim(10 ** -1, 50)
        # axs[area].set_ylim(10 ** -8, 1)
        axs[area].legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=24)
        axs[area].tick_params(axis='both', labelsize=18)

    plt.savefig(fig_name)
    plt.close()


def plt_duration(durations, title, vbins, fig_name):
    fig = plt.figure(figsize=(25, 10), constrained_layout=True)
    fig.tight_layout()

    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=vbins.min(), vmax=vbins.max())
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])
    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)

    # clbar.formatter.set_powerlimits((0, 0))  # 设置为不使用科学计数法
    # clbar.update_ticks()  # 更新刻度

    # 使用FixedFormatter手动设置刻度标签
    # clbar.ax.yaxis.set_major_formatter(mticker.FixedFormatter([f'{x:.1f}' for x in cbar.get_ticks()]))
    # clbar = fig.colorbar(sm)
    clbar.set_label('Area (frequency)', fontsize='16')
    # 设置 bins 的边界为对数刻度
    avg = np.zeros(6)
    std = np.zeros(6)
    med = np.zeros(6)
    covs = np.zeros(6)
    plt.subplot(1, 2, 1)
    for ind, dur in enumerate(durations):
        bin_centers, hist = get_hist(dur)

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
    # plt.legend(loc='lower left')
    # ax2.legend(loc='upper right')
    # plt.plot(covs, 'g', linewidth=5, label='Average', alpha=0.5)

    plt.savefig(fig_name)
    plt.close()


def get_hist(dur):
    bins = np.unique(np.logspace(np.log10(min(dur)), np.log10(max(dur)), 30).round())
    # 计算直方图数据，density=True 以获取频率
    hist, bin_edges = np.histogram(dur, bins=bins, density=True)
    # 计算 bin 中心点，用于作为 x 轴数据
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, hist


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
