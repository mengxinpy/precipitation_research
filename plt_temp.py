import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr
import numpy.fft as fft


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


def pt(onat_list, th_list, dr_list, sp):
    fig, ax = plt.subplots(figsize=(30, 5), constrained_layout=True)
    fig.tight_layout()

    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])
    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)
    # 绘制时间序列和阈值线
    # colors = plt.cm.get_cmap(cmap, len(onat_list))
    for idx, (dr, th, onat) in enumerate(zip(dr_list, th_list, onat_list)):
        print(f'th:{th}')
        df_2011 = dr.sel(time=slice('2011-01-01', '2011-12-31'))
        ax.plot(df_2011, label=f'ONAT {onat}', color=colors[idx])
        ax.axhline(y=th, color=colors[idx], linestyle='--')  # 绘制阈值线kk
    # 设置标题和标签
    ax.set_title('Time Series with Thresholds')
    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.xlim(0,365)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # 保存图表
    plt.savefig(sp, bbox_inches='tight')
    plt.show()


def get_fft_values(y_values, N, f_s):
    f_values = np.linspace(0.0, f_s, N)
    fft_values_ = fft.fft(y_values, N)
    # power spectrum
    fft_values = np.abs(fft_values_) ** 2
    # 归一化
    fft_values = fft_values / np.sum(fft_values[0:N // 2])

    return f_values[0:N // 2], fft_values[0:N // 2]


def just_spectrum(x):
    plt.close()
    fft_poiint_num = 16384
    fig = plt.figure(figsize=(8, 5))
    M = 8180
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
    return period[0:M] / 365.25, inverse_X[0:M]


def show_all_spectrum(dr_list, sp):
    fig = plt.figure(figsize=(20, 40), constrained_layout=True)
    # fig.tight_layout()

    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])
    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    # plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)
    # clbar = fig.colorbar(sm)
    clbar.set_label('Area (frequency)', fontsize='16')
    for ind, drs in enumerate(dr_list):
        period, inverse_X = just_spectrum(drs)
        ax = plt.subplot(2, 3, ind + 1)
        # plt.stem(period, inverse_X, linefmt='-', markerfmt='o', basefmt=" ", use_line_collection=True, linecolor=colors[ind], markeredgecolor=colors[ind], markerfacecolor=colors[ind])
        markerline, stemlines, baseline = plt.stem(period, inverse_X, linefmt='-', markerfmt=' ', basefmt=" ")

        # 设置颜色
        plt.setp(markerline, 'color', colors[ind])
        plt.setp(stemlines, 'color', colors[ind])
        plt.setp(baseline, 'color', colors[ind])
        ax.set_title('Fourier power spectrum', fontsize=15)
        ax.set_xlabel('Period (year)', fontsize=10)
        plt.yticks(size=10)
        plt.grid()
    plt.savefig(sp)
    plt.show()
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
    plt.savefig(".\\temp_fig\\fig\\" + str(name) + '.png')
    plt.close()


def era5_draw_area_dataArray(dataArray, name):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()  # 你的数据应该是一个二维数组，你需要创建一个网格来表示地理坐标
    c = ax.contourf(dataArray.longitude, dataArray.latitude, dataArray, transform=ccrs.PlateCarree(), levels=20, cmap='rainbow')
    fig.colorbar(c, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    plt.savefig(".\\temp_fig\\" + str(name) + '.png')


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


def plt_duration(durations, fig_name):
    fig = plt.figure(figsize=(25, 10), constrained_layout=True)
    fig.tight_layout()

    # 显示图表
    # cmap =
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cmap2 = plt.get_cmap(cmap, 6)
    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])
    # cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)
    # clbar = fig.colorbar(sm)
    clbar.set_label('Area (frequency)', fontsize='16')
    # 设置 bins 的边界为对数刻度
    avg = np.zeros(6)
    std = np.zeros(6)
    med = np.zeros(6)
    covs = np.zeros(6)
    plt.subplot(1, 2, 1)
    for ind, dur in enumerate(durations):
        bins = np.unique(np.logspace(np.log10(min(dur)), np.log10(max(dur)), 30).round())

        # 计算直方图数据，density=True 以获取频率
        hist, bin_edges = np.histogram(dur, bins=bins, density=True)
        avg[ind] = np.mean(dur)
        med[ind] = np.median(dur)
        std[ind] = np.std(dur)

        # 计算 bin 中心点，用于作为 x 轴数据
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

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
    plt.title('Probability Distribution of Durations', fontsize=24)
    plt.xlabel('Duration(day)', fontsize=24)
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
    line1 = plt.plot(std, 'r*-', linewidth=2, markersize=8, label='Standard Deviation', alpha=0.5)[0]
    line2 = plt.plot(med, 'b*-', linewidth=2, markersize=8, label='Median', alpha=0.5)[0]
    line3 = plt.plot(avg, 'g*-', linewidth=2, markersize=8, label='Average', alpha=0.5)[0]
    plt.xlabel('Frequency', fontsize=24)
    plt.ylabel('Value', fontsize=24)
    current_xticks = plt.xticks()[0]
    # 设置新的刻度标签和字体大小
    plt.xticks(current_xticks, fontsize=18)
    current_yticks = plt.yticks()[0]
    # 设置新的刻度标签和字体大小
    plt.yticks(current_yticks, fontsize=18)
    # plt.legend(fontsize=18)
    # 创建一个共享x轴的新轴对象
    ax2 = plt.twinx()
    line4 = ax2.plot(covs, color='black', marker='*', linestyle='--', linewidth=2, markersize=8, label='k')[0]  # 使用黄色来区分

    # 设置y轴的标签
    ax2.set_ylabel('Coefficient', fontsize=24)
    # 设置第二个 y 轴的刻度标签的字体大小
    ax_current_yticks = ax2.get_yticks()
    ax2.set_yticklabels(ax_current_yticks, fontsize=18)
    ax2.ylim(1, 1e-2)

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
