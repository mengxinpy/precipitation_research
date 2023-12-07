import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr
from config_calc_power import *
from scipy.interpolate import interp1d


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


def find_gap(percentiles_x1, percentiles_x2):
    # 创建一个共同的y轴刻度，假设两组数据的百分位数都在1到100之间
    common_y = np.arange(1, 101)
    # percentiles_x2[percentiles_x2 < 1] = 1.02
    percentiles_x1 = np.insert(percentiles_x1, 0, percentiles_x2[0])
    common_y_x1 = np.insert(common_y, 0, 1)
    # 对两组数据进行插值，这里x值表示百分位数对应的数据，y值表示百分位数
    interp1 = interp1d(np.log10(percentiles_x1), common_y_x1, kind='linear', fill_value='extrapolate')
    interp2 = interp1d(np.log10(percentiles_x2), common_y, kind='linear', fill_value='extrapolate')

    # 创建一个共同的x轴刻度，这里假设数据的范围覆盖了两组百分位数向量的最小值和最大值
    common_x = np.linspace(np.log10(min(np.min(percentiles_x1), np.min(percentiles_x2))),
                           np.log10(max(np.max(percentiles_x1), np.max(percentiles_x2))), num=500)
    # common_x = np.logspace(np.log10(min(np.min(percentiles_x1), np.min(percentiles_x2))),
    #                        np.log10(min(np.max(percentiles_x1), np.max(percentiles_x2))), num=500)

    # 在共同的x轴刻度上得到插值后的y值
    interpolated_y1 = interp1(common_x)
    interpolated_y2 = interp2(common_x)
    interpolated_y2 = np.where(np.isnan(interpolated_y2), interpolated_y1, interpolated_y2)
    # 计算差异的绝对值
    interpolated_y1[interpolated_y1 < 0] = 0
    abs_differences = np.abs(interpolated_y1 - interpolated_y2)

    # 找到最大gap
    max_gap = np.max(abs_differences)
    max_gap_index = np.argmax(abs_differences)
    max_gap_x_value = common_x[max_gap_index]

    print("最大gap为:", max_gap, "在x值为:", max_gap_x_value)
    return max_gap, 10 ** max_gap_x_value, interpolated_y1[max_gap_index], interpolated_y2[max_gap_index]


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
cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=20)
colors = cmap(np.linspace(0, 1, 100))
colors2 = cmap(np.linspace(0, 1, 20))
percentile_all = np.load('era5_percentile_all_40years.npy')
percentile_top30 = np.load('era5_percentile_top30_40years.npy')
percentile_lag = np.load('era5_percentile_klag_40years.npy')
array1_expanded = np.expand_dims(percentile_all, axis=0)
array2_expanded = np.expand_dims(percentile_top30, axis=0)

all_data = np.concatenate((array1_expanded, array2_expanded, percentile_lag), axis=0).transpose((1, 0, 2))

fig = plt.figure(figsize=(10, 10))

fig.suptitle(f'era5_0.25_1990-1999', fontsize=18)

cmap = plt.get_cmap(cmap, 20)
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.55])
clbar = fig.colorbar(sm, cax=cbar_ax)
clbar.set_label("Wet-day frequency (%)", fontsize='16')
bins_v = np.array([45, 50, 55, 60, 65])
plt.close()
for area_num, area_data in enumerate(all_data):
    top30 = area_data[1]
    al = area_data[0]
    for ind, p in enumerate(area_data):
        # if str(type(p)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
        #     None
        # else:
        if ind == 0:
            plt.plot(p[:: 2], np.arange(1, 101)[:: 2], '.', color=colors[area_num], markersize=5)
            # axname = 'all'
        elif ind == 1:
            plt.plot(p[:: 2], np.arange(1, 101)[:: 2], 's', color=colors[area_num], markerfacecolor='none', markeredgecolor=colors[area_num], markersize=5)
            # axname = 'top30'
        else:
            # axname = f"{[1, 2, 3, 4, 5, 10, 50, 100][ind - 2]}lag"
            plt.plot(p[:: 2], np.arange(1, 101)[:: 2], '-', color=colors[area_num], markersize=10, alpha=np.linspace(1, 0.2, 60)[ind - 2])
            print(area_num, '|', ind)
            gap_value, x_value, y_value_s2, y_value_s3 = find_gap(top30, p)
            gap_value_all, x_value_all, y_value_s2_all, y_value_s3_all = find_gap(top30, al)
            # print(y_value_s2,'|',y_value_s3)
            plt.annotate(
                '',
                xy=(x_value, y_value_s2),  # 将标记放在两个y值的中点
                xytext=(x_value, y_value_s3),  # 文本位置稍微上移一点
                arrowprops=dict(arrowstyle='|-|', mutation_scale=2, lw=1),  # 使用双向箭头，中间带横线
                ha='center'
            )
            plt.annotate(
                '',
                xy=(x_value_all, y_value_s2_all),  # 将标记放在两个y值的中点
                xytext=(x_value_all, y_value_s3_all),  # 文本位置稍微上移一点
                arrowprops=dict(arrowstyle='|-|', facecolor='blue', mutation_scale=20, lw=1),  # 使用双向箭头，中间带横线
                ha='center'
            )
            # text_x = x_value + 0.2  # x轴方向上的偏移量
            # text_y = (y_value_s2 + y_value_s3) / 2  # y轴上的中点            axname = f"{[1, 2, 3, 4, 5, 10, 50, 100][ind - 2]}lag"

    plt.title(f"frequency:{area_num / 100}")
    plt.ylabel('Percentile')
    plt.xlabel(f'Cumulative precipitation (mm/day)')
    plt.yticks([1, 10, 25, 50, 75, 90, 99])
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.xscale("log")
    plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])  # ,fontsize = 14)
    # plt.legend()
    plt.savefig(f'.\\temp_fig\\ear5_klag_area_60time\\ear5_percentile_lag_40years_area-{area_num}.png')
    plt.close()
