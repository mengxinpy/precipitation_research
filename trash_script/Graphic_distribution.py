import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr
from scipy.interpolate import interp1d


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


def find_gap(percentiles_x1, percentiles_x2):
    common_y = np.arange(1, 101)
    percentiles_x1 = np.insert(percentiles_x1, 0, percentiles_x2[0])
    common_y_x1 = np.insert(common_y, 0, 1)
    interp1 = interp1d(np.log10(percentiles_x1), common_y_x1, kind='linear', fill_value='extrapolate')
    interp2 = interp1d(np.log10(percentiles_x2), common_y, kind='linear', fill_value='extrapolate')

    common_x = np.linspace(np.log10(min(np.min(percentiles_x1), np.min(percentiles_x2))),
                           np.log10(min(np.max(percentiles_x1), np.max(percentiles_x2))), num=500)

    interpolated_y1 = interp1(common_x)
    interpolated_y2 = interp2(common_x)
    interpolated_y2 = np.where(np.isnan(interpolated_y2), interpolated_y1, interpolated_y2)
    interpolated_y1[interpolated_y1 < 0] = 0
    abs_differences = np.abs(interpolated_y1 - interpolated_y2)

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


def draw_distribution(bins, log_points, ltp, var, fig_path, sample=5):
    cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=len(bins))
    colors = cmap(np.linspace(0, 1, 100))
    fig = plt.figure(figsize=(10, 10))

    bins_v = np.array([45, 50, 55, 60, 65])

    assert ltp.shape == (4, 6, 27, 100)
    for t, top in enumerate(ltp):
        top_path = fig_path + f'distribution\\top{[40,30, 20,10][t]}\\'
        os.makedirs(top_path, exist_ok=True)
        for area_num, area_data in enumerate(top):
            top30 = area_data[1]
            al = area_data[0]
            for ind, p in enumerate(area_data):
                if ind == 0:
                    plt.plot(al[:: 2], np.arange(1, 101)[:: 2], '.', color=colors[area_num * 17], markersize=5)  # todo:注意绘图area参数

                elif ind == 1:
                    plt.plot(top30[:: 2], np.arange(1, 101)[:: 2], 's', color=colors[area_num * 17], markerfacecolor='none', markeredgecolor=colors[area_num * 17], markersize=5)
                    gap_value_all, x_value_all, y_value_s2_all, y_value_s3_all = find_gap(top30, al)
                    plt.annotate(
                        '',
                        xy=(x_value_all, y_value_s2_all),  # 将标记放在两个y值的中点
                        xytext=(x_value_all, y_value_s3_all),  # 文本位置稍微上移一点
                        arrowprops=dict(arrowstyle='|-|', mutation_scale=5, lw=1),  # 使用双向箭头，中间带横线
                        ha='center'
                    )
                else:
                    if (ind - 2) % sample == 0:
                        plt.plot(p[:: 2], np.arange(1, 101)[:: 2], '-', color=colors[area_num * 17], markersize=10, alpha=1 - ind * (1 / 2) / log_points.shape[0],
                                 label=f'lag_day:{log_points[ind]}')
                        print(area_num, '|', ind)
                        gap_value, x_value, y_value_s2, y_value_s3 = find_gap(top30, p)
                        plt.annotate(
                            '',
                            xy=(x_value, y_value_s2),  # 将标记放在两个y值的中点
                            xytext=(x_value, y_value_s3),  # 文本位置稍微上移一点
                            arrowprops=dict(arrowstyle='|-|', mutation_scale=2, lw=1),  # 使用双向箭头，中间带横线
                            ha='center'
                        )

            plt.title(f"Frequency:{round(bins[area_num] * 100)}", fontsize=20)
            plt.ylabel('Percentile')
            plt.xlabel(f'Cumulative precipitation (mm/day)')
            plt.yticks([1, 10, 25, 50, 75, 90, 99])
            plt.grid(ls="--", color='k', alpha=0.5)
            plt.xscale("log")
            plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])  # ,fontsize = 14)
            plt.legend()
            plt.savefig(f'{top_path}{area_num}.png')



def draw_distribution_test(bins, log_points, ltp, var, toparea_percentile):
    cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=len(bins))
    colors = cmap(np.linspace(0, 1, 100))
    fig = plt.figure(figsize=(10, 10))

    bins_v = np.array([45, 50, 55, 60, 65])

    assert ltp.shape == (6, 27, 100)
    for area_num, area_data in enumerate(ltp):
        top30 = area_data[1]
        al = area_data[0]
        for ind, p in enumerate(area_data):
            if ind == 0:
                plt.plot(al[:: 2], np.arange(1, 101)[:: 2], '.', color=colors[area_num * 17], markersize=5)  # todo:注意绘图area参数

            elif ind == 1:
                plt.plot(top30[:: 2], np.arange(1, 101)[:: 2], 's', color=colors[area_num * 17], markerfacecolor='none', markeredgecolor=colors[area_num * 17], markersize=5)
                gap_value_all, x_value_all, y_value_s2_all, y_value_s3_all = find_gap(top30, al)
                plt.annotate(
                    '',
                    xy=(x_value_all, y_value_s2_all),  # 将标记放在两个y值的中点
                    xytext=(x_value_all, y_value_s3_all),  # 文本位置稍微上移一点
                    arrowprops=dict(arrowstyle='|-|', mutation_scale=5, lw=1),  # 使用双向箭头，中间带横线
                    ha='center'
                )
            else:
                if ind % 4 == 0:
                    plt.plot(p[:: 2], np.arange(1, 101)[:: 2], '-', color=colors[area_num * 17], markersize=10, alpha=1 - ind * (1 / 2) / log_points.shape[0],
                             label=f'lag_day:{log_points[ind]}')
                    print(area_num, '|', ind)
                    gap_value, x_value, y_value_s2, y_value_s3 = find_gap(top30, p)
                    plt.annotate(
                        '',
                        xy=(x_value, y_value_s2),  # 将标记放在两个y值的中点
                        xytext=(x_value, y_value_s3),  # 文本位置稍微上移一点
                        arrowprops=dict(arrowstyle='|-|', mutation_scale=2, lw=1),  # 使用双向箭头，中间带横线
                        ha='center'
                    )
        if area_num == 1:
            plt.plot(toparea_percentile[:: 2], np.arange(1, 101)[:: 2], '+', color=colors[area_num * 17], markersize=10)  # todo:注意绘图area参数

        plt.title(f"Frequency:{round(bins[area_num] * 100)}", fontsize=24)
        plt.ylabel('Percentile')
        plt.xlabel(f'Cumulative precipitation (mm/day)')
        plt.yticks([1, 10, 25, 50, 75, 90, 99])
        plt.grid(ls="--", color='k', alpha=0.5)
        plt.xscale("log")
        plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])  # ,fontsize = 14)
        plt.legend()
        # 构建文件夹路径
        folder_path = f'./temp_fig/distribution/{var}'
        # 使用 os.makedirs 创建文件夹，设置 exist_ok=True 来忽略已存在的文件夹
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(f'{folder_path}/{area_num}.png')
        plt.close()
