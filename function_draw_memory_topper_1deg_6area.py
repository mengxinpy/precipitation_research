import matplotlib.pyplot as plt
from matplotlib import colors as clr
from scipy.interpolate import interp1d
import matplotlib as mpl
import numpy as np


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


def dm_area_top(bins, log_points, area_top_per_all, selected_columns, dm_in, fig_path, figure_title, figure_title_font, colorbar_title, dec=None):
    plt.close()
    assert len(bins) == 6
    assert area_top_per_all.shape == (2, 6)
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

    all_data = dm_in.transpose((2, 0, 1, 3))  # 输入接口位置--------------------------------------------------------
    assert all_data.shape == (6, 2, 27, 100)
    area_all_num = len(bins)
    font_scaling = 1
    label_scaling = font_scaling * 0.75
    s1 = np.zeros((all_data.shape[0], all_data.shape[1], all_data.shape[2] - 2))
    s2 = np.zeros((all_data.shape[0], all_data.shape[1], all_data.shape[2] - 2))
    memory = np.zeros((all_data.shape[0], all_data.shape[1], all_data.shape[2] - 2))

    fig = plt.figure(figsize=(40, 20))
    fig.suptitle(figure_title, fontsize=figure_title_font)  # 输入接口位置--------------------------------------------------------
    plt.tight_layout()

    # color bar
    cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=area_all_num)
    colors = cmap(np.linspace(0, 1, 100))
    colors2 = cmap(np.linspace(0, 1, 20))
    cmap = plt.get_cmap(cmap, 20)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax)
    clbar.set_label(colorbar_title, fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------

    for area_num, area_data in enumerate(all_data):
        for per_num, per_data in enumerate(area_data):
            al = per_data[0]
            top30 = per_data[1]
            for lag_num, lag_data in enumerate(per_data[2:, :]):
                gap_value, x_value, y_value_s2, y_value_s3 = find_gap(top30, lag_data)
                gap_value_all, x_value_all, y_value_s2_all, y_value_s3_all = find_gap(top30, al)

                s1[area_num, per_num, lag_num] = gap_value
                s2[area_num, per_num, lag_num] = gap_value_all
                memory[area_num, per_num, lag_num] = 1 - gap_value / gap_value_all

        memory[memory <= 0] = np.nan

        ax = plt.subplot(2, 3, area_num + 1)
        for i in range(all_data.shape[1]):
            plt.plot(log_points, memory[area_num, i, :], '.', color=colors[area_num * round(100 / area_all_num)], markersize=15, label=f'top:{99 - selected_columns[i]}',
                     alpha=1 - i / all_data.shape[1])
        plt.title(f"{colorbar_title}:{bins[area_num]:.2f}-", fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        plt.ylabel('memory', fontsize=figure_title_font * font_scaling)
        plt.xlabel(f'time', fontsize=figure_title_font * font_scaling)
        plt.xlim(1, 180)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks([1, 10, 100], labels=[1, 10, 100], fontsize=figure_title_font * label_scaling)
        plt.yticks([1, 0.1, 0.01, 0.001], labels=[1, 0.1, 0.01, 0.001], fontsize=figure_title_font * label_scaling)
        plt.grid(ls="--", color='k', alpha=0.5)
        plt.legend(fontsize=figure_title_font * font_scaling, loc='lower left')

    plt.savefig(fig_path + 'tmp.png')  # 输出接口位置^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    plt.close()

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(figure_title, fontsize=figure_title_font)  # 输入接口位置--------------------------------------------------------
    plt.tight_layout()

    memory_per = memory.transpose((1, 2, 0))

    for per_num, per_data in enumerate(memory_per):
        ax = plt.subplot(2, 2, per_num + 1)
        for lag_num, lag_data in enumerate(per_data[::3, :]):
            plt.plot(bins, lag_data, '-', markersize=15, label=f'lag:{log_points[lag_num * 3]}')  # alpha=1 - lag_num / per_data[:: 3, :].shape[0]

        plt.title(f"top:{99 - selected_columns[per_num]}%", fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        plt.ylabel('memory', fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        plt.xlabel(colorbar_title, fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        # 获取当前的x轴刻度
        current_ticks = plt.xticks()[0]
        # 设置新的刻度标签和字体大小
        plt.xticks(current_ticks, fontsize=figure_title_font * label_scaling)
        plt.yticks([1, 0.1, 0.01, 0.001], labels=[1, 0.1, 0.01, 0.001], fontsize=figure_title_font * label_scaling)
        plt.yscale('log')
        plt.ylim(0.001, 1)
        if dec == 'cover time':
            plt.xlim(0, 0.5)
        plt.grid(ls="--", color='k', alpha=0.5)
        plt.legend()

    plt.savefig(fig_path + 'fmt.png')  # 输出接口位置^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    plt.close()

    fig = plt.figure(figsize=(20, 20))
    plt.tight_layout()
    s1 = s1.transpose((1, 2, 0))
    s2 = s2.transpose((1, 2, 0))
    ds12 = s2 - s1
    ds12[ds12 <= 0] = np.nan
    for per_num, per_data in enumerate(ds12):
        ax = plt.subplot(2, 2, per_num + 1)
        for lag_num, lag_data in enumerate(per_data[::3, :]):
            plt.plot(bins, lag_data, '-', markersize=15, label=f'lag:{log_points[lag_num * 3]}')  # alpha=1 - lag_num / per_data[:: 3, :].shape[0]

        plt.title(f"top:{99 - selected_columns[per_num]}%", fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        plt.ylabel('s2-s1', fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        plt.xlabel(colorbar_title, fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        # 获取当前的x轴刻度
        current_ticks = plt.xticks()[0]
        plt.yscale('log')
        # 设置新的刻度标签和字体大小
        plt.xticks(current_ticks, fontsize=figure_title_font * label_scaling)
        plt.grid(ls="--", color='k', alpha=0.5)

    for per_num, per_data in enumerate(s2):
        ax = plt.subplot(2, 2, per_num + 3)
        for lag_num, lag_data in enumerate(per_data[::3, :]):
            plt.plot(bins, lag_data, '-', markersize=15, label=f'lag:{log_points[lag_num * 3]}')  # alpha=1 - lag_num / per_data[:: 3, :].shape[0]

        plt.title(f"top:{99 - selected_columns[per_num]}%", fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        plt.ylabel('s2', fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        plt.xlabel(colorbar_title, fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
        # 获取当前的x轴刻度
        current_ticks = plt.xticks()[0]
        # 设置新的刻度标签和字体大小
        plt.xticks(current_ticks, fontsize=figure_title_font * label_scaling)
        plt.ylim(1, 100)
        plt.legend()
    plt.savefig(fig_path + 's1s2.png')  # 输出接口位置^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    plt.close()
