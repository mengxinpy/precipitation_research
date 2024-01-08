import matplotlib.pyplot as plt
from matplotlib import colors as clr
from scipy.interpolate import interp1d
import matplotlib as mpl
from lag_parameter import log_points
from lag_indirect_parameter import bins, selected_columns, dm_in, sp_dm_frequency, sp_dm_lagtime
from lag_indirect_parameter import figure_title, figure_title_font, colorbar_title
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

all_data = np.load(dm_in).transpose((2, 0, 1, 3))  # 输入接口位置--------------------------------------------------------
area_all_num = len(bins)
assert area_all_num == 6
plt.close()
cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=area_all_num)
colors = cmap(np.linspace(0, 1, 100))
colors2 = cmap(np.linspace(0, 1, 20))

gap = np.zeros((all_data.shape[0], all_data.shape[1], all_data.shape[2] - 2))
memory = np.zeros((all_data.shape[0], all_data.shape[1], all_data.shape[2] - 2))

fig = plt.figure(figsize=(40, 20))
fig.suptitle(figure_title, fontsize=figure_title_font)  # 输入接口位置--------------------------------------------------------
plt.tight_layout()

cmap = plt.get_cmap(cmap, 20)
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.55])
clbar = fig.colorbar(sm, cax=cbar_ax)

font_scaling = 0.75
clbar.set_label(colorbar_title, fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------

assert all_data.shape == (6, 4, 27, 100)
for area_num, area_data in enumerate(all_data):
    for per_num, per_data in enumerate(area_data):
        al = per_data[0]
        top30 = per_data[1]
        for lag_num, lag_data in enumerate(per_data[2:, :]):
            gap_value, x_value, y_value_s2, y_value_s3 = find_gap(top30, lag_data)
            gap_value_all, x_value_all, y_value_s2_all, y_value_s3_all = find_gap(top30, al)

            gap[area_num, per_num, lag_num] = gap_value
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
    plt.xticks([1, 10, 100], labels=[1, 10, 100], fontsize=figure_title_font * font_scaling)
    plt.yticks([1, 0.1, 0.01, 0.001], labels=[1, 0.1, 0.01, 0.001], fontsize=figure_title_font * font_scaling)
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.legend(fontsize=figure_title_font * font_scaling, loc='lower left')

plt.savefig(sp_dm_lagtime)  # 输出接口位置^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
plt.close()

fig = plt.figure(figsize=(20, 20))
fig.suptitle(figure_title, fontsize=figure_title_font)  # 输入接口位置--------------------------------------------------------
plt.tight_layout()

memory_per = memory.transpose((1, 2, 0))

for per_num, per_data in enumerate(memory_per):
    ax = plt.subplot(2, 2, per_num + 1)
    for lag_num, lag_data in enumerate(per_data[::3, :]):
        plt.plot(bins, lag_data, '-', markersize=15, label=f'lag:{log_points[lag_num * 3]}')  # alpha=1 - lag_num / per_data[:: 3, :].shape[0]

    plt.title(f"top:{99 - selected_columns[per_num]}percentile", fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
    plt.ylabel('memory', fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
    plt.xlabel(colorbar_title, fontsize=figure_title_font * font_scaling)  # 输入接口位置--------------------------------------------------------
    plt.yscale('log')
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.legend()

plt.savefig(sp_dm_frequency)  # 输出接口位置^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
plt.close()
# 
# data = memory
# time = np.arange(data.shape[0])
# area_num = np.arange(data.shape[1])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# for i, area in enumerate(area_num):
#     ax.plot(log_points, [area] * len(time), data[:, i], color=colors[i * 17], label=f'Area {i + 1}')
# 
# ax.set_xlabel('Time')
# ax.set_ylabel('frequency')
# ax.set_zlabel('Memory')
# ax.set_title('3D Line Plot for Each Area Over Time')
# 
# plt.savefig(f'.\\temp_fig\\ear5_lag_area\\3dtemp.png')
# plt.close()
