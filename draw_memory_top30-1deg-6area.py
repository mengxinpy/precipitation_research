import matplotlib.pyplot as plt
from matplotlib import colors as clr
from scipy.interpolate import interp1d
import matplotlib as mpl
from lag_parameter import log_points, bins, indices, area_top_per_all, selected_columns
from config_calc_power import *
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
cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=6)
colors = cmap(np.linspace(0, 1, 100))
colors2 = cmap(np.linspace(0, 1, 20))
path_out = "E:\\ERA5\\1980-2019\\outer_klag_rain\\"
all_data = np.load(f'{path_out}\\result_klag_1deg_6area_topper.npy').transpose((2, 0, 1, 3))
gap = np.zeros((all_data.shape[0], all_data.shape[1], all_data.shape[2] - 2))
memory = np.zeros((all_data.shape[0], all_data.shape[1], all_data.shape[2] - 2))

fig = plt.figure(figsize=(10, 10))
fig.suptitle(f'era5_1_1979-2019', fontsize=18)
for area_num, area_data in enumerate(all_data):
    for per_num, per_data in enumerate(area_data):
        for lag_num, lag_data in enumerate(per_data):
            top30 = per_data[0]
            al = per_data[1]
            if lag_num >= 2:
                gap_value, x_value, y_value_s2, y_value_s3 = find_gap(top30, lag_data)
                gap_value_all, x_value_all, y_value_s2_all, y_value_s3_all = find_gap(top30, al)
                gap[area_num, per_num, lag_num] = gap_value
                memory[area_num, per_num, lag_num] = 1 - gap_value / gap_value_all
    memory[memory <= 0] = np.nan
    assert all_data.shape[1] == 4
    for i in range(all_data.shape[1]):
        plt.plot(log_points, memory[area_num, i, :], '.', color=colors[i * 17], markersize=10, label=f'top:{99 - selected_columns[i]}')
    plt.title(f"memory")
    plt.ylabel('memory')
    plt.xlabel(f'time')
    plt.xlim(1, 180)
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    # cmap = plt.get_cmap(cmap, 20)
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax)
    clbar.set_label("Wet-day frequency (%)", fontsize='16')
    plt.savefig(f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-1deg-only{area_num}area.png')
    plt.close()
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
# plt.savefig(f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_40years_time-many-3d-1deg-6area.png')
# plt.close()
