import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr
from config_calc_power import *


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
cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=20)
colors = cmap(np.linspace(0, 1, 100))
colors2 = cmap(np.linspace(0, 1, 20))
#
#
path = "F:\\liusch\\remote_project\\climate_new\\temp_data\\"
avg_rain = np.load(f'.\\temp_data\\power{start_year}-{end_year}_{deg}_{vapor_strat}_{vapor_end}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}_avg_rain.npy')
count = np.load(f'.\\temp_data\\power{start_year}-{end_year}_{deg}_{vapor_strat}_{vapor_end}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}_count.npy')
binder = np.load(f'.\\temp_data\\power{start_year}-{end_year}_{deg}_{vapor_strat}_{vapor_end}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}_binder.npy')
var = np.load(f'.\\temp_data\\power{start_year}-{end_year}_{deg}_{vapor_strat}_{vapor_end}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}_var.npy')
# avg_rain = np.load(
#     '.\\temp_data\\' + 'power' + str(start_year) + '-' + str(end_year) + '_' + str(deg) + '_' + str(vapor_bins_count) + '_avg_rain.npy')
# count = np.load(
#     '.\\temp_data\\' + 'power' + str(start_year) + '-' + str(end_year) + '_' + str(deg) + '_' + str(vapor_bins_count) + '_count.npy')
# binder = np.load(
#     '.\\temp_data\\' + 'power' + str(start_year) + '-' + str(end_year) + '_' + str(deg) + '_' + str(vapor_bins_count) + 'binder.npy')
# var = np.load(
#     '.\\temp_data\\' + 'power' + str(start_year) + '-' + str(end_year) + '_' + str(deg) + '_' + str(vapor_bins_count) + 'var.npy')
dist_arr = np.asarray(avg_rain)

fig = plt.figure(figsize=(15, 12))
fig.suptitle(str(start_year) + '-' + str(end_year) + 'year' + '_' + str(deg) + 'deg', fontsize=18)

# colorbar
cmap = plt.get_cmap(cmap, 20)
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# 调整subplot的布局，为colorbar留出空间
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.55])
clbar = fig.colorbar(sm, cax=cbar_ax)
clbar.set_label("Wet-day frequency (%)", fontsize='16')

# 图1
ax = plt.subplot(2, 2, 1)
for ind, d in enumerate(dist_arr[0:20, :]):
    if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
        None
    else:
        plt.plot(np.linspace(vapor_strat, vapor_end, vapor_bins_count), d, '.', color=colors2[ind * int(20 * wetday_gap)], markersize=marker_size)

plt.ylabel('averaged rainrate (mm/day)', fontsize=15)
# plt.ylim(0, 150)
# ax = fig.gca()  # Getting current axes
xlabels = ax.get_xticklabels()  # Getting x-axis labels
plt.setp(xlabels, fontsize=15)  # Setting font size of x-axis labels
ylabels = ax.get_yticklabels()  # Getting x-axis labels
plt.setp(ylabels, fontsize=15)  # Setting font size of x-axis labels

# 图2
ax = plt.subplot(2, 2, 2)
for ind, d in enumerate(binder):
    if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
        None
    else:
        plt.plot(np.linspace(vapor_strat, vapor_end, vapor_bins_count), d, '-', color=colors2[ind * int(20 * wetday_gap)], markersize=marker_size)

plt.ylabel('Binder_4th_order_Cumulant10', fontsize=15)
# ax = fig.gca()  # Getting current axes
xlabels = ax.get_xticklabels()  # Getting x-axis labels
plt.setp(xlabels, fontsize=15)  # Setting font size of x-axis labels
ylabels = ax.get_yticklabels()  # Getting x-axis labels
plt.setp(ylabels, fontsize=15)  # Setting font size of x-axis labels

# 图3
ax = plt.subplot(2, 2, 3)
for ind, d in enumerate(var):
    if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
        None
    else:
        plt.plot(np.linspace(vapor_strat, vapor_end, vapor_bins_count), d, '-', color=colors2[ind * int(20 * wetday_gap)], markersize=marker_size)

plt.ylabel('var', fontsize=15)
xlabels = ax.get_xticklabels()  # Getting x-axis labels
plt.setp(xlabels, fontsize=15)  # Setting font size of x-axis labels
ylabels = ax.get_yticklabels()  # Getting x-axis labels
plt.setp(ylabels, fontsize=15)  # Setting font size of x-axis labels

# 图4
ax = plt.subplot(2, 2, 4)
bottom = np.zeros(count.shape[1])
vapor_bins = np.linspace(vapor_strat, vapor_end, num=vapor_bins_count)  # num是你想要的分组数量
for i, values in enumerate(count):
    plt.bar(vapor_bins, values, bottom=bottom, label=f'Group {i + 1}', color=colors2[i * int(20 * wetday_gap)])
    bottom += values
# plt.yscale('log')
xlabels = ax.get_xticklabels()  # Getting x-axis labels
plt.setp(xlabels, fontsize=15)  # Setting font size of x-axis labels
ylabels = ax.get_yticklabels()  # Getting x-axis labels
plt.setp(ylabels, fontsize=15)  # Setting font size of x-axis labels
plt.ylabel('rain area count', fontsize=15)
plt.xlabel('vapor', fontsize=15)

plt.savefig(f'.\\temp_fig\\power{start_year}-{end_year}_{deg}_{vapor_strat}_{vapor_end}_{vapor_bins_count}_{wetday_gap}_{flat}_{time_gap}.png')
plt.show()
