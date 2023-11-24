import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
from matplotlib import colors as clr
import matplotlib.transforms as mtransforms
from cartopy.util import add_cyclic_point
import matplotlib as mpl


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
path = "F:\\liusch\\remote_project\\climate_new\\"
dist = np.load(path + 'power_10years80_1mm_0.5.npy')
dist_arr = np.asarray(dist)
fig = plt.figure(figsize=(15, 12))
ax = plt.subplot(2, 2, 1)
colors2 = cmap(np.linspace(0, 1, 20))
for ind, d in enumerate(dist_arr[0:20, :]):
    if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
        None
    else:
        plt.plot(np.linspace(50, 80, 150), d , '.', color=colors2[ind * 2], markersize=15)

plt.ylabel('averaged rainrate (mm/hr)', fontsize=15)
# plt.grid(ls="--", color='k', alpha=0.5)
# plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16] )
# plt.yticks([i * 48 for i in range(10)])
# plt.xlim(50, 80)
plt.ylim(0, 150)

# ax = fig.gca()  # Getting current axes
xlabels = ax.get_xticklabels()  # Getting x-axis labels
plt.setp(xlabels, fontsize=15)  # Setting font size of x-axis labels
ylabels = ax.get_yticklabels()  # Getting x-axis labels
plt.setp(ylabels, fontsize=15)  # Setting font size of x-axis labels
cmap = plt.get_cmap(cmap, 20)
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# 调整subplot的布局，为colorbar留出空间
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.55])
clbar = fig.colorbar(sm, cax=cbar_ax)
clbar.set_label("Wet-day frequency (%)", fontsize='16')
count = np.load('count_10years80_1mm_0.5.npy')
ax = plt.subplot(2, 2, 3)
bottom = np.zeros(count.shape[1])
vapor_bins_count = 150
vapor_bins = np.linspace(50, 80, num=vapor_bins_count)  # num是你想要的分组数量
for i, values in enumerate(count):
    plt.bar(vapor_bins, values, bottom=bottom, label=f'Group {i + 1}', color=colors2[i * 2])
    bottom += values
# plt.yscale('log')
xlabels = ax.get_xticklabels()  # Getting x-axis labels
plt.setp(xlabels, fontsize=15)  # Setting font size of x-axis labels
ylabels = ax.get_yticklabels()  # Getting x-axis labels
plt.setp(ylabels, fontsize=15)  # Setting font size of x-axis labels
plt.ylabel('rain area count', fontsize=15)
plt.xlabel('vapor', fontsize=15)
# plt.xlim(50, 90)
binder = np.load('Binder_4th_order_Cumulant_10years80_0.5.npy')
ax = plt.subplot(2, 2, 2)
for ind, d in enumerate(binder):
    if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
        None
    else:
        plt.plot(np.linspace(50, 80, 150), d, '.', color=colors2[ind * 2], markersize=15)

plt.ylabel('Binder_4th_order_Cumulant', fontsize=15)
# plt.grid(ls="--", color='k', alpha=0.5)
# plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
# plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
# plt.xlim(50, 90)

# ax = fig.gca()  # Getting current axes
xlabels = ax.get_xticklabels()  # Getting x-axis labels
plt.setp(xlabels, fontsize=15)  # Setting font size of x-axis labels
ylabels = ax.get_yticklabels()  # Getting x-axis labels
plt.setp(ylabels, fontsize=15)  # Setting font size of x-axis labels
# plt.legend(loc='upper right')
plt.show()
# plt.savefig(path + 'temp_fig\\rain_era5_power40years.png')
