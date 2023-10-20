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

dist = np.load(path + 'area_rainfall_power_40years.npy')
dist_arr = np.asarray(dist)

fig = plt.figure(figsize=(15, 12))

colors2 = cmap(np.linspace(0, 1, 20))
for ind, d in enumerate(dist_arr[0:20, :]):
    if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
        None
    else:
        plt.plot(np.linspace(20, 90, 210), d, '.', color=colors2[ind * 2], markersize=15)

plt.xlabel('vapor', fontsize=15)
plt.ylabel('averaged rainrate (mm/hr)', fontsize=15)
# plt.grid(ls="--", color='k', alpha=0.5)
plt.xlim(50, 90)

ax = fig.gca()  # Getting current axes
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
# plt.show()
plt.savefig(path + 'temp_fig\\rain_era5_power40years.png')
