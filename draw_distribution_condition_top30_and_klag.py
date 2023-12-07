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
percentile_all = np.load('era5_percentile_all_40years.npy')
percentile_top30 = np.load('era5_percentile_top30_40years.npy')
percentile_lag = np.load('era5_percentile_klag_40years.npy')
array1_expanded = np.expand_dims(percentile_all, axis=0)
array2_expanded = np.expand_dims(percentile_top30, axis=0)

all_data = np.concatenate((array1_expanded, array2_expanded, percentile_lag), axis=0)

fig = plt.figure(figsize=(50, 20))
fig.suptitle(f'era5_0.25_1990-1999', fontsize=18)

cmap = plt.get_cmap(cmap, 20)
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.55])
# clbar = fig.colorbar(sm)
clbar = fig.colorbar(sm, cax=cbar_ax)
clbar.set_label("Wet-day frequency (%)", fontsize='16')
for ind, per in enumerate(all_data):
    ax = plt.subplot(2, 5, ind + 1)
    for i, d in enumerate(per):
        if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
            None
        else:
            plt.plot(d, np.arange(0, 100, 1), '.', color=colors[i], markersize=10)
    if ind == 0:
        axname = 'all'
    elif ind == 1:
        axname = 'top30'
    else:
        axname = f"{[1, 2, 3, 4, 5, 10, 50, 100][ind - 2]}lag"
    plt.title(f"{axname}-precipitation-distribution", fontsize='25')
    plt.ylabel('Percentile', fontsize=15)
    plt.xlabel(f'Cumulative precipitation (mm/day)', fontsize=15)
    plt.yticks([1, 10, 25, 50, 75, 90, 99])
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.xscale("log")
    plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])  # ,fontsize = 14)
plt.savefig(f'.\\temp_fig\\ear5_percentile_lag_40years.png')
