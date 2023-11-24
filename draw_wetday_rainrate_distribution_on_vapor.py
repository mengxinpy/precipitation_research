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
data = np.load('era5_percentile_10area_vapor.npy')
# data = np.load('era5_percentile_area_vapor_single_cul.npy')

fig = plt.figure(figsize=(50, 20))
fig.suptitle(f'era5_0.25_vapor_conditioned', fontsize=25)
# fig.suptitle(f'era5_0.25_vapor_conditioned_cul', fontsize=18)

cmap = plt.get_cmap(cmap, 20)
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.55])
clbar = fig.colorbar(sm, cax=cbar_ax)
clbar.set_label("Wet-day frequency (%)", fontsize='16')
bins_v = np.array([45, 50, 55, 60, 65])
for a, d_all in enumerate(data):
    ax = plt.subplot(2, 5, a + 1)
    for i, d in enumerate(d_all):
        if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
            None
        else:
            plt.plot(d, np.arange(0, 100, 1), '.', color=colors2[a * 2], markersize=10, alpha=0.2 + 0.16 * i)

    plt.title(f"wetday-frequency:{a / 10}", fontsize='25')
    plt.ylabel('Percentile', fontsize=22)
    plt.xlabel(f'Cumulative precipitation', fontsize=22)
    plt.yticks([1, 10, 25, 50, 75, 90, 99], fontsize=18)
    plt.grid(ls="--", color='k', alpha=0.5)
    plt.xscale("log")
    plt.xlim(1, 500)
    plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500], fontsize=18)  # ,fontsize = 14)
plt.savefig(f'.\\temp_fig\\precipitation-distribution-on-vapor.png')
# plt.savefig(f'.\\temp_fig\\cul_vapor_conditioned_cul.png')
plt.show()
