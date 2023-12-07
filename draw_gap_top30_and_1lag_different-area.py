import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr
from config_calc_power import *
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
                           np.log10(max(np.max(percentiles_x1), np.max(percentiles_x2))), num=500)

    interpolated_y1 = interp1(common_x)
    interpolated_y2 = interp2(common_x)

    interpolated_y1[interpolated_y1 < 0] = 0
    abs_differences = np.abs(interpolated_y1 - interpolated_y2)

    max_gap = np.max(abs_differences)
    max_gap_index = np.argmax(abs_differences)
    max_gap_x_value = common_x[max_gap_index]

    print("最大gap为:", max_gap, "在x值为:", 10 ** max_gap_x_value)
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
result_1lag = np.load('era5_percentile_1lag_top30_10years.npy')
result_0lag = np.load('era5_percentile_0lag_top30_10years.npy')
result_all = np.load('era5_percentile_all_top30_10years.npy')

fig = plt.figure(figsize=(30, 10))

fig.suptitle(f'era5_0.25_1990-1999', fontsize=18)

gap = np.zeros((100, 1))
memory = np.zeros((100, 1))
for n, i in enumerate(range(100)):
    d1 = result_all[i]
    d2 = result_0lag[i]
    d3 = result_1lag[i]
    gap_value, x_value, y_value_s2, y_value_s3 = find_gap(d2, d3)
    gap_value_all, x_value_all, y_value_s2_all, y_value_s3_all = find_gap(d2, d1)
    gap[i] = gap_value
    memory[i] = 1 - gap_value / gap_value_all
plt.plot(np.arange(0, 1, 0.01), memory * 100, '-', label='memory')
plt.title(f"gap")
plt.ylabel('Percentile')
plt.xlabel(f'wetday frequency')
plt.grid(ls="--", color='k', alpha=0.5)
plt.legend()
plt.savefig(f'.\\temp_fig\\ear5_lag_area\\ear5_gap_lag_10years_area-many.png')
plt.close()
