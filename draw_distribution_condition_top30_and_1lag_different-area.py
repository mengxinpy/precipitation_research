import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr
from config_calc_power import *
from scipy.interpolate import interp1d


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


def find_gap(x1, x2, y1, y2):
    # 创建一个共同的x轴刻度，这里假设两组数据的x值都在1到500之间
    # x2[x2 < 1] = 1.00
    # 设定起始值和结束值的对数（以10为底）
    # 平移x1和x2数据，使所有值大于1
    # 定义一个共同的x轴刻度，覆盖两个数据集的x轴范围
    common_x = np.linspace(min(x1.min(), x2.min()), max(x1.max(), x2.max()), num_points)

    # 对两个曲线进行插值
    interp1 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value="extrapolate")

    # 在共同的x轴刻度上得到插值后的y值
    interpolated_y1 = interp1(common_x)
    interpolated_y2 = interp2(common_x)

    # 计算差异的绝对值
    interpolated_y_diff = np.abs(interpolated_y1 - interpolated_y2)

    # 找到最大gap
    max_interpolated_diff = np.max(interpolated_y_diff)
    max_interpolated_diff_index = np.argmax(interpolated_y_diff)
    max_interpolated_diff_x = common_x[max_interpolated_diff_index]

    max_interpolated_diff, max_interpolated_diff_x

    print("最大gap为:", max_gap, "在x值为:", max_gap_x_value)
    return max_gap, max_gap_x_value, interpolated_y1[max_gap_index], interpolated_y2[max_gap_index]


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
# 加载保存的数组
result_1lag = np.load('era5_percentile_1lag_top30_10years.npy')
result_0lag = np.load('era5_percentile_0lag_top30_10years.npy')
result_all = np.load('era5_percentile_all_top30_10years.npy')
# data = np.load('era5_percentile_area_vapor_single_cul.npy')

# fig, ax = plt.subplots(2, 5, figsize=(10, 30))
fig = plt.figure(figsize=(30, 10))

fig.suptitle(f'era5_0.25_1990-1999', fontsize=18)
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
# ax = plt.subplot(1, 3, 1)
for n, i in enumerate([0, 20, 40, 60, 80, 90, 92, 94, 98, 99]):
    # ax = plt.subplot(2, 3, v + 1)
    # for i, d in enumerate(d_all):
    ax = plt.subplot(2, 5, n + 1)
    d1 = result_all[i]
    d2 = result_0lag[i]
    d3 = result_1lag[i]
    plt.plot(d1[:: 2], np.arange(1, 101)[:: 2], '.', color=colors[i], markersize=10, label='all precipitation distribution')
    line2 = plt.plot(d3[:: 2], np.arange(1, 101)[:: 2], '+', color=colors[i], markersize=10, label='top 30 precipitation with 1lag distribution')
    line1 = plt.plot(d2[::3], np.arange(1, 101)[::3], 's', markerfacecolor='none', markeredgecolor=colors[i], markersize=5, label='top 30 precipitation distribution')
    # 从曲线对象中提取数据点坐标
    x1, y1 = line1.get_data()
    x2, y2 = line2.get_data()
    gap_value, x_value, y_value_s2, y_value_s3 = find_gap(x1, y1, x2, y2)
    # 标记最大gap
    plt.annotate(
        # f'{gap_value / 100:.2f}',  # 保留两位小数
        '',
        xy=(x_value, y_value_s2),  # 将标记放在两个y值的中点
        xytext=(x_value, y_value_s3),  # 文本位置稍微上移一点
        arrowprops=dict(arrowstyle='|-|', mutation_scale=5, lw=1.5, connectionstyle='bar,fraction=1,angle=0'),  # 使用双向箭头，中间带横线
        ha='center'
    )
# 在箭头旁边添加文本标注
text_x = x_value + 0.2  # x轴方向上的偏移量
text_y = (y_value_s2 + y_value_s3) / 2  # y轴上的中点
plt.text(text_x, text_y, f'{gap_value / 100:.2f}', ha='center', va='center')
plt.title(f"frequency:{i / 100}")
plt.ylabel('Percentile')
plt.xlabel(f'Cumulative precipitation (mm/day)')
plt.yticks([1, 10, 25, 50, 75, 90, 99])
plt.grid(ls="--", color='k', alpha=0.5)
plt.xscale("log")
# plt.xlim(1, 500)
plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])  # ,fontsize = 14)
# 创建共享图例
# 创建共享图例
lines, labels = plt.gca().get_legend_handles_labels()
fig = plt.gcf()  # 获取当前figure
fig.legend(lines, labels, loc='lower center', ncol=3, fontsize='large', handlelength=2, handleheight=2, labelspacing=0.5, borderpad=1)
plt.savefig(f'.\\temp_fig\\ear5_lag_area\\ear5_percentile_lag_10years_area-many.png')
plt.close()
# plt.show()
# plt.close(fig)
# for i in range(result_all.shape[0]):
#     # ax = plt.subplot(2, 3, v + 1)
#     # for i, d in enumerate(d_all):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     d1 = result_all[i]
#     d2 = result_0lag[i]
#     d3 = result_1lag[i]
#     plt.plot(d1, np.arange(0, 100, 1), '.', color=colors[i], markersize=10, alpha=1)
#     plt.plot(d2, np.arange(0, 100, 1), '.', color=colors[i], markersize=10, alpha=0.6)
#     plt.plot(d3, np.arange(0, 100, 1), '.', color=colors[i], markersize=10, alpha=0.3)
#     plt.title(f"all-precipitation-distribution", fontsize='25')
#     plt.ylabel('Percentile', fontsize=15)
#     plt.xlabel(f'Cumulative precipitation (mm/day)', fontsize=15)
#     plt.yticks([1, 10, 25, 50, 75, 90, 99])
#     plt.grid(ls="--", color='k', alpha=0.5)
#     plt.xscale("log")
#     plt.xlim(1, 500)
#     plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])  # ,fontsize = 14)
#     plt.savefig(f'.\\temp_fig\\ear5_lag_area\\ear5_percentile_lag_10years_area{i}.png')
#     plt.close(fig)
# plt.savefig(f'.\\temp_fig\\cul_vapor_conditioned_cul.png')
# plt.show()
