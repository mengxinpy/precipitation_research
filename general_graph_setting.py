import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr


# 定义一个格式化函数
def format_tick(tick_val, pos):
    return "%.2f" % tick_val


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


def setup_plot(vbins, figsize=(6.4, 4.8), cmap_name='viridis', bins=6, cdict=None):
    if cdict is None:
        cdict = {
            'red': ((0.0, inter_from_256(64), inter_from_256(64)),
                    (1 / 5 * 1, inter_from_256(102), inter_from_256(102)),
                    (1 / 5 * 2, inter_from_256(200), inter_from_256(200)),  # 调整后的值
                    (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
                    (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
                    (1.0, inter_from_256(169), inter_from_256(169))),
            'green': ((0.0, inter_from_256(57), inter_from_256(57)),
                      (1 / 5 * 1, inter_from_256(178), inter_from_256(178)),
                      (1 / 5 * 2, inter_from_256(210), inter_from_256(210)),  # 调整后的值
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
        cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=bins)
    else:
        cmap = clr.LinearSegmentedColormap(cmap_name, segmentdata=cdict, N=bins)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.tight_layout()

    norm = mpl.colors.Normalize(vmin=vbins.min(), vmax=vbins.max())

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.subplots_adjust(left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
    clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)

    clbar.set_label('Area (frequency)', fontsize=16)

    colors = cmap(np.linspace(0, 1, bins))

    return fig, clbar, cmap, colors


def set_font_sizes(ax, fig, scale_factor=0.05):
    """根据图形尺寸调整字体大小"""
    fig_width, fig_height = fig.get_size_inches()
    font_size = min(fig_width, fig_height) * scale_factor * 100  # 100是一个调整因子

    # 设置标题和标签的字体大小
    ax.title.set_size(font_size)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(font_size)

    # 设置Colorbar的字体大小
    if hasattr(ax, 'colorbar'):
        ax.colorbar.ax.yaxis.label.set_size(font_size)
        for label in ax.colorbar.ax.get_yticklabels():
            label.set_fontsize(font_size)

    # 设置第二个Y轴的字体大小
    if hasattr(ax, 'right_ax'):
        ax.right_ax.yaxis.label.set_size(font_size)
        for label in ax.right_ax.get_yticklabels():
            label.set_fontsize(font_size)

    # 递归地处理子轴
    if hasattr(ax, 'child_axes'):
        for child_ax in ax.child_axes:
            set_font_sizes(child_ax, fig, scale_factor)


def adjust_all_font_sizes(fig, scale_factor=0.05):
    """调整所有子图的字体大小"""
    for ax in fig.get_axes():
        set_font_sizes(ax, fig, scale_factor)

# cdict = {
#     'red': ((0.0, inter_from_256(64), inter_from_256(64)),
#             (1 / 5 * 1, inter_from_256(102), inter_from_256(102)),
#             (1 / 5 * 2, inter_from_256(235), inter_from_256(235)),
#             (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
#             (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
#             (1.0, inter_from_256(169), inter_from_256(169))),
#     'green': ((0.0, inter_from_256(57), inter_from_256(57)),
#               (1 / 5 * 1, inter_from_256(178), inter_from_256(178)),
#               (1 / 5 * 2, inter_from_256(240), inter_from_256(240)),
#               (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
#               (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
#               (1 / 5 * 5, inter_from_256(23), inter_from_256(23))),
#     'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
#              (1 / 5 * 1, inter_from_256(255), inter_from_256(255)),
#              (1 / 5 * 2, inter_from_256(185), inter_from_256(185)),
#              (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
#              (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
#              (1.0, inter_from_256(69), inter_from_256(69))),
# }
