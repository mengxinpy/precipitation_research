from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as clr
from matplotlib.colors import LinearSegmentedColormap

# 重置 rcParams 到默认值
plt.rcParams['font.family'] = 'Helvetica'

plt.rcParams['figure.dpi'] = 100


def set_fonts_for_fig(fig,
                      scale_factor=5,
                      legend_scale_factor=0.5,
                      label_scale_factor=1,
                      title_scale_factor=1,
                      tick_scale_factor=0.8):
    """
    调整图形中所有子图的字体大小，包括标题、标签、刻度标签、图例等。

    参数：
    - fig: matplotlib.figure.Figure 对象
    - scale_factor: 基础字体大小的缩放因子
    - legend_scale_factor: 图例字体大小的缩放因子
    - label_scale_factor: 标签字体大小的缩放因子
    - title_scale_factor: 标题字体大小的缩放因子
    - tick_scale_factor: 刻度标签字体大小的缩放因子
    """
    for ax in fig.get_axes():
        fig_width, fig_height = fig.get_size_inches()
        base_font_size = min(fig_width, fig_height) * scale_factor
        print(f'base_font_size: {base_font_size}')

        # 设置标题和标签的字体大小
        title_font_size = base_font_size * title_scale_factor
        label_font_size = base_font_size * label_scale_factor
        print(f'title_font_size: {title_font_size}')
        print(f'label_font_size: {label_font_size}')
        ax.title.set_size(title_font_size)
        ax.xaxis.label.set_size(label_font_size)
        ax.yaxis.label.set_size(label_font_size)

        # 设置刻度标签的字体大小
        tick_font_size = base_font_size * tick_scale_factor
        print(f'tick_font_size: {tick_font_size}')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(tick_font_size)

        # 设置Colorbar的字体大小
        if hasattr(ax, 'colorbar'):
            colorbar_font_size = label_font_size
            colorbar_font_size = base_font_size * 0.7  # 固定的 colorbar_scale_factor
            print(f'colorbar_font_size: {colorbar_font_size}')
            colorbar = ax.colorbar
            if colorbar.ax.yaxis.label:
                colorbar.ax.yaxis.label.set_size(colorbar_font_size)
            for label in colorbar.ax.get_yticklabels():
                label.set_fontsize(colorbar_font_size)

        # 设置第二个Y轴的字体大小
        if hasattr(ax, 'right_ax'):
            right_ax = ax.right_ax
            right_ax.yaxis.label.set_size(label_font_size)
            for label in right_ax.get_yticklabels():
                label.set_fontsize(label_font_size * 0.8)  # 固定的 kick_scale_factor

        # 设置图例的字体大小
        legend = ax.get_legend()
        if legend:
            legend_font_size = base_font_size * legend_scale_factor
            print(f'legend_font_size: {legend_font_size}')
            for text in legend.get_texts():
                text.set_fontsize(legend_font_size)

        # 递归地处理子轴（如果有的话）
        if hasattr(ax, 'child_axes'):
            for child_ax in ax.child_axes:
                set_fonts_for_fig(child_ax.figure,
                                  scale_factor,
                                  legend_scale_factor,
                                  label_scale_factor,
                                  title_scale_factor,
                                  tick_scale_factor)


def auto_close_plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Close any existing figures before the function
        plt.close('all')
        try:
            # Execute the decorated function
            return func(*args, **kwargs)
        finally:
            # Ensure that all figures are closed after the function
            plt.close('all')

    return wrapper


# 定义一个格式化函数
def format_tick(tick_val, pos):
    return "%.2f" % tick_val


def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


def setup_colorbar(fig, vbins, cmap_name, ax=None, orientation='vertical', title=""):
    if ax is None:
        ax = fig.axes
    nbins = len(vbins)

    # norm = mpl.colors.BoundaryNorm(boundaries=vbins, ncolors=nbins)
    norm = mpl.colors.Normalize(vmin=vbins.min(), vmax=vbins.max())
    cmap_this = mpl.colors.ListedColormap(plt.get_cmap(cmap_name)(np.linspace(0, 1, nbins)))
    sm = plt.cm.ScalarMappable(cmap=cmap_this, norm=norm)
    sm.set_array([])

    # Automatically place the colorbar
    cl_bar = fig.colorbar(sm, ax=ax, orientation=orientation)

    cl_bar.set_label(f'{title}')
    return cl_bar


def setup_colors(cmap=None):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
        # 提取 colormap 的一部分
        colors = cmap(np.linspace(0.1, 0.9, 256))
        cmap = LinearSegmentedColormap.from_list('truncated_cmap', colors)
    if cmap is None:
        cdict = {
            'red': ((0.0, inter_from_256(64), inter_from_256(64)),
                    (1 / 5 * 1, inter_from_256(102), inter_from_256(102)),
                    (1 / 5 * 2, inter_from_256(200), inter_from_256(200)),
                    (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
                    (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
                    (1.0, inter_from_256(169), inter_from_256(169))),
            'green': ((0.0, inter_from_256(57), inter_from_256(57)),
                      (1 / 5 * 1, inter_from_256(178), inter_from_256(178)),
                      (1 / 5 * 2, inter_from_256(210), inter_from_256(210)),
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
        cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=100)
    colors = cmap(np.linspace(0, 1, 100))
    return cmap, colors


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


def set_font_sizes(ax, fig, scale_factor=5, coloarbar_scale_factor=0.7, kick_scale_factor=0.8, legend_scale_factor=0.9, label_scale_factor=1, title_scale_factor=1):
    """根据图形尺寸调整字体大小"""
    fig_width, fig_height = fig.get_size_inches()
    font_size = min(fig_width, fig_height) * scale_factor
    kick_font_size = font_size * kick_scale_factor

    # 设置标题和标签的字体大小
    label_font_size = font_size * label_scale_factor
    ax.title.set_size(font_size * title_scale_factor)
    ax.xaxis.label.set_size(label_font_size)
    ax.yaxis.label.set_size(label_font_size)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(kick_font_size)

    # 设置Colorbar的字体大小
    colorbar_font_size = font_size * coloarbar_scale_factor
    if hasattr(ax, 'colorbar'):
        ax.colorbar.ax.yaxis.label.set_size(colorbar_font_size)
        for label in ax.colorbar.ax.get_yticklabels():
            label.set_fontsize(colorbar_font_size)

    # 设置第二个Y轴的字体大小
    if hasattr(ax, 'right_ax'):
        ax.right_ax.yaxis.label.set_size(font_size)
        for label in ax.right_ax.get_yticklabels():
            label.set_fontsize(kick_font_size)

    # 设置图例的字体大小
    legend = ax.get_legend()
    if legend:
        legend_font_size = font_size * legend_scale_factor
        for text in legend.get_texts():
            text.set_fontsize(legend_font_size)

    # 递归地处理子轴
    if hasattr(ax, 'child_axes'):
        for child_ax in ax.child_axes:
            set_font_sizes(child_ax, fig, scale_factor, coloarbar_scale_factor, kick_scale_factor, legend_scale_factor)


def adjust_all_font_sizes(fig, scale_factor=5, legend_scale_factor=0.9, label_scale_factor=1, title_scale_factor=1):
    """调整所有子图的字体大小"""
    for ax in fig.get_axes():
        set_font_sizes(ax, fig, scale_factor, legend_scale_factor=legend_scale_factor, label_scale_factor=label_scale_factor, title_scale_factor=title_scale_factor)
