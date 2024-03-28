import numpy as np
import matplotlib.pyplot as plt


def calculate_event_durations(precipitation_array, percentile_th, mask_array):
    # 标记降水事件的开始和结束
    start_events = np.diff((precipitation_array > percentile_th).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th).astype('int'), append=0, axis=0) == -1
    # todo:检查一下逻辑
    # 初始化持续时间列表
    durations = []

    # 遍历每个格点
    for lat in range(precipitation_array.shape[1]):
        for lon in range(precipitation_array.shape[2]):
            # 如果当前格点不在感兴趣的区域内，则跳过
            if not mask_array[lat, lon]:
                continue

            # 获取当前格点的开始和结束事件
            start_indices = np.where(start_events[:, lat, lon])[0]
            end_indices = np.where(end_events[:, lat, lon])[0]

            # 计算并存储每个事件的持续时间
            event_durations = end_indices - start_indices + 1
            durations.extend(event_durations)

    # 将持续时间转换为数组以便进行统计分析
    durations = np.array(durations)
    return durations
    # plt.close()
    # # 绘制直方图，更多自定义样式
    # # 设置bins的边界为对数刻度
    # bins = np.unique(np.logspace(np.log10(min(durations)), np.log10(max(durations)), 30).round())
    # plt.hist(durations, bins=bins, alpha=0.5, color='skyblue', edgecolor='black', histtype='stepfilled', rwidth=0.9, log=True)
    #
    # # 设置图表的标题和坐标轴标签
    # plt.title('Histogram of Durations')
    # plt.xlabel('Duration')
    # plt.ylabel('Frequency')
    #
    # # 添加网格线
    # plt.grid(axis='y', alpha=0.75)
    # # 设置y轴为对数刻度
    # plt.yscale('log')
    # # plt.xscale('log')
    #
    # # 显示图表
    # plt.savefig(f'.\\temp_fig\\durations_time\\{fig_name}')
    # plt.close()
