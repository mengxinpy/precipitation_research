import time
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.decomposition import TruncatedSVD

start_time = time.time()  # 记录开始时间
# 一些数据的初始化
area_name_list = ['western_pacific', 'eastern_pacific', 'Atlantic_Ocean', 'indian_ocean']
file_name_list = ['split_wp', 'split_ep', 'split_al', 'split_io']
variable_name_list = ['all_st_' + _ for _ in ['wp', 'ep', 'al', 'io']]
out_num = 20

# start_list为每个地区的起始经度, 以及一面的这几个变量用于为画图标经纬度记列标签
start_list = [120, 150, 60, 30]
start_lat = 20
grid_label_x_wp = [str(start_list[0] + i * 0.25)[:-2] + 'E' for i in range(0, 241, 40)] + [str(180 - i * 0.25)[:-2] + 'W' for i in range(40, 121, 40)]
grid_label_x_ep = [str(start_list[1] - i * 0.25)[:-2] + 'W' for i in range(0, 361, 40)]
grid_label_x_al = [str(start_list[2] - i * 0.25)[:-2] + 'W' for i in range(0, 241, 40)] + [str(i * 0.25)[:-2] + 'E' for i in range(40, 121, 40)]
grid_label_x_io = [str(start_list[3] + i * 0.25)[:-2] + 'E' for i in range(0, 361, 40)]
grid_label_y = [str(start_lat - i * 0.25)[:-2] + 'N' for i in range(0, 81, 40)] + [str(i * 0.25)[:-2] + 'W' for i in range(40, 81, 40)]
grid_label_list_x = [grid_label_x_wp, grid_label_x_ep, grid_label_x_al, grid_label_x_io]

# 地区格点形成的矩阵的大小,定义为全局变量
global_var = (720, 360)
exo_threshold = 0.01

# 存放奇异值的矩阵
singular_list_vapor = np.zeros((4, out_num))
singular_list_rain = np.zeros((4, out_num))


# 插值函数
def fill_vector(vector, layer):
    # vector = np.squeeze(vector)
    x = np.arange(len(vector))
    known_indices = (vector >= 0) & (~np.isnan(vector))
    known_x = x[known_indices]
    known_values = vector[known_indices]
    # print(layer)
    interp_func = interp1d(known_x, known_values, kind='linear', bounds_error=False, fill_value=np.nan)
    return interp_func(x)


# 数据处理的条件
def condition(element):
    return 0 <= element < 250


# 对数归一化
def log_normalize(data):
    log_data = np.log1p(data)  # 对数据进行对数变换，使用log1p避免对0取对数的情况
    log_mean = np.mean(log_data)
    normalized_data = log_data / log_mean
    return normalized_data


# 对矩阵元素进行平方归一化
def square_normalize(matrix):
    squared_matrix = np.square(matrix)  # 对矩阵元素进行平方
    sum_of_elements = np.nansum(squared_matrix)  # 计算平方后的矩阵元素之和
    normalized_matrix = matrix / np.sqrt(sum_of_elements)  # 对矩阵元素进行归一化

    return normalized_matrix


# 最大最小值归一化
def max_min_normalize(data):
    min_val, max_val = np.min(data), np.max(data)  # 计算对数变换后的数据的最小值和最大值
    normalized_data = (data - min_val) / (max_val - min_val)  # 对数据进行归一化

    return normalized_data


# 将本征矢,也就是svd后的行向量进行解码
def decode_matrix(data, indices):
    out_matrix = np.full(global_var, np.nan)
    out_matrix[indices[:, 0], indices[:, 1]] = data
    return out_matrix


# 自定义的svd分解函数
def my_svd(m, k, way):
    if way == 'vapor':
        m[m <= 0 & m > 250] = 0
    else:
        m[m <= 0 & m > 250] = 0


for a in range(0, 4):
    # 数据的读入,因为数据量比较大,所以每次只读入一个地区的变量,处理完后关闭
    with h5py.File(file_name_list[a] + '.mat', 'r') as f:
        print(f.keys())
        raw_variables = f[variable_name_list[a]][()]

    # 去掉格点数据中的陆地的点
    indices = np.argwhere(np.vectorize(condition)(raw_variables[0, :, :, 0]) & np.vectorize(condition)(raw_variables[1, :, :, 0]))
    filtered_elements = raw_variables[:, indices[:, 0], indices[:, 1], :]

    # 找出有大量异常值点的天数去掉
    eox_indices = np.unique(np.argwhere(np.sum(filtered_elements[0, :, :] <= 0, axis=0) > 0.05 * filtered_elements.shape[1]))
    filtered_elements = np.delete(filtered_elements, eox_indices, axis=2)
    # filled_data_vapor = np.array([fill_vector(filtered_elements[0, :, layer], layer) for layer in range(filtered_elements.shape[-1])])
    # filled_data_rain = np.array([fill_vector(filtered_elements[1, :, layer], layer) for layer in range(filtered_elements.shape[-1])])

    # 删除部分缺失数据比较多,无法进行插值的数据
    # filled_data_rain = np.delete(filled_data_rain, np.unique(np.argwhere(np.isnan(filled_data_rain))[:, 0]), axis=0).T
    # filled_data_vapor = np.delete(filled_data_vapor, np.unique(np.argwhere(np.isnan(filled_data_vapor))[:, 0]), axis=0).T

    # 对数据进行归一化操作,注意这里的降雨求平均有好几种方式
    # 这里尝试一下对所有值求平均
    avg_vapor = np.mean(np.ma.masked_array(filtered_elements[0, :, :], mask=filtered_elements[0, :, :] <= 0), axis=1).reshape((-1, 1))
        avg_rain = np.mean(np.ma.masked_array(filtered_elements[1, :, :], mask=filtered_elements[1, :, :] < 0), axis=1).reshape((-1, 1))
    filtered_elements_normalized_vapor = filtered_elements[0, :, :] - avg_vapor
    filtered_elements_normalized_rain = filtered_elements[1, :, :] - avg_rain
    filtered_normalized_vapor = square_normalize(filtered_elements_normalized_vapor)
    filtered_normalized_rain = square_normalize(filtered_elements_normalized_rain)

    # 进行奇异值分解并将数据保存
    svd = TruncatedSVD(n_components=out_num)
    reduced_matrix_vapor = svd.fit_transform(filtered_normalized_vapor)
    reduced_matrix_rain = svd.fit_transform(filtered_normalized_rain)
    singular_list_rain[a, :] = svd.singular_values_
    singular_list_vapor[a, :] = svd.singular_values_

    # 从分解后的矩阵提取元素同时将图片保存到文件夹
    for i in range(out_num):
        for v in ['vapor', 'rain']:
            if v == 'vapor':
                decode_out_variables = np.flipud(decode_matrix(reduced_matrix_vapor[:, i], indices))
            else:
                decode_out_variables = np.flipud(decode_matrix(reduced_matrix_rain[:, i], indices))
            # 保存解码后的数据
            np.save('decode_' + v + str(a), decode_out_variables)
            # 画图
            stats_ax = sns.heatmap(decode_out_variables, cmap="viridis")
            stats_ax.set_xticks(range(0, 361, 40))
            stats_ax.set_xticklabels(grid_label_list_x[a])
            stats_ax.set_yticks(range(0, 161, 40))
            stats_ax.set_yticklabels(grid_label_y)
            stats_ax.xaxis.set_tick_params(rotation=0)
            save_path = 'pcaPicture3th\\' + area_name_list[a] + '\\' + area_name_list[a] + '_' + str(i) + 'th_' + v + '.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
    f.close()
np.save('singular_vapor.npy', singular_list_vapor)
np.save('singular_rain.npy', singular_list_rain)
end_time = time.time()  # 记录结束时间
run_time = end_time - start_time  # 计算运行时间
print("程序运行时间为：", run_time, "秒")

# @njit
# def process_data(filtered_elements):
#     # 找出有大量异常值点的天数去掉
#     eox_indices = np.unique(np.argwhere(np.sum(filtered_elements[0, :, :] < 0, axis=0) > 1000))
#     filtered_elements = np.delete(filtered_elements, eox_indices, axis=2)
#
#     filled_data_vapor = np.empty_like(filtered_elements[0])
#     filled_data_rain = np.empty_like(filtered_elements[1])
#
#     for layer in range(filtered_elements.shape[-1]):
#         filled_data_vapor[:, layer] = fill_vector(filtered_elements[0, :, layer], layer)
#         filled_data_rain[:, layer] = fill_vector(filtered_elements[1, :, layer], layer)
#
#     # 删除部分缺失数据比较多,无法进行插值的数据
#     filled_data_rain = np.delete(filled_data_rain, np.unique(np.argwhere(np.isnan(filled_data_rain))[:, 0]), axis=0).T
#     filled_data_vapor = np.delete(filled_data_vapor, np.unique(np.argwhere(np.isnan(filled_data_vapor))[:, 0]), axis=0).T
#
#     # 对数据进行归一化操作,注意这里的降雨求平均有好几种方式
#     avg_vapor = np.mean(np.ma.masked_array(filled_data_vapor > 0, mask=filled_data_vapor), axis=1).reshape((-1, 1))
#     avg_rain = np.mean(np.ma.masked_array(filled_data_rain > 0, mask=filled_data_rain), axis=1).reshape((-1, 1))
#     filtered_elements_normalized_vapor = filled_data_vapor - avg_vapor
#     filtered_elements_normalized_rain = filled_data_rain - avg_rain
#     filtered_normalized_vapor = square_normalize(filtered_elements_normalized_vapor)
#     filtered_normalized_rain = square_normalize(filtered_elements_normalized_rain)
#
#     return filtered_normalized_vapor, filtered_normalized_rain
#
