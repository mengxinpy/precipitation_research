from common_imports import *
from config import *


def condition(element):
    return 0 <= element < 250


# 对矩阵元素进行平方归一化
def square_normalize(matrix):
    squared_matrix = np.square(matrix)  # 对矩阵元素进行平方
    sum_of_elements = np.nansum(squared_matrix)  # 计算平方后的矩阵元素之和
    normalized_matrix = matrix / np.sqrt(sum_of_elements)  # 对矩阵元素进行归一化

    return normalized_matrix


# 将本征矢,也就是svd后的行向量进行解码

def data_raw_process(filtered_elements, indices):
    eox_indices_time = np.unique(np.argwhere(np.sum(filtered_elements[0, :, :] <= 0, axis=0) > 0.1 * filtered_elements.shape[1]))
    eox_indices_area = np.unique(np.argwhere(np.sum(filtered_elements[0, :, :] < 0, axis=1) > 0.1 * filtered_elements.shape[2]))
    # np.save('eox_' + area_name_list[a], eox_indices)
    filtered_elements = np.delete(filtered_elements, eox_indices_time, axis=2)
    filtered_elements = np.delete(filtered_elements, eox_indices_area, axis=1)
    indices = np.delete(indices, eox_indices_area, axis=0)
    filtered_elements = np.where(filtered_elements >= 0, filtered_elements, np.nan)
    # 去单位化
    avg_vapor = np.mean(np.ma.masked_array(filtered_elements[0, :, :], mask=((filtered_elements[0, :, :] == 0) | (np.isnan(filtered_elements[0, :, :])))))
    avg_rain = np.mean(np.ma.masked_array(filtered_elements[1, :, :], mask=((filtered_elements[1, :, :] == 0) | (np.isnan(filtered_elements[1, :, :])))))
    filtered_elements_normalized_vapor = (filtered_elements[0, :, :] - avg_vapor) / avg_vapor
    filtered_elements_normalized_rain = (filtered_elements[1, :, :] - avg_rain) / avg_rain
    # 处理为涨落的变化
    avg_vapor = np.mean(np.ma.masked_array(filtered_elements[0, :, :], mask=((filtered_elements[0, :, :] == 0) | (np.isnan(filtered_elements[0, :, :])))), axis=1).reshape((-1, 1))
    avg_rain = np.mean(np.ma.masked_array(filtered_elements[1, :, :], mask=((filtered_elements[1, :, :] == 0) | (np.isnan(filtered_elements[1, :, :])))), axis=1).reshape((-1, 1))
    filtered_elements_normalized_vapor = (filtered_elements[0, :, :] - avg_vapor) / avg_vapor
    filtered_elements_normalized_rain = (filtered_elements[1, :, :] - avg_rain) / avg_rain

    filtered_elements_normalized = np.concatenate((filtered_elements_normalized_vapor, filtered_elements_normalized_rain), axis=0)
    filtered_elements_normalized_derivative = filtered_elements_normalized / np.nanstd(filtered_elements_normalized, axis=1).reshape((-1, 1))
    filtered_normalized = square_normalize(filtered_elements_normalized_derivative)
    filtered_normalized_completed = SoftImpute(max_iters=2).fit_transform(filtered_normalized)
    return filtered_normalized_completed


def data_filter(raw_variables, a):
    lon_index_range, lat_index_range = get_index_range.get_index_range((start_lon_list[a], end_lon_list[a]), lat_range)
    land_mary = np.flipud(np.load('ocean_mask.npy')[lat_index_range[0]:lat_index_range[1], lon_index_range[0]:lon_index_range[1]])
    iterations = 2  # 您可以更改这个值来调整扩大范围的程度
    expanded_ocean_mask = binary_erosion(land_mary & np.vectorize(condition)(raw_variables[0, :, :, 0]) & np.vectorize(condition)(raw_variables[1, :, :, 0]), iterations=iterations)
    indices = np.argwhere(expanded_ocean_mask)
    filtered_elements = raw_variables[:, indices[:, 0], indices[:, 1], :]
    return filtered_elements, indices
