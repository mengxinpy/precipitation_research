from plt_lib import *
from process_lib import *
from common_imports import *
import matplotlib.pyplot as plt
import config


def vapor_rain(restore_matrix):
    restore_vapor = np.split(restore_matrix, 2, axis=1)[0]
    restore_rain = np.split(restore_matrix, 2, axis=1)[1]
    vapor_rain_list = np.zeros((out_num, round(vapor_range[1] // gap)))
    for i in range(out_num):
        for j in range(round(vapor_range[1] // gap)):
            target_vapor_value = j * gap
            mask = (restore_vapor[i, :, :] == target_vapor_value)
            avg_rain = np.nanmean(restore_rain[i, :, :][mask])
            vapor_rain_list[i, j] = avg_rain

    return vapor_rain_list


config.out_num = 5
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
start_time = time.time()  # 记录开始时间
# 从文件名中生成版本号
th = ''.join(('_'.join(os.path.basename(__file__).split("_")[1:])).split(".")[:-1])

# 画图的函数

for a in range(1, 4):
    # 数据的读入,因为数据量比较大,所以每次只读入一个地区的变量,处理完后关闭
    with h5py.File(config.file_name_list[a] + '.mat', 'r') as f:
        print(f.keys())
        raw_variables = f[config.variable_name_list[a]][:, 280:440, :, :]
        f.close()

    # 数据筛选
    filtered_elements, indices = data_filter(raw_variables, a)
    # 数据预处理
    filtered_normalized_completed = data_raw_process(filtered_elements, indices)
    # 进行奇异值分解
    svd = TruncatedSVD(n_components=config.out_num)
    reduced_matrix = svd.fit_transform(filtered_normalized_completed)
    config.singular_list[a, :] = svd.singular_values_
    v_matrix = svd.components_
    # 重塑矩阵方便利用numpy的广播机制
    V = v_matrix[:, np.newaxis, :]
    U = reduced_matrix.T[:, :, np.newaxis]
    result = np.einsum('ijk,ikl->ijl', U, V)
    out = vapor_rain(result)

    # 画图
    draw_timeseries(v_matrix=v_matrix, th=th, a=a, out_num=config.out_num)
    draw_area(reduced_matrix, indices, th=th, a=a, out_num=out_num)
plt_singular(config.singular_list, th, 'all')
loglog_singular(config.singular_list, th, 'all')
np.save('pcaPicture_' + th + '//singular_' + th + '.npy', config.singular_list)
# np.save('pcaPicture_' + th + '//components_' + th + '.npy', v_matrix)
end_time = time.time()  # 记录结束时间
run_time = end_time - start_time  # 计算运行时间
print("程序运行时间为：", run_time, "秒")
