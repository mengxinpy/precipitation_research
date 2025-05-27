import numpy as np

start = 1
end = 180
num_points = 30

base = end ** (1 / (num_points - 1))  # 生成对数点并取最近的整数
log_points = np.unique([int(round(start * base ** n)) for n in range(num_points)])
path_out = "C:/ERA5/1980-2019/outer_klag_rain/"
# path_png = f'F:/liusch/remote_project/climate_new/temp_fig/ear5_lag_area/'
path_test_png = f'F:/liusch/remote_project/climate_new/temp_fig/'
global_png_path = f'F:/liusch/remote_project/climate_new/temp_fig/'
path_all = 'F:/ERA5/1980-2019/'
intern_data_path='./internal_data/'
fig='./fig/'
path_png = fig
onat_list = [(-67, 0), (150, 5), (0, -55), (-120, -45), (-74, 50), (60, -33)]
onat_list_one = [(100, 5), (189, 5), (38, -32), (21, -47), (347, -39), (328.375, -39.375)]
onat_list.reverse()
