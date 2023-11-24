# 一些数据的初始化
import numpy as np

# area_name_list = ['western_pacific', 'eastern_pacific', 'Atlantic_Ocean', 'indian_ocean']
# file_name_list = ['split_wp', 'split_ep', 'split_al', 'split_io']
# variable_name_list = ['all_st_' + _ for _ in ['wp', 'ep', 'al', 'io']]
# out_num = 5
# grid_size = 0.25
# # 地区格点形成的矩阵的大小,定义为全局变量
# global_var = (160, 360)
# lat_range = (-20, 20)
# # 存放奇异值的矩阵
# singular_list = np.zeros((4, out_num))
# lat = np.linspace(-(global_var[0] // 8), (global_var[0] // 8), global_var[0])  # 纬度范围
# start_lon_list = [120, -150, -60, 30]
# end_lon_list = [-150, -60, 30, 120]
# gap = 0.1
# vapor_range = (0, 120)
start_year = 1990
end_year = start_year + 10
deg = 0.25
vapor_strat = 40
vapor_end = 85
vapor_bins_count = 210
marker_size = 5
wetday_gap = 10
rain_threshold = 1
flat = 90
