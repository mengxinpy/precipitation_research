import numpy as np
start = 1
end = 180
num_points = 30

base = end ** (1 / (num_points - 1))  # 生成对数点并取最近的整数
log_points = np.unique([int(round(start * base ** n)) for n in range(num_points)])
path_out = "C:\\ERA5\\1980-2019\\outer_klag_rain\\"
path_png = f'F:\\liusch\\remote_project\\climate_new\\temp_fig\\ear5_lag_area\\'
path_all = 'F:\\ERA5\\1980-2019\\'
