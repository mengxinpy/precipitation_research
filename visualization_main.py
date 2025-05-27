from matplotlib import pyplot as plt
from workflow import main_process, load_data
from utils import get_list_form_onat
from config import intern_data_path
from plot_lib import multi_scatter_plot, plt_duration_origin, plt_duration_gamma, scatter_plots_combined_no_rect, scatter_plots_combined_final
from utils import get_dataset_duration_all
from config import path_png
import numpy as np
from utils import depart_ml_lat, point_path_data
from utils import get_refine_da_list
from plot_lib import scatter_plots_combined_with_map, scatter_plots_depart, pt_single, pt_6_combine, scatter_plots_combined_with_map_new
from map_lib import plot_map, plot_map_1, plot_distribution, combined_plot_separate_saves, plot_time_series, plot_time_series_origin
import xarray as xr
from plot_lib import plt_duration_hist
from map_lib import combined_plot
from plot_lib import multi_threshold_scatter_plot, single_scatter_plot
from processor_map import get_duration_frequency

data_set_list = ['era5', 'mswep', 'persiann']
data_set = data_set_list[0]
dr = point_path_data('total_precipitation', data_set)
data_dict = xr.open_dataset('./internal_data/data_dict.nc')
all_th= xr.open_dataarray(f'{intern_data_path}/all_th.nc')  # Assuming all_th.nc is a DataArr

# %% fig1b
coordinates = [(-67, 0), (150, 5), (0, -55), (-120, -45), (-74, 50), (88, -30)]
# onat_list_fig1 = [ (-114.76, 23.89), (-130.48, 19.91), (-120.26, -4.78), (-135.98, -10.35), (-155.63, -14.34), (-172.93, -12.74) ]
# grid_list1=[ (88.27, -24.13), (89.49, -16.93), (90.38, -13.06), (92.34, -7.72), (95.37, -3.11), (99.11, 5.01) ]
# grid_list2=[ (70.83, -29.33), (74.04, -23.80), (76.36, -20.29), (78.31, -15.68), (79.61, -12.03), (82.46, -5.95) ]
# grid_list3=[ (70.83, -29.33), (74.04, -23.80), (76.36, -20.29), (78.31, -15.68), (79.61, -12.03), (88.34, -2.00) ]
grid_list4= [(-67, 0), (80, -5), (-30, -50), (-120, -35), (-105, 50), (110, -24)]
onat_list_fig1 =grid_list4
# #
plot_time_series_origin(dr=dr,sp_fp_ts=path_png+'6ts.svg',onat_list=onat_list_fig1,all_th=all_th)

# %% fig2
data_percentile, bins, indices = load_data(f'internal_data/wet_{data_set}/wet_data.npz')
bins_6=bins
duration_hist=np.load(f'internal_data/wet_{data_set}/ltp_duration_newest.npy',allow_pickle=True) # todo: 需要重新跑数据
plt_duration_origin(duration_hist[1], title='Duration', vbins=bins_6, fig_name=f'fig/wet_{data_set}/sts_duration_newest_{data_set}.svg')
# %% fig3
key_list = ['duration', 'wet']
data_list = get_refine_da_list(['top20%','wet'], dataset=data_set, log_duration_etc=False, unify=False, wet_th=True)
data_list[0]=data_list[0].rename('Duration (day)')

low_lat_list, mid_lat_list = depart_ml_lat(data_list)


# 调用修改后的函数，添加range_filter参数
kb_mid = scatter_plots_combined_final(
    low_lat_list,
    mid_lat_list,
    save_path=f'./fig/wet_{data_set}/fig-3-{data_set}',
)
# %% fig4
#
from utils import get_refine_da_list
from utils import depart_ml_lat


def generate_percentage_labels(start=0, stop=100, num=10):
    percentages = np.linspace(start, stop, num)
    labels = [f"{round(p)}%" for p in percentages]
    return percentages, labels


# Example usage
# labels = generate_percentage_labels(start=10, stop=40, num=10)
# print(labels[-1])

wet_data_list = []
wet_data = get_refine_da_list(['wet'])[0]
#
percentile_list_10, labels_10 = generate_percentage_labels(10, 40, 4)
percentile_list1, labels1 = generate_percentage_labels(1, 7, 4)
#
for ind, percentile in enumerate(percentile_list_10):
    percentile_wet_data = get_all_duration_frequency(point_path_data('total_precipitation',data_set), percentile,out_dir=f'wet_{data_set}')
    print(f'finished:{ind} percentile:{percentile}')
percentile_key_list = [f'top{_}' for _ in labels1]+[f'top{_}' for _ in labels_10]
percentile_duration_data = get_dataset_duration_all(percentile_key_list,data_set,unify=False)
# %% visionlize dataarraylist
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

percentile_duration_data = get_refine_da_list(percentile_key_list, unify=False)


for idx, da in enumerate(percentile_duration_data):
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    # ax.set_extent([])  # 设定地图范围，可以根据你的数据调整

    # 绘制海岸线、国家边界等地图背景
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    # 假设每个 da 是一个 xarray DataArray，并且它有经纬度坐标
    # c1 = ax.contourf(da['longitude'], da['latitude'], da.values, 60, transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=4)
    c1 = ax.contourf(da['longitude'], da['latitude'], da.values, 60, transform=ccrs.PlateCarree(), cmap='Blues')

    # 可以根据需要添加标题
    ax.set_title(f"Percentile: {percentile_key_list[idx]}%", fontsize=14)
    # 关联colorbar
    fig.colorbar(c1,ax=ax, orientation='horizontal')
    plt.show()



intensity = xr.open_dataarray(f'./internal_data/intensity_era5/intensity_frequency.nc')
intensity_condition = intensity > 1
# 准备数据列表
all_wet_low, all_duration_low = [], []
all_wet_mid, all_duration_mid = [], []
titles = []

for ind, duration_data in enumerate(percentile_duration_data):
    duration_data, intensity = xr.align(duration_data, intensity, join="right")
    duration_data = duration_data.where(intensity > 1, np.nan)
    duration_list_low, duration_list_mid = depart_ml_lat([duration_data])
    wet_list_low, wet_list_mid = depart_ml_lat([wet_data])

    all_wet_low.append(wet_list_low[0])
    all_duration_low.append(duration_list_low[0])
    all_wet_mid.append(wet_list_mid[0])
    all_duration_mid.append(duration_list_mid[0])
    s = percentile_key_list[ind]
    # 除去首位后的前三个字符
    first_three = s[3:]
    # 字符串中的最后一个字符
    last_char = first_three[:-1]
    titles.append(f"Threshold: {100-eval(last_char)}th percentile")  # 生成子图标题

# 计算行列数（例如5个子图可以排成2行3列）
n_subplots = len(percentile_duration_data)
ncols = 2
nrows = (n_subplots + ncols - 1) // ncols

# 调用合并绘图函数
multi_scatter_plot(
    wet_data_list_low=all_wet_low,
    duration_list_low=all_duration_low,
    wet_data_list_mid=all_wet_mid,
    duration_list_mid=all_duration_mid,
    nrows=nrows,
    ncols=ncols,
    titles=titles,
    save_path=f"./fig/wet_{data_set}/fig-4-combined_plot.png"
)
# %% multi_graph
# for ind, duration_data in enumerate(percentile_duration_data):
#     # 去掉强度低于1的部分
#     duration_data = duration_data.where(intensity > 1, np.nan)
#     duration_data_list_low, duration_data_list_mid = depart_ml_lat([duration_data])
#     wet_data_list_low, wet_data_list_mid = depart_ml_lat([wet_data])
#     print(duration_data_list_mid[0])
#     single_scatter_plot(wet_data_list_low[0], duration_data_list_low[0], wet_data_list_mid[0], duration_data_list_mid[0])
# single_scatter_plot(wet_data_list_low[0], duration_data_list_low[0], wet_data_list_mid[0], duration_data_list_mid[0],save_path=f'{path_png}{duration_data_list_low[0].name.replace(".", "d")}')

# single_scatter_plot()
# multi_threshold_scatter_plot(low_list, mid_list, save_path=f'./fig/ml_single_percentile_{ind}')


# year_power = dataset['power']
# dr = point_path_data('total_precipitation')
# all_th = xr.open_dataarray(f'{intern_data_path}/all_th.nc')  # Assuming all_th.nc is a DataArray
# # coordinates = [(-155.73, -35.04), (-157.35, -20.71), (-154.92, -14.34), (-165.44, -9.56), (-167.87, -1.59), (-159.78, 5.58)]
# coordinates_duration = [ (132.84, -28.67), (130.48, -15.13), (133.62, -9.56), (135.20, -1.59), (139.13, 2.39), (125.76, 0.00) ]
# combined_plot(year_power, dr, onat_list=coordinates_duration, sp_fp='combined_power', all_th=all_th)
# %% fig5
key_list = ['duration', 'wet']
data_list = get_refine_da_list(key_list, log_duration_etc=True, unify=False,wet_th=True)

#读取风速的数据并处理
wind_10 = xr.open_dataarray('/Volumes/DiskShared/ERA5/1980-2019/data_0.nc').coarsen(longitude=4, latitude=4, boundary='trim').mean().sel(latitude=slice(60, -60))
power = data_dict['power'].rename('Yearly spectrum')
wind_10 = wind_10.mean(dim='date').rename('Wind(m/s)')
data_list.append(wind_10)
data_list.append(power)
low_lat_list, mid_lat_list = depart_ml_lat(data_list)
scatter_plots_depart(low_lat_list, mid_lat_list, save_path='./fig/wet_duration_si10_')