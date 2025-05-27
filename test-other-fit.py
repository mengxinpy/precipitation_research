from matplotlib import pyplot as plt
from workflow import main_process, load_data
from utils import get_list_form_onat
from config import intern_data_path
from plot_lib import multi_scatter_plot, plt_duration_origin, plt_duration_gamma, scatter_plots_combined_no_rect, scatter_plots_combined_final, plt_duration_exp
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
from processor_main import get_duration_frequency, get_all_duration_frequency

data_set_list = ['era5', 'mswep', 'persiann']
data_set = data_set_list[0]

data_percentile, bins, indices = load_data(f'internal_data/wet_{data_set}/wet_data.npz')
bins_6=bins
duration_hist=np.load(f'internal_data/wet_{data_set}/ltp_duration_newest.npy',allow_pickle=True) # todo: 需要重新跑数据
plt_duration_gamma(duration_hist[0], title='Duration', vbins=bins_6, fig_name=f'fig/wet_{data_set}/sts_duration_newest_{data_set}_gamma.svg')
# plt_duration_exp(duration_hist[0], title='Duration', vbins=bins_6, fig_name=f'fig/wet_{data_set}/sts_duration_newest_{data_set}_exp.svg')
# plt_duration_origin(duration_hist[1], title='Duration', vbins=bins_6, fig_name=f'fig/wet_{data_set}/sts_duration_newest_{data_set}_exp.svg')