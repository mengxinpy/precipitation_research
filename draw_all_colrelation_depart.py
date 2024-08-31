import numpy as np
import xarray as xr
from plt_temp import scatter_plots, scatter_plots_depart, scatter_plots_depart_ns, scatter_plots_depart_month
from lag_path_parameter import path_out


def depart_sn_glb(key_list):
    south_hemisphere_list = []
    north_hemisphere_list = []

    for key in key_list:
        if key == 'wet':
            file_name = f'wetday_vt_wet_frequency_lat60.nc'
        else:
            file_name = f'wetday_vt_duration_{key}_frequency_lat60.nc'
        data = xr.open_dataarray(path_out + file_name)

        # low_lat = data.sel(latitude=slice(30, -30))
        south_hemisphere = data.sel(latitude=slice(0, -60))
        north_hemisphere = data.sel(latitude=slice(60, 0))
        # combined = xr.concat([north_hemisphere, south_hemisphere], dim="latitude")
        # mid_lat = combined.sortby('latitude')

        if key != 'wet':
            south_hemisphere = np.log(south_hemisphere)
            north_hemisphere = np.log(north_hemisphere)

        south_hemisphere_list.append(south_hemisphere)
        north_hemisphere_list.append(north_hemisphere)

    scatter_plots_depart_ns(matrices_south=south_hemisphere_list, matrices_north=north_hemisphere_list, var_names=key_list, figure_name='south_north_all_correlation_depart')


def depart_ml_lat(key_list):
    low_lat_list = []
    mid_lat_list = []

    for key in key_list:
        file_name = f'wetday_vt_{key}_frequency_lat60.nc'
        data = xr.open_dataarray(path_out + file_name)

        low_lat = data.sel(latitude=slice(30, -30))
        south_hemisphere = data.sel(latitude=slice(-30, -60))
        north_hemisphere = data.sel(latitude=slice(60, 30))
        combined = xr.concat([north_hemisphere, south_hemisphere], dim="latitude")
        mid_lat = combined.sortby('latitude')

        if key in ['duration', 'quiet', 'intensity']:
            low_lat = np.log(low_lat)
            mid_lat = np.log(mid_lat)

        low_lat_list.append(low_lat)
        mid_lat_list.append(mid_lat)

    scatter_plots_depart(matrices_low=low_lat_list, matrices_mid=mid_lat_list, var_names=key_list, figure_name='all_correlation_depart')

def depart_ml_lat_month(key_list):
    low_lat_list = []
    mid_lat_list = []

    for key in key_list:
        if key == 'wet':
            file_name = f'wetday_vt_wet_frequency_lat60.nc'
        else:
            file_name = f'wetday_vt_duration_{key}_frequency_lat60.nc'
        data = xr.open_dataarray(path_out + file_name)

        low_lat = data.sel(latitude=slice(30, -30))
        south_hemisphere = data.sel(latitude=slice(-30, -60))
        north_hemisphere = data.sel(latitude=slice(60, 30))
        combined = xr.concat([north_hemisphere, south_hemisphere], dim="latitude")
        mid_lat = combined.sortby('latitude')

        if key != 'wet':
            low_lat = np.log(low_lat)
            mid_lat = np.log(mid_lat)

        low_lat_list.append(low_lat)
        mid_lat_list.append(mid_lat)

    scatter_plots_depart_month(matrices_low=low_lat_list, matrices_mid=mid_lat_list, var_names=key_list, figure_name='mid_lat_all_correlation_depart_month')


if __name__ == '__main__':
    key_list = ['k', 'duration', 'wet', 'intensity']
    depart_ml_lat(key_list)
