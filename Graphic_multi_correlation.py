import numpy as np
import xarray as xr
from Graphics import scatter_plots, scatter_plots_depart


def depart_ml_lat(key_list, key_limit):
    low_lat_list = []
    mid_lat_list = []

    for key in key_list:
        file_name = f'.\\internal_data\\{key}_era5\\{key}_frequency.nc'
        file_data = xr.open_dataarray(file_name)
        if key in ['duration', 'quiet', 'intensity']:
            file_data=np.log(file_data)
        # 归一化
        data = (file_data - file_data.min()) / (file_data.max() - file_data.min())

        low_lat = data.sel(latitude=slice(30, -30))
        south_hemisphere = data.sel(latitude=slice(-30, -60))
        north_hemisphere = data.sel(latitude=slice(60, 30))
        combined = xr.concat([north_hemisphere, south_hemisphere], dim="latitude")
        mid_lat = combined.sortby('latitude')

        low_lat_list.append(low_lat)
        mid_lat_list.append(mid_lat)

    scatter_plots_depart(das_low=low_lat_list, das_mid=mid_lat_list,limited_key=key_limit, save_path=f'.\\fig\\ml_{"_".join(key_list)}')


if __name__ == '__main__':
    key_list = ['k', 'duration', 'wet', 'intensity']
    depart_ml_lat(key_list)
