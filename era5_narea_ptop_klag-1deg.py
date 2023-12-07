import numpy as np
import xarray as xr
from test_log import log_points
from lag_parameter import bins, indices, area_top_per_all

pathr = "E:\\ERA5\\1980-2019\\total_precipitation\\"
path_out = "E:\\ERA5\\1980-2019\\outer_klag_rain\\"

dr = xr.open_mfdataset(pathr + '*processed_day_1.nc')
result_klag = np.zeros((area_top_per_all.shape[0], len(log_points) + 2, len(bins), 100))
raw_data = dr.to_array().values.squeeze()
condition_wetday = raw_data > 1
assert len(bins) == indices.max()
assert indices.min() == 1
for j in range(len(bins)):
    print(f'area:{j}')
    condition_area = (indices == j + 1)
    for p, area_topper in enumerate(area_top_per_all):
        print(f'per:{p}')
        condition_topper = raw_data > area_topper[j]
        condition_topper_area = condition_topper & condition_area

        rain_area = raw_data[condition_wetday & condition_area]
        result_klag[p, 0, j, :] = np.nanpercentile(rain_area, np.arange(1, 101))
        rain_area_topper = raw_data[condition_topper_area & condition_wetday]
        result_klag[p, 1, j, :] = np.nanpercentile(rain_area_topper, np.arange(1, 101))

        for ind, k in enumerate(log_points):
            print(k)
            condition_topper_area_klag = np.roll(condition_topper_area, shift=k, axis=0)
            condition_topper_area_klag = condition_topper_area_klag & condition_wetday
            condition_topper_area_klag[0:k, :, :] = False
            result_klag[p, ind + 2, j, :] = np.nanpercentile(raw_data[condition_topper_area_klag], np.arange(1, 101))

np.save(f'{path_out}\\result_klag_1deg_6area_topper', result_klag)
