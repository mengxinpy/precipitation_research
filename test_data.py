import xarray as xr
import os
import fnmatch

hah = xr.open_dataset('F:\\ERA5\\1980-2019\\total_precipitation\\198001_processed_day_1.nc')
aa = hah['tp'].values
print(aa.max())
