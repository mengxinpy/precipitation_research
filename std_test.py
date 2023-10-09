import xarray as xr
import calendar
import numpy as np
import matplotlib.pyplot as plt
import time

path = "F:\\zhaodan\\1940t_merged.nc"
ds = xr.open_dataset(path)
dst = ds.std(dim="time")['t']
plt.contourf(dst.longitude, dst.latitude, dst)
plt.show()
