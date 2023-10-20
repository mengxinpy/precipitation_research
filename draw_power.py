import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import os
from matplotlib import colors as clr
import matplotlib.transforms as mtransforms
from cartopy.util import add_cyclic_point
import matplotlib as mpl

data = np.load('rain_area_rainfall_power_10years.npy')
