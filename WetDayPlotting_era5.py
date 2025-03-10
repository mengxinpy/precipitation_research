import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
from matplotlib import colors as clr
import matplotlib.transforms as mtransforms
from cartopy.util import add_cyclic_point
import matplotlib as mpl
from lag_parameter import sp_frequency,sp_percentile
from lag_indirect_parameter import colorbar_title


# this function defines the colourbar
def inter_from_256(x):
    return np.interp(x=x, xp=[0, 255], fp=[0, 1])


cdict = {
    'red': ((0.0, inter_from_256(64), inter_from_256(64)),
            (1 / 5 * 1, inter_from_256(102), inter_from_256(102)),
            (1 / 5 * 2, inter_from_256(235), inter_from_256(235)),
            (1 / 5 * 3, inter_from_256(253), inter_from_256(253)),
            (1 / 5 * 4, inter_from_256(244), inter_from_256(244)),
            (1.0, inter_from_256(169), inter_from_256(169))),
    'green': ((0.0, inter_from_256(57), inter_from_256(57)),
              (1 / 5 * 1, inter_from_256(178), inter_from_256(178)),
              (1 / 5 * 2, inter_from_256(240), inter_from_256(240)),
              (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
              (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
              (1 / 5 * 5, inter_from_256(23), inter_from_256(23))),
    'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
             (1 / 5 * 1, inter_from_256(255), inter_from_256(255)),
             (1 / 5 * 2, inter_from_256(185), inter_from_256(185)),
             (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
             (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
             (1.0, inter_from_256(69), inter_from_256(69))),
}
cmap = clr.LinearSegmentedColormap('new_cmap', segmentdata=cdict, N=6)
colors = cmap(np.linspace(0, 1, 100))
path = "F:\\liusch\\remote_project\\climate_new\\precipitationnature-v2 (1)\\CameronMcE-precipitationnature-8349226\\"
# loading in wet day frequency data for subplot 1
# infile= xarray.open_dataset(path+'Figure1\\1980_2019_total_precipitation_masked.nc')
# infile = xarray.open_dataset('F:\\liusch\\remote_project\\climate_new\\cmporph_process_12.nc')['cmorph'] * 100
# infile = xarray.open_dataset('F:\\liusch\\remote_project\\climate_new\\imerg5_process.nc')['precipitationCal'] * 100
infile = xarray.open_dataset('lsprf_frequency_lat60.nc').to_array().squeeze() * 100
# infile = xarray.open_dataset('lspf_frequency.nc').to_array().squeeze() * 100
# infile = xarray.open_dataset('lsp_frequency_1.nc')['lsp'] * 100
# infile = xarray.open_dataset('lspf_frequency.nc')['lspf'] * 100
# infile_npy = np.load('_wetday_40year.npy') / (365 * 40 + 11) * 100
# loading in wet day percentile data for subplot 2
# dist = np.load(path + 'Extended Data\\EDF6\\CMORPH_wet_day_intensity_distribution.npy')
dist = np.load('lsprf_percentile_lat60.npy')
# dist = np.load('lspf_percentile.npy')
# dist = np.load('ear5_percentile_area_1deg_6area_lsp.npy')
# dist = np.load('ear5_percentile_area_1deg.npy')
# dist = np.load(path + 'Figure1\\total_precipitation_distribution.npy')
dist_arr = np.asarray(dist)
# dist_arr = np.multiply(dist_arr, 1000)  # converts from m to mm.

##PLOTTING##

# Defining the figure size
fig = plt.figure(figsize=(13, 15))
# fig.subplots_adjust(top=0.975)
# fig.subplots_adjust(left=0.1, right=0.8)
fig.tight_layout()

# setting axes and labels for subplot 1
ax = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, 'a.', transform=ax.transAxes + trans,
        fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold', color='black',
        bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
ax.coastlines()

ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())

# Plotting
plt.xlabel('Longitude', fontsize='15')
plt.ylabel('Latitude', fontsize='15')
# tp, longitude = add_cyclic_point(infile.precipitationCal.transpose(), infile.lon)  # connects the two ends of the longitude array
cont = plt.contourf(infile.longitude, infile.latitude, infile, levels=20, cmap=cmap, vmin=0, vmax=100)
ax.set_global()
# lat = np.linspace(90, -90, 721)
# lon = np.linspace(0, 360, 1440)
# cont = plt.contourf(lon, lat, infile, levels=20, cmap=cmap, vmin=0, vmax=100)

# setting axes and labels for subplot 2
ax = plt.subplot(2, 1, 2)
trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, 'b.', transform=ax.transAxes + trans,
        fontsize='large', verticalalignment='top', fontfamily='sans-serif', weight='bold',
        bbox=dict(facecolor='none', edgecolor='none', pad=3.0))

for ind, d in enumerate(dist_arr[0:100, :]):
    if str(type(d)) == "<class 'float'>":  # this statement ignores data that doesn't exist.
        None
    else:
        # plt.plot(d, np.arange(1, 101), '.', color=colors[ind], markersize=10)
        plt.plot(d, np.arange(1, 101), '.', color=colors[ind * 17], markersize=10)

# labelling figure and setting scaling and limits
plt.ylabel('Percentile', fontsize=15)
plt.xlabel('Cumulative precipitation (mm/day)', fontsize=15)
plt.yticks([1, 10, 25, 50, 75, 90, 99])
plt.grid(ls="--", color='k', alpha=0.5)
plt.xscale("log")
plt.xlim(1, 500)
plt.xticks([1, 10, 100, 500], labels=[1, 10, 100, 500])  # ,fontsize = 14)

# adding the colourbar
cmap = plt.get_cmap(cmap, 20)
norm = mpl.colors.Normalize(vmin=0, vmax=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.subplots_adjust(left=0.05, right=0.85)
cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.55])
clbar = fig.colorbar(sm, cax=cbar_ax, pad=-5)
clbar.set_label(colorbar_title+' (cover) ', fontsize='16')
# saving
plt.show()
# plt.savefig(path + 'percentile_era5.png')
