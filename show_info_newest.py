import asyncio
import sys
import logging

# Modify event loop policy for Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np
import pandas as pd
import xarray as xr
import hvplot.xarray
import holoviews as hv
import geoviews as gv
import panel as pn
from scipy.fft import fft
import cartopy.crs as ccrs
from Function_common import point_path_data  # Ensure this function is properly defined and imported
from cartopy.util import add_cyclic_point
import param

# Start extensions
hv.extension('bokeh')
gv.extension('bokeh')
pn.extension()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Read data
dataset = xr.open_dataset('./internal_data/data_dict.nc')
duration = dataset['duration']
dot = dataset['dot']
wet = dataset['wet']
power = dataset['power']
ar1_JAS = dataset['ar1_JAS']
ar1_DJF = dataset['ar1_DJF']
power_3days = dataset['power_3days']
gen_duration = dataset['gen_duration']  # 新增的gen_duration
gen_wet = dataset['gen_wet']  # 新增的gen_wet
AR_Coefficient = dataset['AR_Coefficient']
power_hour = dataset['power_hour']
AR_C = dataset['AR_C']

# Read threshold data (Step 4)
intern_data_path = './internal_data'  # Adjust the path as needed
all_th = xr.open_dataarray(f'{intern_data_path}/all_th.nc')  # Assuming all_th.nc is a DataArray
all_th = all_th.rename({'top_bins': 'threshold'})


# 2. Create a class to hold click data
class ClickData(param.Parameterized):
    data = param.Parameter(default=(None, None, ''))


click_data = ClickData()


# 3. Modify plot_map function to use a single TapStream
def plot_map(data_array, title):
    logging.debug(f"Creating map for {title}")

    # Add cyclic point to handle longitude wrapping
    data_cyclic, lon_cyclic = add_cyclic_point(data_array, data_array.longitude)

    # Create new DataArray with cyclic longitude
    data_filtered = xr.DataArray(
        data_cyclic,
        dims=['latitude', 'longitude'],
        coords={'latitude': data_array.latitude, 'longitude': lon_cyclic}
    )

    from bokeh.models import TapTool

    def customize_tap_behavior(plot, element):
        for tool in plot.state.tools:
            if isinstance(tool, TapTool):
                tool.behavior = 'inspect'

    map_plot = data_filtered.hvplot.contourf(
        x='longitude',
        y='latitude',
        geo=True,
        projection=ccrs.PlateCarree(),
        crs=ccrs.PlateCarree(),
        cmap='rainbow',
        colorbar=True,
        width=600,
        height=300,
        xlim=(-180, 180),
        ylim=(-20, 20),
        title=title,
        tools=['tap', 'hover'],
        levels=20,
        global_extent=False
    ).opts(
        selection_alpha=1.0,
        nonselection_alpha=1.0,
        selection_color=None,
        nonselection_color=None,
        colorbar_opts={'width': 10},
        hooks=[customize_tap_behavior]
    )

    coast = gv.feature.coastline.opts(
        projection=ccrs.PlateCarree(),
        line_width=1,
        color='black',
        global_extent=False,
        xlim=(-180, 180),
        ylim=(-20, 20),
    )

    # Create TapStream and add subscriber
    tap_stream = hv.streams.Tap(source=map_plot)

    # Modify tap_subscriber to update the data parameter
    def tap_subscriber(x, y):
        click_data.data = (x, y, title)
        logging.info(f"Clicked on {title} at x={x}, y={y}")

    tap_stream.add_subscriber(tap_subscriber)

    return map_plot * coast


# 加载原始降水数据和生成的降水数据
dr = point_path_data('total_precipitation').load()


# 4. Define the click event processing function
def process_click(lon, lat, source_map):
    try:
        logging.info(f"Processing click from map: {source_map}, at lon: {lon}, lat: {lat}")

        # Use point_path_data function to get data

        # Select data for the year 2010
        selected_year = '2010'

        # 处理原始数据的时间序列
        time_series_original = dr.sel(longitude=lon, latitude=lat, method='nearest') + 1e-4
        year_data_original = time_series_original.sel(time=str(selected_year))

        if year_data_original.size == 0:
            logging.warning(f"No data found for {selected_year} at lon: {lon}, lat: {lat}")
            placeholder = hv.Text(0.5, 0.5, f"No data for {selected_year}").opts(
                fontsize=20, align='center', text_align='center'
            )
            return pn.Column(placeholder, placeholder, placeholder, placeholder, placeholder, placeholder)

        # Convert time to pandas datetime for better handling
        time_series_pds_original = time_series_original.to_pandas()

        # Verify the length of the time series
        n_original = len(time_series_pds_original)
        logging.debug(f"Length of original time_series: {n_original}")

        if n_original < 2:
            logging.warning(f"Original time series too short for FFT: {n_original} points")
            placeholder = hv.Text(0.5, 0.5, f"Original time series too short for FFT").opts(
                fontsize=20, align='center', text_align='center'
            )
            return pn.Column(placeholder, placeholder, placeholder, placeholder, placeholder, placeholder)

        # Apply Hanning window before FFT on original data
        window_original = np.hanning(n_original)
        windowed_data_original = time_series_pds_original.values * window_original
        logging.debug(f"Hanning window applied to original data. Windowed data length: {len(windowed_data_original)}")

        # Perform Fourier Transform on windowed original data
        fft_vals_original = np.abs(fft(windowed_data_original))
        freqs_original = np.fft.fftfreq(len(windowed_data_original), d=1)  # Assuming daily data; adjust 'd' if different
        logging.debug(f"FFT performed on original data. fft_vals length: {len(fft_vals_original)}, freqs length: {len(freqs_original)}")

        # Keep only positive frequencies (exclude zero frequency)
        half_length_original = len(freqs_original) // 2
        positive_freqs_original = freqs_original[:half_length_original]
        positive_fft_original = fft_vals_original[:half_length_original]
        logging.debug(f"Positive frequencies extracted from original data. positive_freqs length: {len(positive_freqs_original)}")

        # Exclude zero frequency
        mask_positive_original = positive_freqs_original > 0
        positive_freqs_original = positive_freqs_original[mask_positive_original]
        positive_fft_original = positive_fft_original[mask_positive_original]
        logging.debug(f"Zero frequency excluded from original data. Mask length: {len(mask_positive_original)}, positive_freqs length: {len(positive_freqs_original)}")

        # Ensure that positive_freqs and positive_fft have the same length
        if len(positive_freqs_original) != len(positive_fft_original):
            logging.error(f"Mismatch after masking in original data: positive_freqs length {len(positive_freqs_original)} vs positive_fft length {len(positive_fft_original)}")
            raise ValueError("Mismatch between positive_freqs and positive_fft lengths after masking in original data.")

        # Normalize power spectrum for original data
        sum_power_original = np.sum(positive_fft_original)
        if sum_power_original == 0:
            logging.warning("Sum of power spectrum for original data is zero. Cannot normalize.")
            power_spectrum_original = positive_fft_original
        else:
            power_spectrum_original = positive_fft_original / sum_power_original
        logging.debug(f"Power spectrum for original data normalized. Sum of power_spectrum: {np.sum(power_spectrum_original)}")

        # Calculate periods for original data
        periods_in_days_original = 1 / positive_freqs_original  # Periods in days
        periods_in_years_original = periods_in_days_original / 365.25  # Periods in years
        logging.debug(f"Periods calculated for original data. periods_in_years length: {len(periods_in_years_original)}")

        # Time series plot for original data with abbreviated month names
        ts_plot_original = hv.Curve(year_data_original, 'time', 'tp').opts(
            title=f'2010 Precipitation Time Series - Lon: {lon:.2f}, Lat: {lat:.2f}',
            width=600,
            height=300,
            tools=['hover'],
            logy=True,
            ylim=(1e-4, 1e3),
            xlabel='Month',
            ylabel='Precipitation (mm)',
            # xformatter='%b'  # Abbreviated month names
        )
        logging.debug("Original time series plot created.")

        # Add 30% threshold line to original time series
        try:
            threshold_30 = all_th.sel(threshold=30, method='nearest')
            threshold_value_original = threshold_30.sel(longitude=lon, latitude=lat, method='nearest').values
            logging.info("The value of threshold_30 for original data: %s", threshold_value_original)
        except KeyError:
            logging.warning("Threshold value 30 not found in all_th data for original data.")
            threshold_value_original = None

        if threshold_value_original is not None:
            threshold_line_original = hv.HLine(threshold_value_original, name='30% Threshold Line').opts(
                color='red',
                line_dash='dashed',
                line_width=2,
                show_legend=True
            )
            ts_plot_original = ts_plot_original * threshold_line_original
            logging.debug("Threshold line added to original time series plot.")

        # First Fourier Power Spectrum Plot (Period from 0 to 10 years) for original data
        fft_plot_original = hv.Curve((periods_in_years_original, power_spectrum_original), 'Period (Years)', 'Power Spectrum').opts(
            title=f'Fourier Power Spectrum (0-10 Years) - Original Data - Lon: {lon:.2f}, Lat: {lat:.2f}',
            width=600,
            height=300,
            tools=['hover'],
            xlim=(0, 10),  # Extended to 10 years
            ylim=(0, 8e-3)
        )
        logging.debug("First Fourier power spectrum plot for original data created.")

        # Second Fourier Power Spectrum Plot (Frequency between 0 and 0.5) for original data
        mask_freq_original = (positive_freqs_original >= 0.0) & (positive_freqs_original <= 0.5)
        frequencies_original = positive_freqs_original[mask_freq_original]
        power_spectrum_freq_original = power_spectrum_original[mask_freq_original]
        logging.debug(
            f"Frequency masking applied to original data. frequencies length: {len(frequencies_original)}, power_spectrum_freq length: {len(power_spectrum_freq_original)}")

        # Ensure that frequencies and power_spectrum_freq have the same length
        if len(frequencies_original) != len(power_spectrum_freq_original):
            logging.error(f"Mismatch in frequencies and power_spectrum_freq lengths for original data: {len(frequencies_original)} vs {len(power_spectrum_freq_original)}")
            raise ValueError("Mismatch between frequencies and power_spectrum_freq lengths for original data.")

        # Create the plot without additional smoothing for original data
        fft_plot2_original = hv.Curve((frequencies_original, power_spectrum_freq_original), 'Frequency (1/Days)', 'Power Spectrum F').opts(
            title=f'Fourier Power Spectrum (Frequency 0-0.5 1/Days) - Original Data - Lon: {lon:.2f}, Lat: {lat:.2f}',
            width=600,
            height=300,
            tools=['hover'],
            xlim=(0, 0.5),
            ylim=(0, 8e-3)
        )
        logging.debug("Second Fourier power spectrum plot for original data created.")

        # 处理生成数据的时间序列

        # Convert time to pandas datetime for better handling



        # Add 30% threshold line to generated time series
        try:
            threshold_30_gen = all_th.sel(threshold=30, method='nearest')
            threshold_value_generated = threshold_30_gen.sel(longitude=lon, latitude=lat, method='nearest').values
            logging.info("The value of threshold_30 for generated data: %s", threshold_value_generated)
        except KeyError:
            logging.warning("Threshold value 30 not found in all_th data for generated data.")
            threshold_value_generated = None

        if threshold_value_generated is not None:
            # threshold_line_generated = hv.HLine(threshold_value_generated, name='30% Threshold Line').opts(
            #     color='red',
            #     line_dash='dashed',
            #     line_width=2,
            #     show_legend=True
            # )
            # ts_plot_generated = ts_plot_generated * threshold_line_generated
            logging.debug("Threshold line added to generated time series plot.")



        return pn.Column(
            ts_plot_original,
            fft_plot_original,
            fft_plot2_original,
        )

    except Exception as e:
        logging.error(f"Error processing click: {e}")
        placeholder = hv.Text(0.5, 0.5, "An error occurred, please check the logs").opts(
            fontsize=20, align='center', text_align='center'
        )
        return pn.Column(placeholder, placeholder, placeholder, placeholder, placeholder, placeholder)


# 5. Create heatmaps including the newly added gen_wet and gen_duration
maps = {}
maps['Duration'] = plot_map(duration, 'Duration')
maps['Dot'] = plot_map(dot, 'Dot')
maps['Wet'] = plot_map(wet, 'Wet')
maps['gen_duration'] = plot_map(gen_duration, 'Gen_Duration')  # 新增的gen_duration热度图
maps['gen_wet'] = plot_map(gen_wet, 'Gen_Wet')  # 新增的gen_wet热度图
# maps['AR1_JAS'] = plot_map(ar1_JAS, 'AR1_JAS')
# maps['AR1_DJF'] = plot_map(ar1_DJF, 'AR1_DJF')
maps['Year_Power'] = plot_map(power, 'Year_Power')
maps['Power_hour'] = plot_map(power_hour, 'Power_hour')
maps['power_3days'] = plot_map(power_3days, 'power_3days')
maps['AR_Coefficient'] = plot_map(AR_Coefficient, 'AR_Coefficient')
maps['AR_C'] = plot_map(AR_C, 'AR_C')


# 6. Define dynamic plotting function to respond to click events
@pn.depends(click_data.param.data)
def dynamic_plot(data):
    x, y, source_map = data
    if x is not None and y is not None and source_map:
        logging.info(f"Handling click for '{source_map}' map at lon: {x}, lat: {y}")
        return process_click(x, y, source_map)
    else:
        # Return placeholders
        placeholder1 = hv.Text(0.5, 0.5, "请点击地图上的一个格点").opts(
            fontsize=20, align='center', text_align='center'
        )
        placeholder2 = hv.Text(0.5, 0.5, "请点击地图上的一个格点").opts(
            fontsize=20, align='center', text_align='center'
        )
        placeholder3 = hv.Text(0.5, 0.5, "请点击地图上的一个格点").opts(
            fontsize=20, align='center', text_align='center'
        )
        placeholder4 = hv.Text(0.5, 0.5, "请点击地图上的一个格点").opts(
            fontsize=20, align='center', text_align='center'
        )
        placeholder5 = hv.Text(0.5, 0.5, "请点击地图上的一个格点").opts(
            fontsize=20, align='center', text_align='center'
        )
        placeholder6 = hv.Text(0.5, 0.5, "请点击地图上的一个格点").opts(
            fontsize=20, align='center', text_align='center'
        )
        return pn.Column(placeholder1, placeholder2, placeholder3,
                         placeholder4, placeholder5, placeholder6)


# 7. Arrange heatmaps into columns
# First column: five heatmaps (Duration, Dot, Wet, Gen_Duration, Gen_Wet)
first_column = pn.Column(
    maps['Duration'],
    maps['Dot'],
    maps['Wet'],
    maps['gen_duration'],  # 添加gen_duration
    maps['gen_wet']  # 添加gen_wet
)

# Second column: dynamic plots (time series and FFT plots)
second_column = pn.Column(
    dynamic_plot
)

# Third column: four heatmaps (AR1_JAS, AR1_DJF, Year_Power, power_3days)
third_column = pn.Column(
    maps['AR_Coefficient'],
    maps['AR_C'],
    maps['Year_Power'],
    maps['Power_hour']
)

# Combine into three-column layout
layout = pn.Row(
    first_column,  # First column: five heatmaps
    second_column,  # Second column: time series and FFT plots
    third_column  # Third column: four heatmaps
)

# 8. Start the Panel application
if __name__ == '__main__':
    logging.info("Starting Panel application...")
    pn.serve(layout, show=True, port=6088)
