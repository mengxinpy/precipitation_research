import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


def plot_timeseries(dataarray, th, use_plotly=False):
    if 'time' not in dataarray.dims:
        raise ValueError("The provided DataArray does not contain a 'time' dimension.")

    # Convert the DataArray to a DataFrame
    df = dataarray.to_dataframe().reset_index()

    # Plot using seaborn or plotly
    if use_plotly:
        df['time'] = pd.to_datetime(df['time'])
        fig = px.line(df, x='time', y=dataarray.name)
        fig.add_hline(y=th.values, line=dict(color="red", width=3, opacity=0.5))
        fig.update_layout(yaxis_type="log")
        fig.write_html("F:/liusch/remote_project/climate_new/temp_fig/durations_time/plotly_timeseries.html")

    else:
        # Plot using Seaborn
        sns.lineplot(data=df, x='time', y=dataarray.name)
        plt.savefig("F:/liusch/remote_project/climate_new/temp_fig/durations_time/seaborn_timeseries.png")
        plt.close()


# Example usage:
# Assume 'da' is an xarray DataArray with a 'time' dimension
# plot_timeseries(da, use_plotly=True)  # Plot using Plotly
# plot_timeseries(da, use_plotly=False) # Plot using Seaborn


def calculate_event_durations(precipitation_array, percentile_th, mask_array):
    start_events = np.diff((precipitation_array > percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events = np.diff((precipitation_array > percentile_th.values).astype('int'), append=0, axis=0) == -1
    start_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), prepend=0, axis=0) == 1
    end_events_qt = np.diff((precipitation_array < percentile_th.values).astype('int'), append=0, axis=0) == -1
    durations = method_name(end_events, mask_array, precipitation_array, start_events)
    durations_qt = method_name(end_events_qt, mask_array, precipitation_array, start_events_qt)
        # plot_timeseries(dr.sel(longitude=lon, latitude=lat, method='nearest'), percentile_th.sel(longitude=lon, latitude=lat, method='nearest'), use_plotly=True)
    return durations, durations_qt


def method_name(end_events, mask_array, precipitation_array, start_events):
    durations = []
    for lat in range(precipitation_array.shape[1]):
        for lon in range(precipitation_array.shape[2]):
            if not mask_array[lat, lon]:
                continue
            start_indices = np.where(start_events[:, lat, lon])[0]
            end_indices = np.where(end_events[:, lat, lon])[0]

            event_durations = end_indices - start_indices + 1
            durations.extend(event_durations)
    durations = np.array(durations)
    return durations


# 自定义函数，将NaN和小于等于0的值替换为0.0001
def replace_values(x):
    if pd.isna(x) or x <= 0:
        return 0.0001
    else:
        return x


