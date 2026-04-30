# This script is used to plot the 3D scatterplots of (1) load vs temp vs RH, and
# (2) load vs temp vs wsp, points colored by month.
#%%
import os
import gc
import calendar
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize
import plotly.graph_objects as go
#%%
thisdir = os.path.dirname(__file__)
savedir = os.path.abspath(os.path.join(thisdir, '..', 'output'))
#%%
url = "https://raw.githubusercontent.com/BrianIZKom1911/electricload_CTRF/master/data_clean/NC_main.csv"
data = pd.read_csv(url, index_col=None)
data['datetime_UTC'] = pd.to_datetime(data['datetime_UTC'])
data['Year'] = data['Year'].astype(int)
data['Month'] = data['Month'].astype(int)
data['yyyymm'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
dt1 = data[(data['temperature'] >= -10) & (data['temperature'] <= 40)].copy()
del data; gc.collect()
#%%
# Set up
cmap = mpl.colormaps['turbo'].resampled(12)
norm = Normalize(vmin=0, vmax=12)
colors = [cmap(i/12) for i in range(12)]
month_order = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1] # Custom order for months
custom_colors = [colors[i] for i in month_order]
custom_colorscale = [
    [i/11, f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'] 
    for i, c in enumerate(custom_colors)
]
# Type 1. Two inputs, one output
def DRAW_FIG_PLOT1(
        filename, data, x_col, y_col, z_col, color_col, 
        x_title, y_title, z_title, leg_title, fig_title
    ):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=data[x_col],
        y=data[y_col],
        z=data[z_col],
        mode='markers',
        marker=dict(
            size=1,
            color=data[color_col],
            colorscale=custom_colorscale,
            cmin=1, # Jan
            cmax=12 # Dec
        ),
        #hovertext=,
        hoverinfo='x+y+z'
    ))
    # Manually add 12 dummy traces for legend
    for i in range(12):
        color_rgb = cmap(i)[:3]
        color_str = f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})'
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color=color_str, width=4),
            name=calendar.month_abbr[i+1], # Jan, Feb, ... Dec
            showlegend=True
        ))
    # Customize axes
    fig.update_layout(
        scene=dict(
            xaxis_title=x_title, #'Temperature (C)'
            yaxis_title=y_title, #'Relative Humidity (%)'
            zaxis_title=z_title #'Load (MW)'
        ),
        legend_title_text=leg_title, #'Month'
        margin=dict(l=0, r=0, b=0, t=30),
        height=800,
        title= fig_title #'Load vs Temperature and RH',
    )
    fig.write_html(os.path.join(savedir, filename))
#%% Figure 1 and 2
DRAW_FIG_PLOT1(
    filename='3dplot_fig1.html',
    data=dt1,
    x_col='temperature',
    y_col='relative_humidity',
    z_col='load',
    color_col='Month',
    x_title='Temperature (C)',
    y_title='Relative Humidity (%)',
    z_title='Load (MW)',
    leg_title='Month',
    fig_title='Load vs Temperature and RH'
)
# Similar code for load vs temp vs wsp
# Delete extreme windy data (wsp > 20 m/s)
dt2 = dt1[dt1['wind_speed'] <= 20].copy()
DRAW_FIG_PLOT1(
    filename='3dplot_fig2.html',
    data=dt2,
    x_col='temperature',
    y_col='wind_speed',
    z_col='load',
    color_col='Month',
    x_title='Temperature (C)',
    y_title='Wind Speed (m/s)',
    z_title='Load (MW)',
    leg_title='Month',
    fig_title='Load vs Temperature and Windspeed'
)
# End.