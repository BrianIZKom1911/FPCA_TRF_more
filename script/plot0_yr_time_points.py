###########################
#   Interactive 3D plot   # 
# Color gradient by month #
#%%
import gc
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import calendar  # to get month names
#%%
thisdir = os.path.dirname(__file__)
savedir = os.path.abspath(os.path.join(thisdir, '..', 'output'))
os.makedirs(savedir, exist_ok=True)
#%%
# Set external variables in the global scope
cmap = mpl.colormaps['turbo'].resampled(12)
norm = Normalize(vmin=0, vmax=12)
colors = [cmap(i/12) for i in range(12)]
month_order = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1] # Custom order for months
custom_colors = [colors[i] for i in month_order]
custom_colorscale = [
    [i/11, f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'] 
    for i, c in enumerate(custom_colors)
]
# Define a fixed no-return procedure
def DRAW_FIG_PLOT(filename, data, x_col, z_col, x_title, y_title, z_title, leg_title, fig_title):
    fig = go.Figure()
    months = data['yyyymm'].unique()

    fig.add_trace(go.Scatter3d(
        x=data[x_col], #'temperature' or some heat index
        y=data['yyyymm'].astype('category').cat.codes, # convert dates to numeric codes
        z=data[z_col],
        mode='markers',
        marker=dict(
            size=1,
            color=data['Month'], # Color by month
            colorscale=custom_colorscale,
            cmin=1, # January
            cmax=12 # December
            #opacity=0.7,
            #colorbar=dict(title='Month')
        ),
        hovertext=data['yyyymm'], # show date on hover
        hoverinfo='x+z+text'
    ))

    # Manually add 12 dummy traces for legend # to be consistent with line plot
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
            xaxis_title=x_title,
            yaxis=dict(
                title=y_title,
                tickvals=[0, 60, 120, 180, 240],
                ticktext=['2002', '2007', '2012', '2017', '2022']
            ),        
            zaxis_title=z_title
        ),
        legend_title_text=leg_title,
        margin=dict(l=0, r=0, b=0, t=30),
        height=800,
        title=fig_title
    )
    fig.write_html(os.path.join(savedir, filename))

#%% Import data and Call
url = "https://raw.githubusercontent.com/BrianIZKom1911/electricload_CTRF/master/data_clean/NC_main.csv"
data = pd.read_csv(url, index_col=None)
data['datetime_UTC'] = pd.to_datetime(data['datetime_UTC'])
data['Year'] = data['Year'].astype(int)
data['Month'] = data['Month'].astype(int)
data['yyyymm'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
dt1 = data[(data['temperature'] >= -10) & (data['temperature'] <= 40)].copy()
del data; gc.collect()

DRAW_FIG_PLOT(
    filename='scatterplot_1.html',
    data=dt1,
    x_col='temperature',
    z_col='load',
    x_title='Temperature (C)',
    y_title='Year-Month',
    z_title='Load (MW)',
    leg_title='Month',
    fig_title='3D Scatterplot: Temperature vs Load by Month'
)
# New heat index
from S11_temp_months import new_NWS_hi
dt1['new_heat_index'] = dt1.apply(lambda row: new_NWS_hi(row['temperature'], row['relative_humidity'], row['wind_speed']), axis=1)

DRAW_FIG_PLOT(
    filename='scatterplot_1b.html',
    data=dt1,
    x_col='new_heat_index',
    z_col='load',
    x_title='Feel-like temperature (C)',
    y_title='Year-Month',
    z_title='Load (MW)',
    leg_title='Month',
    fig_title='Heat Index vs Load by Month'
)
# End of script.