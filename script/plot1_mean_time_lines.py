# This script is used to plot smoothed load vs temperature (viewed as TRF or CEF)
# Variant 1: temperature could be replaced by some heat index
# Variant 2: For base TRF in CTRF models, the difference is solely in the input file, i.e., 
# how the load is averaged, while all the following part stays the same.
# In addition, the plotting function could admit monthly or yearly frequency, but base TRF 
# should be yearly only.
#%%
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
#%%
def DRAW_FIG_PLOT2(
        savepath, data, freq, x_col, z_col, 
        x_title, y_title, z_title, leg_title, fig_title
    ):
    fig = go.Figure()

    periodmark = 'yyyymm' if freq=='Month' else freq
    periods = sorted(data[periodmark].unique())
    nt = len(periods)

    cmap1 = mpl.colormaps['turbo'].resampled(12)
    cmap2 = mpl.colormaps['turbo'].resampled(nt)
    
    for i, prd in enumerate(periods):
        dt_p = data[data[periodmark] == prd].copy()
        dt_p = dt_p.sort_values(x_col) # sort by temperature

        if freq == 'Month':
            # Plot a sorted line for each month
            month_idx = int(dt_p[freq].iloc[0]) - 1 # get month index for color (0-11)
            color_rgb = cmap1(month_idx)[:3]
            color_str = f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})'
        else:
            # Yearly lines: color by year index (first year -> start of cmap2, last year -> end)
            norm_idx = i / (nt - 1) if nt > 1 else 0
            color_rgb = cmap2(norm_idx)[:3]
            color_str = f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})'

        fig.add_trace(go.Scatter3d(
            x=dt_p[x_col],
            y=[prd]*len(dt_p),
            z=dt_p[z_col],
            mode='lines',
            line=dict(color=color_str, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Legend
    if freq == 'Month':
        for i in range(12):
            color_rgb = cmap1(i)[:3]
            color_str = f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})'
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='lines',
                line=dict(color=color_str, width=4),
                name=calendar.month_abbr[i+1], # Jan, Feb, ... Dec
                showlegend=True
            ))
    else:
        # One colorbar legend for the year span
        year_min, year_max = periods[0], periods[-1]
        colors = [
            f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'
            for c in cmap2(np.linspace(0, 1, nt))
        ]
        colorscale = [[i/(nt-1), col] for i, col in enumerate(colors)]

        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                color=[year_min, year_max],
                colorscale=colorscale,
                cmin=year_min,
                cmax=year_max,
                colorbar=dict(title='Year'),
                size=0
            ),
            showlegend=False
        ))

    # Customize axes
    years = sorted(set(data['Year']))
    fig.update_layout(
        scene=dict(
            xaxis_title=x_title,
            yaxis=dict(
                title=y_title,
                tickvals=[data[data['Year'] == y][periodmark].iloc[0] for y in years],
                ticktext=[str(y) for y in years]
            ),        
            zaxis_title=z_title 
        ),
        legend_title_text=leg_title, 
        margin=dict(l=0, r=0, b=0, t=30),
        height=800,
        title=fig_title 
    )
    fig.write_html(savepath)

#%% Average load
filetosave1 = os.path.join(savedir, '1a_month', 'line_plot_1.html')
filetosave1b = os.path.join(savedir, '1b_month', 'line_plot_1b.html')
if os.path.exists(filetosave1) and os.path.exists(filetosave1b):
    print("Skip month smoothed data...")
else:
    dta = pd.read_csv(os.path.join(savedir, '1_NC_avgload_month.csv'), index_col=None)
    dta['datetime_UTC'] = pd.to_datetime(dta['datetime_UTC'])
    dta['Year'] = dta['Year'].astype(int)
    dta['Month'] = dta['Month'].astype(int)

if os.path.exists(filetosave1):
    print(f"{filetosave1} already exists.")
else: # Call
    DRAW_FIG_PLOT2(
        savepath=filetosave1,
        data=dta,
        freq='Month',
        x_col='temperature',
        z_col='y_avg_0',
        x_title='Temperature (C)',
        y_title='Year',
        z_title='Load (MW)',
        leg_title='Month',
        fig_title='Temperature vs Smoothed Load by Month'
    )
    print(f"Done drawing {filetosave1}.")
if os.path.exists(filetosave1b):
    print(f"{filetosave1b} already exists.")
else:
    DRAW_FIG_PLOT2(
        savepath=filetosave1b,
        data=dta,
        freq='Month',
        x_col='new_heat_index',
        z_col='y_avg_1',
        x_title='Feel-like temperature (C)',
        y_title='Year',
        z_title='Load (MW)',
        leg_title='Month',
        fig_title='Heat Index vs Smoothed Load by Month'
    )
    print(f"Done drawing {filetosave1b}.")
#%% Base TRF
filetosave2 = os.path.join(savedir, '2_year', 'line_plot_2.html')
if os.path.exists(filetosave2):
    print(f"{filetosave2} already exists.")
else:
    dta = pd.read_csv(os.path.join(savedir, '2_NC_baseload_year.csv'), index_col=None)
    dta['datetime_UTC'] = pd.to_datetime(dta['datetime_UTC'])
    dta['Year'] = dta['Year'].astype(int)
    DRAW_FIG_PLOT2(
        savepath=filetosave2,
        data=dta,
        freq='Year',
        x_col='temperature',
        z_col='y_base_1',
        x_title='Temperature (C)',
        y_title='Year',
        z_title='Base load (MW)',
        leg_title='Year',
        fig_title='Base TRF by Year'
    )
    print(f"Done drawing {filetosave2}.")
# End of script.