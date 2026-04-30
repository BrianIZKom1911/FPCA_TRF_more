# This script uses nonparametric estimation (smoothing) on the CTRF model:
# y=w'g(r)+e. Assuming E(e|r,w)=0 at fixed year
# Method: In each neighborhood of r, regress y on (1, w_1, ..., w_K); save the
# estimated coefficient \hat{g}_0 of the constant one. 
# So we get a set of pairs {(r_t, \hat{g}_0t)}.
#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
#%% Local weighted multiple linear regression (LoWeSS)
# I) kNN
def local_multilinear_knn(r_name, x_data, y, k, r_grid, lower_q=0.0, onesided=True, lambda_ridge=0.0):
    # Normalize temperature (r) to get distance for kNN
    r = x_data[r_name].values
    r_norm = r.reshape(-1, 1)
    r_grid_norm = r_grid.reshape(-1, 1)
    # Standardize covariates (w's)
    w_cols = [col for col in x_data.columns if col != r_name]
    w = x_data[w_cols].values
    w_stand = (w - w.mean(axis=0)) / w.std(axis=0)
    
    # Find k-NN for each point in r_grid
    nbs = NearestNeighbors(n_neighbors=k).fit(r_norm)
    distances, indices = nbs.kneighbors(r_grid_norm)
    # Initialize output
    y_pred = np.zeros_like(r_grid)

    for i in range(len(r_grid)):
        r_nbs = r[indices[i]]
        w1_nbs = w_stand[indices[i]]
        y_nbs = y[indices[i]]
        # Compute Epanechnikov weights
        dist_max = distances[i, -1] # The critical bandwidth for r_grid[i]
        u = distances[i] / dist_max # Normalized distances
        weights = 0.75*(1 - u**2) * (np.abs(u) <= 1) # Epanechnikov kernel weights
        
        # ---- trimming by neighborhood percentile ----
        if lower_q > 0:
            upper_q = 1 - lower_q
            ylb = np.quantile(y_nbs, lower_q); yub = np.quantile(y_nbs, upper_q)
            mask = y_nbs >= ylb if onesided else (y_nbs>=ylb) & (y_nbs<=yub)
            r_nbs = r_nbs[mask].copy() # avoid SettingWithCopyWarning
            w1_nbs = w1_nbs[mask].copy()
            y_nbs = y_nbs[mask].copy()
            weights = weights[mask].copy()
        
        # Weighted least squares for local linear fit at r_grid[i]
        X = np.column_stack([np.ones_like(r_nbs), w1_nbs]) # Add intercept
        X_w = np.sqrt(weights)[:, None] * X
        y_w = np.sqrt(weights) * y_nbs
        # Regularization
        beta = np.linalg.solve(
            X_w.T @ X_w + lambda_ridge * np.eye(X.shape[1]),
            X_w.T @ y_w
        )
        y_pred[i] = beta[0] # intercept = fitted value at r_grid[i]
    
    return y_pred
# II) Fixed bandwidth
# Used on simulation temperature range with intrapolation into the dataset 
def local_multilinear_bdw(r_name, x_data, y, r_grid, bdw, k_min=120, lower_q=0.0, onesided=True, lambda_ridge=0.0):
    # Standardize covariates (w's)
    w_cols = [col for col in x_data.columns if col != r_name]
    w = x_data[w_cols].values
    w_stand = (w - w.mean(axis=0)) / w.std(axis=0)

    # Find the neighbors in B(r0, b) for each r0 in r_grid
    r = x_data[r_name].values
    y_pred = np.zeros_like(r_grid)
    for i, r0 in enumerate(r_grid):
        mask = np.abs(r - r0) <= bdw
        r_nbs = r[mask]
        w1_nbs = w_stand[mask]
        y_nbs = y[mask]
        if len(r_nbs) < k_min: # If too few neighbors, skip
            y_pred[i] = np.nan
            continue
        # Compute Epanechnikov weights
        u = np.abs(r_nbs - r0) / bdw
        weights = 0.75*(1 - u**2) * (np.abs(u) <= 1) # Epanechnikov kernel weights

        # ---- trimming by neighborhood percentile ----
        if lower_q > 0:
            upper_q = 1 - lower_q
            ylb = np.quantile(y_nbs, lower_q); yub = np.quantile(y_nbs, upper_q)
            mask = y_nbs >= ylb if onesided else (y_nbs>=ylb) & (y_nbs<=yub)
            r_nbs = r_nbs[mask].copy() # avoid SettingWithCopyWarning
            w1_nbs = w1_nbs[mask].copy()
            y_nbs = y_nbs[mask].copy()
            weights = weights[mask].copy()

        # Weighted least squares for local linear fit at r0
        X = np.column_stack([np.ones_like(r_nbs), w1_nbs]) # Add intercept
        X_w = np.sqrt(weights)[:, None] * X
        y_w = np.sqrt(weights) * y_nbs
        beta = np.linalg.solve(
            X_w.T @ X_w  + lambda_ridge * np.eye(X.shape[1]), 
            X_w.T @ y_w
        )
        y_pred[i] = beta[0] # intercept = fitted value at r0
    return y_pred

#%%
if __name__ == "__main__":
    import gc
    import os
    url = "https://raw.githubusercontent.com/BrianIZKom1911/electricload_CTRF/master/data_clean/NC_main.csv"
    data = pd.read_csv(url, index_col=None)
    # Adjust data type
    data['datetime_UTC'] = pd.to_datetime(data['datetime_UTC'])
    data['Year'] = data['Year'].astype(int)
    data['Month'] = data['Month'].astype(int)
    data['Day'] = data['Day'].astype(int)
    data['Hour'] = data['Hour'].astype(int)
    # Truncate extreme temperatures
    #dt1 = data[(data['temperature'] >= -10) & (data['temperature'] <= 40)].copy()
    dt1 = data.copy()
    # Winsorize extreme participation and windspeed
    dt1.loc[dt1['precipitation'] > 1, 'precipitation'] = 1
    dt1.loc[dt1['wind_speed'] > 20, 'wind_speed'] = 20
    del data; gc.collect() # free up memory
    # I) Smooth every year's load data using local regression method
    dt1['y_base_1'] = np.nan
    dt1['y_sim_2'] = np.nan
    # Loop over year to get base load for each year
    for year in dt1['Year'].unique():
        mask = dt1['Year'] == year
        dtx = dt1.loc[mask, ['temperature', 'precipitation', 'relative_humidity', 'wind_speed', 'skycover']]
        y = dt1.loc[mask, 'load']
        # Select complete cases for the year
        dtx_comp = dtx.dropna()
        y_comp = y.loc[dtx_comp.index].values

        # Do LoWeSS with kNN
        kt = 5*np.sqrt(len(y_comp)).astype(int)
        r0 = dtx_comp['temperature'].values
        y_base1 = local_multilinear_knn('temperature', dtx_comp, y_comp, kt, r0, lower_q=0.005, onesided=True, lambda_ridge=0.1)
        dt1.loc[dtx_comp.index, 'y_base_1'] = y_base1
        
        # Do LoWeSS with fixed bandwidth
        T_bdw = 2.5
        r_sim = np.linspace(dtx_comp['temperature'].min(), dtx_comp['temperature'].max(), 100)
        y_sim2 = local_multilinear_bdw('temperature', dtx_comp, y_comp, r_sim, T_bdw, k_min=240, lower_q=0.005, onesided=True, lambda_ridge=0.01)
        dt1.loc[dtx_comp.index, 'y_sim_2'] = np.interp(dtx_comp['temperature'], r_sim, y_sim2)
        # Interpolate y_sim based on temperature
        dt_year = dt1[mask].sort_values('temperature')
        dt_year['y_sim_2'] = dt_year['y_sim_2'].interpolate(method='linear') 
        ## without limit_direction='both', this does not extrapolate beyond the range of temp whose nbs>=k_min
        dt1.loc[dt_year.index, 'y_sim_2'] = dt_year['y_sim_2']
    
    print("Done smoothing for yearly functions.")

    thisdir = os.path.dirname(__file__)
    savedir = os.path.abspath(os.path.join(thisdir, '..', 'output'))
    cols_save = ['datetime_UTC', 'Year', 'Month', 'Day', 'Hour', 'temperature', 'load', 'y_base_1', 'y_sim_2']
    dt1[cols_save].to_csv(os.path.join(savedir, '2_NC_baseload_year.csv'), index=False)
# End.