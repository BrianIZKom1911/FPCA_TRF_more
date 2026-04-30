#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
#%% Functions for apparent temperature metrics
# Source: NOAA's National Weather Service: https://www.wpc.ncep.noaa.gov/html/calc.shtml
def heat_index(temp, RH):
    """
    Calculate the heat index given temperature (T) in Fahrenheit and relative humidity (RH) in percentage.
    The formula is based on the NOAA's National Weather Service calculation.
    """
    T = temp*(9/5) + 32 # Convert Celsius to Fahrenheit
    HI1 = 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (RH*0.094)) # Simple formula for moderate condition
    HI = (-42.379 + 2.04901523*T + 10.14333127*RH - 0.22475541*T*RH 
          -0.00683783* T**2 - 0.05481717* RH**2 + 0.00122874* T**2 *RH 
          +0.00085282*T* RH**2 - 0.00000199* T**2 * RH**2)
    if T <= 68:
        return T
    elif (HI1+T) < 160:
        return HI1
    else:
        if RH < 13 and 80 < T < 112:
            adj = - ((13-RH)/4) * np.sqrt((17-abs(T-95))/17)
        elif RH > 85 and 80 < T < 87:
            adj = ((RH-85)/10) * ((87-T)/5)
        else:
            adj = 0.0
        return HI + adj

# 2) NWS Wind Chill = x(T, V)
def wind_chill(temp, wsp):
    T = temp*(9/5) + 32 # Convert Celsius to Fahrenheit
    V = wsp * 2.237 # Convert m/s to mph
    if T <= 68 and V >= 3: 
        # I change the threshold to 68F because I am not only worried about wind chill in cold conditions 
        # but wind's cooling effect in warm conditions as well. 
        WC = 35.74 + 0.6215*T - 35.75*(V**0.16) + 0.4275*T*(V**0.16)
        return WC
    else:
        return T

# 3) New Heat Index
def new_NWS_hi(temp, RH, wsp):
    T = temp*(9/5) + 32 # Convert Celsius to Fahrenheit
    HI = heat_index(temp, RH)
    WC = wind_chill(temp, wsp)
    return (HI+WC-T-32)*(5/9) # Convert back to Celsius

# Source: Bureau of Meteorology, Australia: http://www.bom.gov.au/info/thermal_stress/
def apparent_temp(temp, RH, wsp):
    vp = RH/100 * 6.105 * np.exp(17.27*temp/(237.7+temp)) # Vapor pressure in hPa
    AT = temp + 0.33*vp - 0.70*wsp - 4.00
    return AT
#%% Local weighted linear regression
# This function deviates from a canonical LOWESS as it
# (1) trims extremes inside the local method; (2) uses ridge regularization.
def local_linear_knn(x, y, k, x_grid, lower_q=0.0, onesided=True, lambda_ridge=0.0):
    """
    Perform local linear regression using k-NN and Epanechnikov kernel weighting (09/02)
    
    Parameters:
    r (array-like): The independent variable
    y (array-like): The dependent variable
    k (int): Number of nearest neighbors to consider
    x_grid (array-like): Points at which to estimate the conditional expectation function
    lower_q (float in [0,1)): optional, lower quantile to drop within each neighborhood
    onesided (bool): optional, if True, also trim the upper quantile defined by 1-lower_q 
    lambda_ridge (float): optional, Ridge penalty

    Returns:
    np.ndarray: Estimated conditional expectation function at x_grid E[Y|X=x_grid]
    """
    x_norm = x.reshape(-1, 1)
    x_grid_norm = x_grid.reshape(-1, 1)
    
    # Find k-NN for each point in x_grid
    nbrs = NearestNeighbors(n_neighbors=k).fit(x_norm)
    distances, indices = nbrs.kneighbors(x_grid_norm)
    # Initialize output
    y_pred = np.zeros_like(x_grid)

    for i in range(len(x_grid)):
        # Find k-NN data points
        x_nbrs = x[indices[i]]
        y_nbrs = y[indices[i]]
        # Compute Epanechnikov weights
        dist_max = distances[i, -1] # The critical bandwidth for x_grid[i] (distance to its k-th nearest neighbor)
        u = distances[i] / dist_max # Normalized distances
        weights = 0.75*(1 - u**2) * (np.abs(u) <= 1) # Epanechnikov kernel weights

        # ---- trimming by neighborhood percentile ----
        if lower_q > 0:
            upper_q = 1 - lower_q
            ylb = np.quantile(y_nbrs, lower_q); yub = np.quantile(y_nbrs, upper_q)
            mask = y_nbrs >= ylb if onesided else (y_nbrs>=ylb) & (y_nbrs<=yub)
            x_nbrs = x_nbrs[mask]
            y_nbrs = y_nbrs[mask]
            weights = weights[mask]
        
        # Weighted least squares for local linear fit at x_grid[i]
        X = np.column_stack([np.ones_like(x_nbrs), x_nbrs - x_grid[i]])
        X_w = np.sqrt(weights)[:, None] * X
        y_w = np.sqrt(weights) * y_nbrs
        # Regularization
        beta = np.linalg.solve(
            X_w.T @ X_w + lambda_ridge * np.eye(X.shape[1]),
            X_w.T @ y_w
        )
        y_pred[i] = beta[0] # intercept = fitted value at x_grid[i]
    
    return y_pred
#%% Code within this block runs only when the file is executed directly by the interpreter but not when imported
if __name__ == "__main__":
    import gc
    import os
    
    # Import and process data
    url = "https://raw.githubusercontent.com/BrianIZKom1911/electricload_CTRF/master/data_clean/NC_main.csv"
    data = pd.read_csv(url, index_col=None)
    # Adjust data type
    data['datetime_UTC'] = pd.to_datetime(data['datetime_UTC'])
    data['Year'] = data['Year'].astype(int)
    data['Month'] = data['Month'].astype(int)
    data['Day'] = data['Day'].astype(int)
    data['Hour'] = data['Hour'].astype(int)
    data['yyyymm'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
    # Truncate extremes
    dt1 = data[(data['temperature'] >= -10) & (data['temperature'] <= 40)].copy()
    del data; gc.collect() # free up memory
    dt1['new_heat_index'] = dt1.apply(lambda row: new_NWS_hi(row['temperature'], row['relative_humidity'], row['wind_speed']), axis=1)
    dt1['apparent_temperature'] = dt1.apply(lambda row: apparent_temp(row['temperature'], row['relative_humidity'], row['wind_speed']), axis=1)

    # I) Smooth every month's load data using local linear kNN
    # Find the average load for different metrics
    metric_names = ['temperature', 'new_heat_index', 'apparent_temperature']
    for i, metric in enumerate(metric_names):
        dt1[f'y_avg_{i}'] = np.nan
        
        # Loop over year-month to average load for each period
        # Constraints: For each month, trim the lower 1.5% of the load within the local nbds
        ## ridge penalty=0.1 to stabilize the local fit
        months = dt1['yyyymm'].unique()
        T = len(months) # 276 months

        for t in range(T):
            mon = months[t]
            dt_m = dt1.loc[dt1['yyyymm'] == mon, ['yyyymm', 'load', metric]]
            if len(dt_m) < 180: # skip months with too few data points
                continue
            
            # Local weighted linear rgr (kNN)
            r = dt_m[metric].values
            y = dt_m['load'].values
            kt = 10*np.sqrt(len(y)).astype(int) # Debatable (2)
            r0 = r.copy()
            y_avg = local_linear_knn(r, y, kt, r0, lower_q=0.015, onesided=False, lambda_ridge=0.1)
            
            # Store results
            dt1.loc[dt_m.index, f'y_avg_{i}'] = y_avg # assign output only to the rows that remain after filtering
    print("Done smoothing for monthly functions.")
    # Save the average results
    thisdir = os.path.dirname(__file__)
    savedir = os.path.abspath(os.path.join(thisdir, '..', 'output'))
    os.makedirs(savedir, exist_ok=True)
    cols_save = ['datetime_UTC', 'Year', 'Month', 'Day', 'Hour', 'yyyymm', 'load']+metric_names+[f'y_avg_{i}' for i in range(3)]
    dt1[cols_save].to_csv(os.path.join(savedir, '1_NC_avgload_month.csv'), index=False)
    
    # II) Smooth every year's load data using local linear kNN
    for i, metric in enumerate(metric_names):
        dt1[f'y_avg_{i}'] = np.nan
        
        # Loop over year to average load for each period
        years = dt1['Year'].unique()
        T = len(years) # 22 years
        for t in range(T):
            yr = years[t]
            dt_y = dt1.loc[dt1['Year'] == yr, ['Year', 'load', metric]]

            # Local weighted linear rgr (kNN)
            # Constraints: For each year, trim the lower 0.5% of the load within the local nbd
            ## No ridge penalty because the data is less noisy at the yearly level
            r = dt_y[metric].values
            y = dt_y['load'].values
            kt = 5*np.sqrt(len(y)).astype(int) # Debatable (2)
            r0 = r.copy()
            y_avg = local_linear_knn(r, y, kt, r0, lower_q=0.005, onesided=False, lambda_ridge=0.0)
            
            # Store results
            dt1.loc[dt_y.index, f'y_avg_{i}'] = y_avg # assign output only to the rows that remain after filtering
    print("Done smoothing for yearly functions.")
    # Save the average results
    cols_save = ['datetime_UTC', 'Year', 'Month', 'Day', 'Hour', 'load']+metric_names+[f'y_avg_{i}' for i in range(3)]
    dt1[cols_save].to_csv(os.path.join(savedir, '1_NC_avgload_year.csv'), index=False)
# End.