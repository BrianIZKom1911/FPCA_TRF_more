#%% # II) Fit polynomial basis to the monthly average load data
import gc
import os
import pandas as pd
import numpy as np
from numpy.polynomial.legendre import legval, legvander
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
#%%
def legendre_basis_rep(path_input, freq, col_y, col_x, hi=False):
    data = pd.read_csv(path_input, index_col=None)
    # Loop over year-month to fit polynomial basis
    # Version 1: standardize y in advance. Thus coefs are effect \times SD
    dt0 = data.dropna(subset=[col_y]).copy()
    del data; gc.collect()
    if hi:
        a = max(dt0[col_x].values); b = min(dt0[col_x].values)
    else:
        a = -10.0; b = 40.0 # Common support for all months

    mu_y = dt0[col_y].mean()
    s_y = dt0[col_y].std()
    dt0.loc[:, 'y_std'] = (dt0[col_y]-mu_y)/s_y # standardize y to improve numerical stability

    coefs_list = []
    periodmark = 'yyyymm' if freq=='Month' else freq
    periods = sorted(dt0[periodmark].unique())
    for i, prd in enumerate(periods):
        dt_p = dt0.loc[dt0[periodmark]==prd, [periodmark, 'y_std', col_x]].copy()
        # Legendre polynomials on [-1, 1]
        r = dt_p[col_x].values
        y_std = dt_p['y_std'].values
        r_nml = 2*(r-a)/(b-a)-1

        # Design matrix
        X_poly = legvander(r_nml, deg=99) # degrees from 0 to 99, M=100
        # Ridge regression to penalize large coefficients
        ridge = Ridge(alpha=1.0, fit_intercept=False) # debatable (3)
        model = ridge.fit(X_poly, y_std)
        coefs = model.coef_ # Estimated coefficients
        coefs_list.append(coefs)
    
    return np.array(coefs_list) # N by M array
#%%
thisdir = os.path.dirname(__file__)
savedir = os.path.abspath(os.path.join(thisdir, '..', 'output'))
inputfiles = ['1_NC_avgload_month.csv', '1_NC_avgload_year.csv', '2_NC_baseload_year.csv']
y_cols = ['y_avg_0', 'y_avg_1', 'y_base_1']

# 1a and 1b) Month, simple metrics
infe = inputfiles[0]
inpath = os.path.join(savedir, infe)
c_array = legendre_basis_rep(inpath, 'Month', y_cols[0], 'temperature')
np.savetxt(os.path.join(savedir, '1a_month_ridgecoefs100_ystd.csv'), c_array, delimiter=',')
c_array = legendre_basis_rep(inpath, 'Month', y_cols[1], 'new_heat_index', hi=True)
np.savetxt(os.path.join(savedir, '1b_month_ridgecoefs100_ystd.csv'), c_array, delimiter=',')

# 1a and 1b) Year, simple metrics for comparison
infe = inputfiles[1]
inpath = os.path.join(savedir, infe)
c_array = legendre_basis_rep(inpath, 'Year', y_cols[0], 'temperature')
np.savetxt(os.path.join(savedir, '1a_year_ridgecoefs100_ystd.csv'), c_array, delimiter=',')
c_array = legendre_basis_rep(inpath, 'Year', y_cols[1], 'new_heat_index', hi=True)
np.savetxt(os.path.join(savedir, '1b_year_ridgecoefs100_ystd.csv'), c_array, delimiter=',')

# 2) base TRFs
infe = inputfiles[2]
inpath = os.path.join(savedir, infe)
c_array = legendre_basis_rep(inpath, 'Year', y_cols[2], 'temperature')
np.savetxt(os.path.join(savedir, '2_year_ridgecoefs100_ystd.csv'), c_array, delimiter=',')
#.
#%% Complementary: Store the mean and SD of the outcome (All: 5)
def get_mean_sd(data, col_y):
    dt0 = data.dropna(subset=[col_y]).copy()
    mu_y = dt0[col_y].mean()
    s_y = dt0[col_y].std()
    return mu_y, s_y

rows = []
for infe in inputfiles:
    inpath = os.path.join(savedir, infe)
    dt = pd.read_csv(inpath, index_col=None)
    for y_col in y_cols:
        if y_col not in dt.columns:
            continue
        mu_y, s_y = get_mean_sd(dt, y_col)
        rows.append([infe, y_col, mu_y, s_y])

df_std = pd.DataFrame(rows, columns=['file', 'column', 'mu', 'sd'])
df_std.to_csv(os.path.join(savedir, 'df_std.csv'), index=None)
# End of script.