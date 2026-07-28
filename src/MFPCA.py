import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from skfda import FDataGrid
from scipy.linalg import eigh
from skfda.preprocessing.dim_reduction import FPCA

def build_fdata_grid(smoothed_df, sensor_cols, unit_col="unit_number",
                      cycle_col="t_registered", n_grid=100):
    dense_grid = np.linspace(0.0, 1.0, n_grid)
    units = smoothed_df[unit_col].unique()

    data_array = np.zeros((len(units), n_grid, len(sensor_cols)))

    for i, uid in enumerate(units):
        sub = smoothed_df[smoothed_df[unit_col] == uid].sort_values(cycle_col)
        t = sub[cycle_col].to_numpy()

        for j, sensor in enumerate(sensor_cols):
            y = sub[sensor].to_numpy()
            interp = interp1d(t, y, kind="linear", bounds_error=False,
                               fill_value="extrapolate")
            data_array[i, :, j] = interp(dense_grid)

    fd_multi = FDataGrid(
        data_matrix=data_array,
        grid_points=dense_grid,
        argument_names=["t_registered"],
        coordinate_names=sensor_cols,
    )

    return fd_multi, units



def fit_univariate_fpca_per_sensor(fd_multi, sensor_cols, n_components=5):
    scores_list = []
    fpca_list = []

    for j, sensor in enumerate(sensor_cols):
        fd_j = fd_multi.coordinates[j]
        fpca_j = FPCA(n_components=n_components)
        fpca_j.fit(fd_j)
        scores_j = fpca_j.transform(fd_j)

        scores_list.append(scores_j)
        fpca_list.append(fpca_j)

    Xi = np.hstack(scores_list)
    return Xi, fpca_list


def compute_mfpca(Xi, n_mfpc=None, variance_threshold=0.90, top_k_report=10):
    N = Xi.shape[0]
    Z_hat = (Xi.T @ Xi) / (N - 1)

    eigenvalues, eigenvectors = eigh(Z_hat)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues[eigenvalues > 0].sum()
    exp_var = eigenvalues / total_var * 100
    cum_var = exp_var.cumsum()

    k_report = min(top_k_report, len(eigenvalues))
    exp_var_df = pd.DataFrame({
        "mfpc": np.arange(1, k_report + 1),
        "explained_var_pct": exp_var[:k_report],
        "cumulative_pct": cum_var[:k_report],
    })

    if n_mfpc is None:
        n_mfpc = int(np.searchsorted(cum_var, variance_threshold * 100) + 1)
        n_mfpc = min(n_mfpc, len(eigenvalues))

    C_hat = eigenvectors[:, :n_mfpc]
    rho_scores = Xi @ C_hat

    return rho_scores, eigenvalues, C_hat, exp_var_df



def run_mfpca(smoothed_df, sensor_cols, unit_col="unit_number", cycle_col="t_registered",
              n_grid=100, n_components=5, n_mfpc=None, variance_threshold=0.90):
    fd_multi, units = build_fdata_grid(
        smoothed_df, sensor_cols, unit_col, cycle_col, n_grid
    )
    Xi, fpca_list = fit_univariate_fpca_per_sensor(
        fd_multi, sensor_cols, n_components
    )
    rho_scores, eigenvalues, C_hat, exp_var_df = compute_mfpca(
        Xi, n_mfpc, variance_threshold
    )
    print(exp_var_df)
    return rho_scores, exp_var_df
