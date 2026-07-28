import numpy as np
import pandas as pd
from scipy.linalg import eigh
from skfda.representation.basis import BSplineBasis
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed

order = 4
knots = 20
basis = order + knots
domain_train = (0, 1)
lam_bounds = (-8,4)


REGULARIZATION = L2Regularization(LinearDifferentialOperator(2))
basis_train = BSplineBasis(domain_range = domain_train, n_basis=basis, order=order)

omega = REGULARIZATION.penalty_matrix(basis_train)


def registration(df):
    df["t_registered"] = df.groupby("unit_number")["cycles"].transform(
        lambda t: (t - 1) / (t.max() - 1)
    )
    return df

def build_unit_matrices(t, basis=basis_train, omega=omega):
    t = np.asarray(t)

    B = basis(t)[:, :, 0].T
    BtB = B.T @ B

    D, U = eigh(omega, b=BtB)

    return B, BtB, D, U

def transform_response(B, U, y):
    y = np.asarray(y)
    g = U.T @ (B.T @ y)
    residual_const = y @ y - g @ g
    return g, residual_const


def gcv_score(log_lam, D, g, residual_const, n):
    lam = np.exp(log_lam)
    shrink = 1 / (1 + lam * D)
    rss = residual_const + np.sum((g * lam * D * shrink) **2)
    dof = np.sum(shrink)

    denom = (1 - dof/n) **2
    return (rss/n)/denom


def cache_all_units(df, unit_col="unit_id", cycle_col="cycles",
                     min_points_ratio=2.0, n_jobs=-1):
    def _one(unit_id, group_df):
        cycle = np.sort(group_df[cycle_col].to_numpy().astype(float))
        n = len(cycle)

        if n < 5 or n < min_points_ratio * basis:
            return unit_id, {"error": f"n={n} not enough (N_BASIS={basis})"}
        if np.any(np.diff(cycle) == 0):
            return unit_id, {"error": "cycle is overlap"}

        try:
            B, BtB, D, U = build_unit_matrices(cycle)
        except np.linalg.LinAlgError as e:
            return unit_id, {"error": f"Failed: {e}"}

        return unit_id, {"B": B, "D": D, "U": U, "n": n, "cycle": cycle}

    groups = list(df.groupby(unit_col, sort=False))
    outputs = Parallel(n_jobs=n_jobs)(delayed(_one)(uid, g) for uid, g in groups)
    return dict(outputs)


def fleet_gcv_score(log_lam, unit_data_list):
    scores = [gcv_score(log_lam, D_i, g_i, rc_i, n_i)
              for D_i, g_i, rc_i, n_i in unit_data_list]
    return np.mean(scores)


def find_fleet_optimal_lambda(unit_data_list, log_lam_bounds=lam_bounds):
    result = minimize_scalar(
        fleet_gcv_score, bounds=log_lam_bounds, method='bounded',
        args=(unit_data_list,),
    )
    lam_star = np.exp(result.x)
    return lam_star, result.fun

def process_sensor(sensor_name, df, unit_cache, unit_col, cycle_col,
                    log_lam_bounds=lam_bounds):
    unit_data_list = []
    for unit_id, cache in unit_cache.items():
        if "error" in cache:
            continue

        y = (df.loc[df[unit_col] == unit_id]
               .sort_values(cycle_col)[sensor_name]
               .to_numpy())

        if len(y) != cache["n"] or np.any(np.isnan(y)):
            continue

        g, residual_const = transform_response(cache["B"], cache["U"], y)
        unit_data_list.append((cache["D"], g, residual_const, cache["n"]))

    if len(unit_data_list) == 0:
        return None, None

    lam_star, gcv_star = find_fleet_optimal_lambda(unit_data_list, log_lam_bounds)

    return lam_star, gcv_star


def lambda_finding(df, sensor_cols, unit_col="unit_number",
                              cycle_col="t_registered", min_points_ratio=2.0,
                              log_lam_bounds=lam_bounds, n_jobs=-1):
    unit_cache = cache_all_units(df, unit_col, cycle_col,
                                  min_points_ratio, n_jobs)

    summary_rows = []

    for sensor_name in sensor_cols:
        lam_star, gcv_star = process_sensor(
            sensor_name, df, unit_cache, unit_col, cycle_col, log_lam_bounds
        )
        summary_rows.append({
            "sensor": sensor_name,
            "lambda": lam_star,
            "gcv": gcv_star,
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df)
    return summary_df, unit_cache

def fit_coefficients(D, U, g, lam):
    shrink = 1.0 / (1.0 + lam * D)
    return U @ (g * shrink)

def smooth_predict(basis, coef, x_eval):
    x_eval = np.asarray(x_eval, dtype=float)
    Bx = basis(x_eval)[:, :, 0].T
    return Bx @ coef

def compute_smoothed_train(df, summary_df, unit_cache, sensor_cols,
                            unit_col="unit_number", cycle_col="t_registered"):
    lam_lookup = summary_df.set_index("sensor")["lambda"].to_dict()

    missing = [s for s in sensor_cols if s not in lam_lookup
               or lam_lookup[s] is None]
    if missing:
        raise ValueError(f"Invalid Lambda "
                          f"summary_df: {missing}")
    smoothed_df = df[[unit_col, cycle_col]].copy()
    for sensor_name in sensor_cols:
        smoothed_df[sensor_name] = np.nan

    coef_store = {}
    error_rows = []

    for unit_id, cache in unit_cache.items():
        coef_store[unit_id] = {}

        if "error" in cache:
            for sensor_name in sensor_cols:
                error_rows.append({
                    "unit_id": unit_id, "sensor": sensor_name,
                    "error": cache["error"],
                })
            continue

        mask = smoothed_df[unit_col] == unit_id
        group_df = df.loc[mask].sort_values(cycle_col)
        cycle = group_df[cycle_col].to_numpy()

        for sensor_name in sensor_cols:
            y = group_df[sensor_name].to_numpy()

            if len(y) != cache["n"] or np.any(np.isnan(y)):
                error_rows.append({
                    "unit_id": unit_id, "sensor": sensor_name,
                    "error": "y mismatch hoặc NaN",
                })
                continue

            g, _ = transform_response(cache["B"], cache["U"], y)
            lam = lam_lookup[sensor_name]
            coef = fit_coefficients(cache["D"], cache["U"], g, lam)

            coef_store[unit_id][sensor_name] = {
                "coef": coef, "basis": basis_train,
            }

            y_smooth = smooth_predict(basis_train, coef, cycle)
            smoothed_df.loc[group_df.index, sensor_name] = y_smooth

    error_log = pd.DataFrame(error_rows)
    print('Smoothed Dataframe :')
    print(smoothed_df)
    return smoothed_df #,coef_store, error_log
