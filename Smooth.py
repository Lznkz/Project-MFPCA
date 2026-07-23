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

def build_unit_matrices(t, basis=basis_train, omega=omega):
    """
    t : array of t_registered and we run it for all unit this function ...
    """
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
    """
    Chạy build_unit_matrices cho mỗi unit, dùng chung cho tất cả 9 sensor.
    Giả định cycle_col ĐÃ được chuẩn hóa từ bên ngoài, khớp domain_range
    của BASIS.

    Trả về dict: unit_id -> {"B","D","U","n","cycle"} hoặc {"error": ...}
    """
    def _one(unit_id, group_df):
        cycle = np.sort(group_df[cycle_col].to_numpy().astype(float))
        n = len(cycle)

        if n < 5 or n < min_points_ratio * basis:
            return unit_id, {"error": f"n={n} không đủ (N_BASIS={basis})"}
        if np.any(np.diff(cycle) == 0):
            return unit_id, {"error": "cycle trùng lặp"}

        try:
            B, BtB, D, U = build_unit_matrices(cycle)
        except np.linalg.LinAlgError as e:
            return unit_id, {"error": f"eigendecomp thất bại: {e}"}

        return unit_id, {"B": B, "D": D, "U": U, "n": n, "cycle": cycle}

    groups = list(df.groupby(unit_col, sort=False))
    outputs = Parallel(n_jobs=n_jobs)(delayed(_one)(uid, g) for uid, g in groups)
    return dict(outputs)


def fleet_gcv_score(log_lam, unit_data_list):
    """
    GCV(lambda) trung bình qua toàn bộ unit — đây là hàm mục tiêu
    thật sự truyền vào optimizer. unit_data_list: list các tuple
    (D_i, g_i, residual_const_i, n_i), đã precompute cho 1 sensor cụ thể.
    """
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
    """
    Với 1 sensor: transform_response qua mọi unit hợp lệ, tìm lambda* fleet.

    Returns
    -------
    lam_star, gcv_star : float
    """
    unit_data_list = []   # (D_i, g_i, residual_const_i, n_i)

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


def smooth_all_sensors_fleet(df, sensor_cols, unit_col="unit_number",
                              cycle_col="cycles", min_points_ratio=2.0,
                              log_lam_bounds=lam_bounds, n_jobs=-1):
    """
    Chạy toàn bộ pipeline tìm lambda: cache 100 unit 1 lần,
    sau đó loop 9 sensor.

    Returns
    -------
    summary_df : pd.DataFrame  (sensor, lambda, gcv)
    unit_cache : dict
    """
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
    return summary_df, unit_cache

import numpy as np
import pandas as pd


def compute_smoothed_train(df, summary_df, unit_cache, sensor_cols,
                            unit_col="unit_number", cycle_col="t_registered"):
    """
    Khối 1: tính giá trị đã mượt hóa cho toàn bộ train, dùng đúng
    lambda tối ưu (theo sensor) đã tìm được ở nhánh A.

    Returns
    -------
    smoothed_df : pd.DataFrame
        Cùng cấu trúc với df (unit_col, cycle_col, sensor_cols),
        giá trị sensor đã được thay bằng giá trị mượt hóa.
        Các (unit, sensor) bị lỗi sẽ giữ nguyên NaN ở cột sensor đó.
    coef_store : dict
        {unit_id: {sensor_name: {"coef": ndarray, "basis": basis_train}}}
        Giữ lại riêng cho bước MFPCA sau này (cần hệ số, không chỉ giá trị).
    error_log : pd.DataFrame
        Cột: unit_id, sensor, error -- để kiểm tra (unit, sensor) nào lỗi.
    """
    lam_lookup = summary_df.set_index("sensor")["lambda"].to_dict()

    missing = [s for s in sensor_cols if s not in lam_lookup
               or lam_lookup[s] is None]
    if missing:
        raise ValueError(f"Các sensor sau chưa có lambda hợp lệ trong "
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
    return smoothed_df, coef_store, error_log

def fit_coefficients(D, U, g, lam):
    """
    c = U @ (g / (1 + lam*D)) -- suy trực tiếp từ nghiệm penalized
    least squares, không cần giải lại hệ phương trình.
    """
    shrink = 1.0 / (1.0 + lam * D)
    return U @ (g * shrink)

def smooth_predict(basis, coef, x_eval):
    """
    Evaluate spline đã fit tại các điểm x_eval bất kỳ (trong domain của basis).

    Parameters
    ----------
    basis : skfda BSplineBasis (BASIS toàn cục cho train, hoặc
            local_basis lưu trong cache cho test)
    coef  : từ fit_coefficients
    x_eval : array_like

    Returns
    -------
    ndarray, shape (len(x_eval),)
    """
    x_eval = np.asarray(x_eval, dtype=float)
    Bx = basis(x_eval)[:, :, 0].T        # shape (len(x_eval), n_basis)
    return Bx @ coef



# ---------- Basis + Omega adaptive cho 1 unit test ----------

def build_unit_matrices_adaptive(cycle, order=order, min_points_ratio=2.0):
    """
    Dựng basis/Omega CỤC BỘ cho 1 unit test, domain = [cycle.min(), cycle.max()]
    (raw cycle, không chuẩn hóa), số knot theo quy tắc bài báo:
    1 knot cho mỗi 4-5 điểm quan sát, tối đa 20 knot.

    Parameters
    ----------
    cycle : array_like, shape (n,)
        Cycle thô, đã sort, không trùng lặp.
    order : int
        Bậc spline (order=4 cho cubic).
    min_points_ratio : float
        Ngưỡng n >= min_points_ratio * n_basis để coi hợp lệ.

    Returns
    -------
    dict với "B", "D", "U", "n_basis" nếu hợp lệ,
    hoặc {"error": ...} nếu không đủ điều kiện.
    """
    n = len(cycle)

    n_knots = min(20, max(4, n // 4))
    n_basis = n_knots + order

    if n < 5 or n < min_points_ratio * n_basis:
        return {"error": f"n={n} không đủ cho n_basis={n_basis} "
                          f"(cần >= {min_points_ratio * n_basis:.0f})"}

    c_min, c_max = cycle[0], cycle[-1]

    local_basis = BSplineBasis(
        domain_range=(c_min, c_max), n_basis=n_basis, order=order
    )
    local_omega = REGULARIZATION.penalty_matrix(local_basis)

    B = local_basis(cycle)[:, :, 0].T       # shape (n, n_basis)
    BtB = B.T @ B

    try:
        D, U = eigh(local_omega, b=BtB)
    except np.linalg.LinAlgError as e:
        return {"error": f"eigendecomposition thất bại: {e}"}

    return {"B": B, "D": D, "U": U, "n": n, "cycle": cycle, "n_basis": n_basis}


# ---------- Tìm lambda riêng cho 1 (unit, sensor) ----------

def find_lambda_single_unit(D, g, residual_const, n,
                             log_lam_bounds=lam_bounds):
    result = minimize_scalar(
        gcv_score, bounds=log_lam_bounds, method="bounded",
        args=(D, g, residual_const, n),
    )
    lam_star = np.exp(result.x)
    return lam_star, result.fun

