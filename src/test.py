import numpy as np
from skfda.representation.basis import BSplineBasis


def build_basis(T_test, order=4, min_points_ratio=2.0, n_eval=20):
    cycle = np.arange(1, T_test + 1, dtype=float)
    n = len(cycle)

    n_knots = min(20, max(4, n // 4))
    n_basis = n_knots + order

    if n < 5 or n < min_points_ratio * n_basis:
        return {"error": f"T_test={T_test} không đủ cho n_basis={n_basis} "
                          f"(cần >= {min_points_ratio * n_basis:.0f})"}

    basis = BSplineBasis(domain_range=(cycle[0], cycle[-1]),
                          n_basis=n_basis, order=order)

    B = basis(cycle)[:, :, 0].T            # (n, n_basis)
    BtB = B.T @ B

    try:
        L = np.linalg.cholesky(BtB)
    except np.linalg.LinAlgError as e:
        return {"error": f"Cholesky thất bại: {e}"}

    eval_grid = np.linspace(cycle[0], cycle[-1], n_eval)
    B_eval = basis(eval_grid)[:, :, 0].T   # (n_eval, n_basis)

    return {
        "B": B,
        "L": L,
        "B_eval": B_eval,
        "eval_grid": eval_grid,
        "basis": basis,
        "n_basis": n_basis,
        "cycle": cycle,
    }

def regression_fit(B, L, y):
    Bty = B.T @ y
    z = np.linalg.solve(L, Bty)
    c = np.linalg.solve(L.T, z)
    return c

def truncate_func(df, candidates, cycle_cutoff):
    truncated_df = df[(df['unit_number'].isin(candidates)) &
                (df['cycles'] <= cycle_cutoff)].copy()
    return truncated_df

def auto(candidates, train_df, test_df, sensor_cols):
    results = {}   # lưu lại kết quả mỗi vòng, không để mất

    for i in candidates.keys():          # không hard-code range(1,101)
        candidate, cycle_validate = candidates[i]

        truncated_train_df = truncate_func(train_df, candidate['unit_number'], cycle_validate)
        test_sub = test_df[test_df['unit_number'] == i]

        basis = build_basis(cycle_validate)

        if "error" in basis:
            results[i] = {"error": basis["error"]}
            continue
        B, L, B_eval = basis["B"], basis["L"], basis["B_eval"]

        test_sorted = test_sub.sort_values('cycles')
        test_cycle = test_sorted['cycles'].to_numpy()

        if not np.allclose(test_cycle, cycle_validate):
            results[i] = {"error": "cycle test không khớp basis"}
            continue

        test_eval = np.zeros((B_eval.shape[0], len(sensor_cols)))
        for j, sensor in enumerate(sensor_cols):
            y = test_sorted[sensor].to_numpy()
            coef = regression_fit(B, L, y)
            test_eval[:, j] = B_eval @ coef

        # ---- evaluate từng candidate ----
        candidate_evals = {}
        for cand_uid in candidate['unit_number']:
            cand_sorted = truncated_train_df[
                truncated_train_df['unit_number'] == cand_uid
            ].sort_values('cycles')
            cand_cycle = cand_sorted['cycles'].to_numpy()

            if not np.allclose(cand_cycle, cycle_validate):
                continue  # bỏ qua candidate lỗi cycle, không làm crash cả vòng

            cand_eval = np.zeros((B_eval.shape[0], len(sensor_cols)))
            for j, sensor in enumerate(sensor_cols):
                y = cand_sorted[sensor].to_numpy()
                coef = regression_fit(B, L, y)
                cand_eval[:, j] = B_eval @ coef

            candidate_evals[cand_uid] = cand_eval

        results[i] = {
            "test_eval": test_eval,
            "candidate_evals": candidate_evals,
        }

    return results
