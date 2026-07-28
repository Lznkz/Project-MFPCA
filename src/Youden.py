import numpy as np
from Label import find_lifespan


def Youden(labels, data):
    labels = np.array(labels)
    data = np.array(data)

    # Xác định hướng
    mean1 = data[labels == 1].mean()
    mean0 = data[labels == 0].mean()
    reverse = mean1 < mean0   # True nếu label=1 có data thấp hơn

    thresholds = np.unique(data)
    P = labels.sum()
    N = len(labels) - P

    best_J, best_thresh = -np.inf, None
    best_sens, best_spec = 0, 0

    for t in thresholds:
        if reverse:
            pred = (data <= t).astype(int)
        else:
            pred = (data >= t).astype(int)

        TP = ((pred == 1) & (labels == 1)).sum()
        TN = ((pred == 0) & (labels == 0)).sum()
        sens = TP / P
        spec = TN / N
        J = sens + spec - 1

        if J > best_J:
            best_J, best_thresh = J, t
            best_sens, best_spec = sens, spec

    result= {
        "cutoff": best_thresh,
        "youden_J": best_J,
        "sensitivity": best_sens,
        "specificity": best_spec,
        "direction": "<=" if reverse else ">="
    }
    return result
def Youden_diagnostic(labels, data):
    """Trả về TOÀN BỘ threshold đạt J cực đại (không chỉ 1), để kiểm tra tie."""
    labels = np.array(labels)
    data = np.array(data)

    mean1 = data[labels == 1].mean()
    mean0 = data[labels == 0].mean()
    reverse = mean1 < mean0

    thresholds = np.unique(data)
    P = labels.sum()
    N = len(labels) - P

    J_values = []
    for t in thresholds:
        pred = (data <= t).astype(int) if reverse else (data >= t).astype(int)
        TP = ((pred == 1) & (labels == 1)).sum()
        TN = ((pred == 0) & (labels == 0)).sum()
        sens = TP / P
        spec = TN / N
        J_values.append(sens + spec - 1)

    J_values = np.array(J_values)
    best_J = J_values.max()
    tie_mask = np.isclose(J_values, best_J)
    tied_thresholds = thresholds[tie_mask]

    return {
        "best_J": best_J,
        "n_ties": tie_mask.sum(),
        "tied_thresholds": tied_thresholds,
        "reverse": reverse,
    }
def Youden_near_ties(labels, data, tolerance=0.02):
    """In ra TOP 5 threshold có J gần với J tối đa nhất, không chỉ tie tuyệt đối."""
    labels = np.array(labels)
    data = np.array(data)
    mean1 = data[labels == 1].mean()
    mean0 = data[labels == 0].mean()
    reverse = mean1 < mean0
    thresholds = np.unique(data)
    P = labels.sum()
    N = len(labels) - P

    J_values = []
    for t in thresholds:
        pred = (data <= t).astype(int) if reverse else (data >= t).astype(int)
        TP = ((pred == 1) & (labels == 1)).sum()
        TN = ((pred == 0) & (labels == 0)).sum()
        J_values.append(TP / P + TN / N - 1)

    J_values = np.array(J_values)
    order = np.argsort(J_values)[::-1][:5]   # top 5 gần nhất
    for idx in order:
        print(f"threshold={thresholds[idx]:.4f}  J={J_values[idx]:.4f}")

def Youden_with_tiebreak(labels, data, tie_pick="first"):
    """
    tie_pick: "first" (nhỏ nhất, = hành vi hiện tại), "last" (lớn nhất),
              "middle" (trung vị các tie)
    """
    diag = Youden_diagnostic(labels, data)
    tied = diag["tied_thresholds"]

    if tie_pick == "first":
        cutoff = tied[0]
    elif tie_pick == "last":
        cutoff = tied[-1]
    elif tie_pick == "middle":
        cutoff = tied[len(tied) // 2]
    else:
        raise ValueError("tie_pick phải là 'first'/'last'/'middle'")

    data = np.array(data)
    labels = np.array(labels)
    reverse = diag["reverse"]
    P = labels.sum()
    N = len(labels) - P
    pred = (data <= cutoff).astype(int) if reverse else (data >= cutoff).astype(int)
    TP = ((pred == 1) & (labels == 1)).sum()
    TN = ((pred == 0) & (labels == 0)).sum()

    return {
        "cutoff": cutoff,
        "youden_J": diag["best_J"],
        "sensitivity": TP / P,
        "specificity": TN / N,
        "direction": "<=" if reverse else ">=",
    }

def take_iv(df, sensor_cols, unit_col="unit_number"):
    iv = df.groupby(unit_col)[sensor_cols].apply(lambda x: x.head(5).mean())
    iv = iv.reset_index()   # đưa unit_number từ index thành cột
    return iv

def Youden_merged(label_train, iv_train_df):
    labels = label_train['label']
    result = {}
    for col in iv_train_df:
        value = Youden(labels, iv_train_df[col])
        result[col] = value
    return result

def registration(df):
    df["t_registered"] = df.groupby("unit_number")["cycles"].transform(
        lambda t: (t - 1) / (t.max() - 1)
    )
    return df

# def normalize(candidates, train_smoothed, sensor_cols):
#     X_bar_j = train_smoothed[sensor_cols].mean().to_numpy()

#     results = {}
#     for i in candidates.keys():
#         test_eval = candidates[i]['test_eval']
#         candidate_evals = candidates[i]['candidate_evals']

#         u_test = test_eval / X_bar_j
#         u_candidates = {
#             uid: arr / X_bar_j for uid, arr in candidate_evals.items()
#         }

#         results[i] = {
#             "u_test": u_test,
#             "u_candidates": u_candidates,
#         }

#     return results


def normalize(candidates):
    results = {}

    for i in candidates:
        test_eval = candidates[i]['test_eval']
        candidate_evals = candidates[i]['candidate_evals']

        if len(candidate_evals) == 0:
            results[i] = {"error": "không có candidate để tính X_bar_j(t)"}
            continue

        all_candidate_arrays = np.stack(list(candidate_evals.values()))
        X_bar_j_t = all_candidate_arrays.mean(axis=0)

        u_test = test_eval / X_bar_j_t
        u_candidates = {
            uid: arr / X_bar_j_t for uid, arr in candidate_evals.items()
        }

        results[i] = {
            "u_test": u_test,
            "u_candidates": u_candidates,
        }

    return results

# def distance_cal(normalized):
#     d_xy_t = {}
#     D_total = {}
#     for i in normalized:
#         if "error" in normalized[i]:
#             continue
#         u_test = normalized[i]['u_test']
#         u_candidates = normalized[i]['u_candidates']

#         d_xy_t[i] = {}
#         D_total[i] = {}
#         for j, u_cand in u_candidates.items():
#             diff_sq = (u_cand - u_test)**2
#             sum_sqrt_t = diff_sq.sum(axis=1)

#             d_xy_t[i][j] = np.sqrt(sum_sqrt_t)
#             D_total[i][j] = float(np.sqrt(sum_sqrt_t.sum()))

#     return D_total
def distance_cal(normalized):
    D_total = {}

    for i in normalized:
        if "error" in normalized[i]:
            continue

        u_test = normalized[i]['u_test']         # (20, 9)
        u_candidates = normalized[i]['u_candidates']

        D_total[i] = {}
        for j, u_cand in u_candidates.items():
            diff_sq = (u_cand - u_test) ** 2       # (20, 9)
            per_sensor_dist = np.sqrt(diff_sq.sum(axis=0))   # (9,) -- L2 theo thời gian, từng sensor
            D_total[i][j] = float(per_sensor_dist.sum())      # cộng dồn 9 sensor

    return D_total

def validate(D_total, df, candidates, k=8):
    lifespan_df = find_lifespan(df)
    results = {}
    for i in D_total:
        _, T_test = candidates[i]

        topk_uids = sorted(D_total[i], key=D_total[i].get)[:k]
        topk_lifespan = lifespan_df[
            lifespan_df['unit_number'].isin(topk_uids)
        ].copy()

        topk_lifespan['RUL'] = topk_lifespan['cycles'] - T_test

        RUL = {
            "mean": topk_lifespan['RUL'].mean(),
            "median": topk_lifespan['RUL'].median(),
        }

        results[i] = (topk_lifespan, RUL)

    return results
