import numpy as np

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


def find_lifespan(df):
    result = df.groupby('unit_number')['cycles'].max().reset_index()
    return result

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


def ff(smoothed_test, df, candidates, k=8):
    normalized = normalize(smoothed_test)
    D_total = distance_cal(normalized)
    validate_info = validate(D_total, df, candidates, k=k)
    return validate_info
