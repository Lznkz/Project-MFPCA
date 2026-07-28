import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq

def fit_gmm(df, rho_scores, random_state=0):
    units = df['unit_number'].unique()
    rho_scores = rho_scores[:, 0]
    gmm = GaussianMixture(n_components=2, random_state=random_state)
    gmm.fit(rho_scores.reshape(-1, 1))

    # True boundary từ posterior
    means = gmm.means_.flatten()
    boundary = brentq(
        lambda x: gmm.predict_proba([[x]])[0][np.argmax(means)] -
                  gmm.predict_proba([[x]])[0][np.argmin(means)],
        means.min(), means.max()
    )

    labels = (rho_scores >= boundary).astype(int)
    result = pd.DataFrame({
        "unit_number": units,
        "label": labels,
    })
    return result


# def take_iv(df, unit_col="unit_number", sensor_col="T24"):
#     iv = df.groupby(unit_col)[sensor_col].first()
#     iv_numpy = iv.to_numpy()
#     return iv_numpy
# def take_iv(df, sensor_cols, unit_col="unit_number"):
#     iv = df.groupby(unit_col)[sensor_cols].apply(lambda x: x.head(5).mean())
#     units = iv.index.to_numpy()
#     iv_numpy = iv.to_numpy()
#     return iv_numpy, units





def label(df, iv_test, youden_info):
    units = df['unit_number'].unique()
    cutoff = youden_info['cutoff']
    direction = youden_info['direction']

    reverse = (direction == '<=')

    if reverse:
        label  = (iv_test <= cutoff).astype(int)
    else:
        label = (iv_test >= cutoff).astype(int)
    result = pd.DataFrame({
        "unit_number": units,
        "label": label
    })
    return result

def label_merged(df, iv_test_df, youden_info, sensor_cols,
                  unit_col="unit_number", low_threshold=5):
    per_sensor_labels = {}

    for sensor in sensor_cols:
        info = youden_info[sensor]
        cutoff = info['cutoff']
        direction = info['direction']
        reverse = (direction == '<=')

        merged = df[[unit_col]].drop_duplicates().merge(
            iv_test_df[[unit_col, sensor]], on=unit_col, how="left"
        )

        if reverse:
            lbl = (merged[sensor] <= cutoff).astype(int)
        else:
            lbl = (merged[sensor] >= cutoff).astype(int)

        per_sensor_labels[sensor] = pd.Series(
            lbl.to_numpy(), index=merged[unit_col].to_numpy()
        )

    label_matrix = pd.DataFrame(per_sensor_labels)   # index=unit_number, cột=sensor

    low_count = (label_matrix == 0).sum(axis=1)
    final_label = np.where(low_count >= low_threshold, 0, 1)

    result = pd.DataFrame({
        unit_col: label_matrix.index,
        "label": final_label,
        "low_count": low_count.to_numpy(),
    })

    return result

def find_lifespan(df):
    """
    So we'll find the maximum cycle in df
    """
    result = df.groupby('unit_number')['cycles'].max().reset_index()
    return result

def filter_candidates(train_df, test_df, label_train_df, label_test_df,
                       k=8, cycle_col="cycles"):
    lifespan_train = find_lifespan(train_df).merge(
        label_train_df[["unit_number", "label"]], on="unit_number", how="left"
    )
    lifespan_test = find_lifespan(test_df).merge(
        label_test_df[["unit_number", "label"]], on="unit_number", how="left"
    )

    results = {}
    fallback_log = []

    for _, row in lifespan_test.iterrows():
        uid, cycle_validate, group_validate = (
            row["unit_number"], row[cycle_col], row["label"]
        )

        candidates = lifespan_train[
            (lifespan_train["label"] == group_validate) &
            (lifespan_train[cycle_col] >= cycle_validate)
        ]

        if len(candidates) < k:
            opposite_group = 1 - group_validate   # giả định label nhị phân 0/1
            candidates = lifespan_train[
                (lifespan_train["label"] == opposite_group) &
                (lifespan_train[cycle_col] >= cycle_validate)
            ]
            fallback_log.append({
                "unit_number": uid, "own_group": group_validate,
                "n_own": (lifespan_train["label"] == group_validate).sum(),
                "n_opposite": len(candidates),
            })

        results[uid] = (candidates, cycle_validate)

    return results, fallback_log
