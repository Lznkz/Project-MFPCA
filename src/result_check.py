import pandas as pd
import numpy as np
from src.k_nn import validate, normalize
from skfda.representation.basis import FDataBasis

def check_rmse(results, rul_true_df, unit_col="unit_number", true_col="RUL"):
    rows = []
    for i, (topk_lifespan, RUL) in results.items():
        rows.append({
            unit_col: i,
            "RUL_pred_mean": RUL["mean"],
            "RUL_pred_median": RUL["median"],
        })

    pred_df = pd.DataFrame(rows)

    compare_df = pred_df.merge(
        rul_true_df[[unit_col, true_col]], on=unit_col, how="inner"
    )

    missing = set(pred_df[unit_col]) - set(compare_df[unit_col])
    if missing:
        print(f"Warning: {len(missing)} unit didn't have RUL info, skipped: "
              f"{sorted(missing)}")

    rmse_mean = np.sqrt(
        ((compare_df["RUL_pred_mean"] - compare_df[true_col]) ** 2).mean()
    )
    rmse_median = np.sqrt(
        ((compare_df["RUL_pred_median"] - compare_df[true_col]) ** 2).mean()
    )

    rmse = {"mean": rmse_mean, "median": rmse_median}

    return compare_df, rmse

eval_grid = np.linspace(0, 1, 100)

def compute_second_derivative(basis, coef, eval_grid):
    fd = FDataBasis(basis, coef.reshape(1, -1))
    fd_deriv2 = fd.derivative(order=2)
    values = fd_deriv2(eval_grid)
    return values[0, :, 0]

def compute_all_second_derivatives(coef_store, sensor_cols, eval_grid=eval_grid):
    results = {}
    for unit_id, sensor_dict in coef_store.items():
        results[unit_id] = {}
        for sensor_name in sensor_cols:
            if sensor_name not in sensor_dict:
                continue
            coef = sensor_dict[sensor_name]["coef"]
            basis = sensor_dict[sensor_name]["basis"]
            results[unit_id][sensor_name] = compute_second_derivative(
                basis, coef, eval_grid
            )
    return results

def average_second_derivative_by_group(d2_results, label_train_df, sensor_cols,
                                         unit_col="unit_number"):
    label_map = dict(zip(label_train_df[unit_col], label_train_df["label"]))

    grouped = {0: {s: [] for s in sensor_cols}, 1: {s: [] for s in sensor_cols}}

    for unit_id, sensor_d2 in d2_results.items():
        label = label_map.get(unit_id)
        if label is None:
            continue
        for sensor_name, d2 in sensor_d2.items():
            grouped[label][sensor_name].append(d2)

    avg_result = {0: {}, 1: {}}
    for label in [0, 1]:
        for sensor_name in sensor_cols:
            arrs = grouped[label][sensor_name]
            if len(arrs) == 0:
                continue
            avg_result[label][sensor_name] = np.mean(np.stack(arrs), axis=0)

    return avg_result

def compute_alarm_points(rul_predictions, candidates, threshold=0.8):
    rows = []
    for uid, rul_pred in rul_predictions.items():
        _, T_test = candidates[uid]
        alarm_cycle = threshold * (T_test + rul_pred)
        rows.append({
            "unit_number": uid, "T_test": T_test,
            "RUL_pred": rul_pred, "alarm_cycle": alarm_cycle,
        })
    return pd.DataFrame(rows)

def evaluate_alarm_performance(alarm_df, check_df, unit_col="unit_number"):
    merged = alarm_df.merge(check_df, on=unit_col, how="inner")

    merged["true_failure_cycle"] = merged["T_test"] + merged["RUL"]
    merged["alarm_before_failure"] = merged["alarm_cycle"] < merged["true_failure_cycle"]
    merged["pct_life_at_alarm"] = merged["alarm_cycle"] / merged["true_failure_cycle"]

    pct_safe = merged["alarm_before_failure"].mean() * 100
    pct_last_20 = (merged["pct_life_at_alarm"] >= 0.8).mean() * 100
    pct_last_40 = (merged["pct_life_at_alarm"] >= 0.6).mean() * 100

    print(f"Alarms triggered before actual failure: {pct_safe:.1f}%")
    print(f"Alarms in last 20% of life: {pct_last_20:.1f}%")
    print(f"Alarms in last 40% of life: {pct_last_40:.1f}%")

    return merged

def alarm_performance_table(merged):

    n_total = len(merged)
    n_later = (~merged["alarm_before_failure"]).sum()
    n_earlier = merged["alarm_before_failure"].sum()

    n_last_40 = (merged["pct_life_at_alarm"] >= 0.60).sum()
    n_last_30 = (merged["pct_life_at_alarm"] >= 0.70).sum()
    n_last_20 = (merged["pct_life_at_alarm"] >= 0.80).sum()
    n_last_10 = (merged["pct_life_at_alarm"] >= 0.90).sum()
    n_last_5  = (merged["pct_life_at_alarm"] >= 0.95).sum()

    table = pd.DataFrame({
        "Metric": [
            "No of Total Test Engines",
            '"Alarm Point" is later than "True Failure Point"',
            '"Alarm Point" is earlier than "True Failure Point"',
            '"Alarm Point" is in last 40% of Total Life',
            '"Alarm Point" is in last 30% of Total Life',
            '"Alarm Point" is in last 20% of Total Life',
            '"Alarm Point" is in last 10% of Total Life',
            '"Alarm Point" is in last 5% of Total Life',
        ],
        "Unit": [
            n_total, n_later, n_earlier,
            n_last_40, n_last_30, n_last_20, n_last_10, n_last_5,
        ],
    })
    print(table)
    return table

def alarm_performance(coef_store, candidates, sensor_col, label_train, result, check_df):
    '''
    coef_store is from smoothing spline
    result is from knn_apply()
    check_df is contain RUL
    '''
    d2_results = compute_all_second_derivatives(coef_store, sensor_col, eval_grid)
    avg_d2_by_group = average_second_derivative_by_group(d2_results, label_train, sensor_col)
    rul_predictions = {i: r[1]["mean"] for i, r in result.items()}
    alarm_df = compute_alarm_points(rul_predictions, candidates, threshold=0.8)
    merged = evaluate_alarm_performance(alarm_df, check_df)
    table = alarm_performance_table(merged)

def round_predictions(df):
    df['RUL_pred_mean'] = df['RUL_pred_mean'].round().astype(int)
    df['RUL_pred_median'] = df['RUL_pred_median'].round().astype(int)

    df['matched (mean)'] = (df['RUL_pred_mean'] == df['RUL']).astype(int)
    df['matched (median)'] = (df['RUL_pred_median'] == df['RUL']).astype(int)

    print(df['matched (mean)'].value_counts(),
          df['matched (median)'].value_counts()
          )
    return df

def check_sensor_variance(smoothed_test, sensor_cols):
    normalized = normalize(smoothed_test)
    valid_u_test = [
        normalized[i]['u_test']
        for i in normalized
        if 'error' not in normalized[i]
    ]

    if len(valid_u_test) == 0:
        raise ValueError("Invalid unit in 'normalized' ")

    all_u_test = np.stack(valid_u_test)   # (n_test_units, 20, 9)
    variance_per_sensor = all_u_test.var(axis=(0, 1))

    result = pd.DataFrame({
        "sensor": sensor_cols,
        "variance": variance_per_sensor,
    }).sort_values("variance", ascending=False).reset_index(drop=True)
    print(result)
    return result
