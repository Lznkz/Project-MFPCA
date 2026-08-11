import pandas as pd
import numpy as np
from src.Youden import validate

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
        print(f"Cảnh báo: {len(missing)} unit không có RUL_true, đã bỏ qua: "
              f"{sorted(missing)}")

    rmse_mean = np.sqrt(
        ((compare_df["RUL_pred_mean"] - compare_df[true_col]) ** 2).mean()
    )
    rmse_median = np.sqrt(
        ((compare_df["RUL_pred_median"] - compare_df[true_col]) ** 2).mean()
    )

    rmse = {"mean": rmse_mean, "median": rmse_median}

    return compare_df, rmse




def sweep_k(result, rul_true_df, D_total, train_df, candidates, k_values=range(3, 31)):
    rows = []
    for k in k_values:
        results = validate(D_total, train_df, candidates, k=k)
        _, rmse = check_rmse(results, rul_true_df)
        rows.append({"k": k, "rmse_mean": rmse["mean"], "rmse_median": rmse["median"]})

    return pd.DataFrame(rows)
