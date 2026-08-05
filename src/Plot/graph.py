import matplotlib.pyplot as plt

def plot_rul_comparison(compare_df: pd.DataFrame, pred_col, true_col="RUL_true",
                          title="RUL Prediction", ax=None):
    sorted_df = compare_df.sort_values(true_col).reset_index(drop=True)
    x = range(1, len(sorted_df) + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(x, sorted_df[pred_col], facecolors="none", edgecolors="red",
               label="Predicted RUL", s=25)
    ax.scatter(x, sorted_df[true_col], facecolors="none", edgecolors="black",
               label="True RUL", s=25)

    ax.set_xlabel(f"{title} - {len(sorted_df)} total Test Engines")
    ax.set_ylabel("RUL")
    ax.legend(loc="lower right", frameon=True)
    ax.set_title("")

    return ax
