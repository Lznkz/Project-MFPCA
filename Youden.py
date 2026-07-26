import numpy as np
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

    return {
        "cutoff": best_thresh,
        "youden_J   ": best_J,
        "sensitivity": best_sens,
        "specificity": best_spec,
        "direction": "<=" if reverse else ">="
    }