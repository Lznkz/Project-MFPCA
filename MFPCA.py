import numpy as np
from scipy.interpolate import interp1d
from skfda import FDataGrid


def build_fdata_grid(smoothed_df, sensor_cols, unit_col="unit_id",
                      cycle_col="cycle", n_grid=100):
    """
    Nội suy các đường cong đã mượt (lưới cycle không đều) sang lưới
    đồng nhất dense_grid, rồi đóng gói thành FDataGrid cho MFPCA.

    Parameters
    ----------
    smoothed_df : pd.DataFrame
        Kết quả từ compute_smoothed_train.
    sensor_cols : list[str]
    unit_col, cycle_col : str
    n_grid : int
        Số điểm trên lưới đồng nhất [0, 1].

    Returns
    -------
    fd_multi : skfda.FDataGrid
        shape data_matrix: (n_units, n_grid, n_sensors)
    units : ndarray
        Thứ tự unit_id tương ứng với chiều đầu tiên của fd_multi
        (quan trọng để map ngược kết quả MFPCA sau này).
    """
    dense_grid = np.linspace(0.0, 1.0, n_grid)
    units = smoothed_df[unit_col].unique()

    data_array = np.zeros((len(units), n_grid, len(sensor_cols)))

    for i, uid in enumerate(units):
        sub = smoothed_df[smoothed_df[unit_col] == uid].sort_values(cycle_col)
        t = sub[cycle_col].to_numpy()

        for j, sensor in enumerate(sensor_cols):
            y = sub[sensor].to_numpy()
            interp = interp1d(t, y, kind="linear", bounds_error=False,
                               fill_value="extrapolate")
            data_array[i, :, j] = interp(dense_grid)

    fd_multi = FDataGrid(
        data_matrix=data_array,
        grid_points=dense_grid,
        argument_names=["t_registered"],
        coordinate_names=sensor_cols,
    )

    return fd_multi, units
