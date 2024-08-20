import numpy as np


def field_mean(field: dict, axis: int) -> dict:
    """
    This function calculates the mean of a given field along a specified axis.

    Parameters:
        field (dict): A dictionary containing the field data and its axis labels.
        axis (int): The axis along which to calculate the mean.

    Returns:
        dict: A dictionary containing the mean field data and its updated axis labels.
    """
    field["data"] = np.mean(field["data"], axis=axis)
    field["axis_labels"] = np.delete(field["axis_labels"], axis)

    if axis == 0:
        field["x_position"] = np.mean(field["x_space"])
        del field["x_space"]
    elif axis == 1:
        field["y_position"] = np.mean(field["y_space"])
        del field["y_space"]
    elif axis == 2:
        field["z_position"] = np.mean(field["z_space"])
        del field["z_space"]

    return field


def field_transpose(field: dict) -> dict:
    """
    This function transposes the data and axis labels of a given field.

    Parameters:
        field (dict): A dictionary containing the field data and its axis labels.

    Returns:
        dict: The input field with its data and axis labels transposed.
    """
    assert len(field["data"].shape) == 2, "data must be a 2D array"

    field["data"] = np.transpose(field["data"])
    field["axis_labels"] = np.flip(field["axis_labels"])

    return field
