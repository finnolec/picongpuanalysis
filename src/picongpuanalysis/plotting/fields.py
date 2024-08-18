import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np


def plot_electric_field_mesh(
    ax: plt.Axes, field: dict, cmap: str = "bwr", title: str = None, aspect: str = "equal"
) -> None:
    """
    Plot the electric field mesh on the given Axes object.

    Parameters:
        ax (plt.Axes): The Axes object to plot the electric field mesh on.
        field (dict): A dictionary containing the electric field data.
            - "data" (ndarray): The 2D array representing the electric field.
            - "name" (str): The name of the electric field.
            - "iteration" (int): The timestep of the electric field.
        cmap (str, optional): The colormap for the mesh. Defaults to "bwr".
        title (str, optional): The title of the plot. If None, the title will be
            "{field['name']} - timestep {field['iteration']:06d}". Defaults to None.
        aspect (str, optional): The aspect ratio of the plot. Must be "equal" or "auto".
            Defaults to "equal".

    Returns:
        None
    """
    assert aspect == "equal" or aspect == "auto", "aspect must be 'equal' or 'auto'"

    data = field["data"]
    assert len(data.shape) == 2, "data must be a 2D array"

    xm, ym = _get_field_coodinate_mesh(field)
    # TODO add units to label, must be together with rescaling of meshes
    xlabel, ylabel = _get_field_coordinate_names(field)

    if title is None:
        title = f"{field['name']} - timestep {field['iteration']:06d}"

    # Ensure symmetric colorbar
    norm = col.Normalize(-np.max(np.abs(data)), np.max(np.abs(data)))

    pcm = ax.pcolormesh(xm, ym, data, cmap=cmap, norm=norm)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect(aspect)

    # TODO units
    ax.get_figure().colorbar(pcm)


def _get_field_coodinate_mesh(field: dict) -> tuple:
    """
    This function generates a coordinate mesh for a given field.

    Parameters:
        field (dict): A dictionary containing the field data and its axis labels.

    Returns:
        tuple: A tuple containing the x and y coordinates of the mesh.

    Raises:
        ValueError: If the axis labels are invalid.
    """
    # The indices x/y look weird here since pcolormesh has transposed axes
    if field["axis_labels"][1] == "x_position":
        x = field["x_space"]
    elif field["axis_labels"][1] == "y_position":
        x = field["y_space"]
    elif field["axis_labels"][1] == "z_position":
        x = field["z_space"]
    else:
        raise ValueError("Invalid axis_labels")

    if field["axis_labels"][0] == "x_position":
        y = field["x_space"]
    elif field["axis_labels"][0] == "y_position":
        y = field["y_space"]
    elif field["axis_labels"][0] == "z_position":
        y = field["z_space"]
    else:
        raise ValueError("Invalid axis_labels")

    return x, y


def _get_field_coordinate_names(field: dict) -> tuple:
    """
    This function generates the coordinate names for a given field.

    Parameters:
        field (dict): A dictionary containing the field data and its axis labels.

    Returns:
        tuple: A tuple containing the x and y coordinate names.

    Raises:
        ValueError: If the axis labels are invalid.
    """

    # The indices x/y look weird here since pcolormesh has transposed axes
    if field["axis_labels"][1] == "x_position":
        x = "x"
    elif field["axis_labels"][1] == "y_position":
        x = "y"
    elif field["axis_labels"][1] == "z_position":
        x = "z"
    else:
        raise ValueError("Invalid axis_labels")

    if field["axis_labels"][0] == "x_position":
        y = "x"
    elif field["axis_labels"][0] == "y_position":
        y = "y"
    elif field["axis_labels"][0] == "z_position":
        y = "z"
    else:
        raise ValueError("Invalid axis_labels")

    return x, y
