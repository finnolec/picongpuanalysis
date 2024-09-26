import itertools
import openpmd_api as opmd
import numpy as np
import typeguard

from picongpuanalysis.utils.units import unit_m, unit_unitless, unit_omega


@typeguard.typechecked
def load_shadowgraphy_fourier(path: str, iteration: int, use_si_units: bool = True) -> dict:
    """
    Loads full shadowgraphy plugin fourier data from an openPMD file.

    Parameters:
        path (str): The path to the shadowgraphy plugin openPMD file.
        iteration (int): The iteration number of simulation.
        use_si_units (bool): Whether to use SI units for the data. Defaults to True.

    Returns:
        dict: A dictionary containing the loaded shadowgraphy fourier data.
    """
    field_components = ["x", "y"]
    field_names = ["E", "B"]
    field_signs = ["positive", "negative"]

    ret_dict = {}

    for field_name, field_component, field_sign in itertools.product(field_names, field_components, field_signs):
        ret_key = f"{field_name}{field_component} - {field_sign}"
        ret_dict = {
            **ret_dict,
            ret_key: load_shadowgraphy_fourier_component(
                path, iteration, field_name, field_component, field_sign, use_si_units=use_si_units
            ),
        }

    return ret_dict


@typeguard.typechecked
def load_shadowgraphy_fourier_component(
    path: str, iteration: int, field_name: str, field_component: str, field_sign: str, use_si_units: bool = True
) -> dict:
    """
    Loads single shadowgraphy plugin fourier data component from an openPMD file.

    Parameters:
        path (str): The path to the shadowgraphy plugin openPMD file.
        iteration (int): The iteration number of simulation.
        field_name (str): The name of the field. Must be "E" or "B".
        field_component (str): The component of the field. Must be "x" or "y".
        field_sign (str): The sign of the field. Must be "positive" or "negative".
        use_si_units (bool): Whether to use SI units for the data. Defaults to True.

    Returns:
        dict: A dictionary containing the loaded shadowgraphy fourier data.
    """
    assert (field_name == "E") or (field_name == "B"), "field_name must be E or B"
    assert (field_component == "x") or (field_component == "y"), "field_component must be x or y"
    assert (field_sign == "positive") or (field_sign == "negative"), "field_sign must be positive or negative"

    opmd_name, opmd_component = _get_openpmd_field_component_name(field_name, field_component, field_sign)

    series = opmd.Series(path, opmd.Access.read_only)
    i = series.iterations[iteration]

    chunkdata = i.meshes[opmd_name][opmd_component].load_chunk()
    unit = i.meshes[opmd_name][opmd_component].get_attribute("unitSI")

    series.flush()

    shape = chunkdata.shape

    # Transpose to move data in x, y, omega order, it is omega, y, x before
    ret_dict = {"data": chunkdata.transpose() * unit}
    del chunkdata

    if "unitDimension" in i.meshes[opmd_name].attributes:
        unit_dimension = i.meshes[opmd_name].get_attribute("unitDimension")
        ret_dict["unit_dimension"] = unit_dimension

    if use_si_units:
        ret_dict["axis_labels"] = ["x_position", "y_position", "omega_frequency"]
        ret_dict["axis_units"] = [unit_m, unit_m, unit_omega]

        x_space = i.meshes["Spatial positions"]["x"].load_chunk()
        y_space = i.meshes["Spatial positions"]["y"].load_chunk()
        omega_space = i.meshes["Fourier Transform Frequencies"]["omegas"].load_chunk()
        series.flush()

        ret_dict["x_space"] = x_space[0, 0, :]
        ret_dict["y_space"] = y_space[0, :, 0]
        if field_sign == "positive":
            ret_dict["omega_space"] = omega_space[:, 0, 0][shape[0] :]
        else:
            ret_dict["omega_space"] = omega_space[:, 0, 0][: shape[0]]
    else:
        ret_dict["axis_labels"] = ["x_position_index", "y_position_index", "omega_frequency_index"]
        ret_dict["axis_units"] = [unit_unitless, unit_unitless, unit_unitless]
        ret_dict["x_space"] = np.arange(shape[2])
        ret_dict["y_space"] = np.arange(shape[1])
        ret_dict["omega_space"] = np.arange(shape[0])

    series.close()

    del i
    del series

    return ret_dict


@typeguard.typechecked
def _get_openpmd_field_component_name(field_name: str, field_component: str, field_sign: str) -> tuple:
    """
    Returns the openPMD field component name based on the provided field name, component, and sign.

    Parameters:
        field_name (str): The name of the field. Must be "E" or "B".
        field_component (str): The component of the field. Must be "x" or "y".
        field_sign (str): The sign of the field. Must be "positive" or "negative".

    Returns:
        tuple: A tuple containing the openPMD field name and the field component name.
    """
    assert (field_name == "E") or (field_name == "B"), "field_name must be E or B"
    assert (field_component == "x") or (field_component == "y"), "field_component must be x or y"
    assert (field_sign == "positive") or (field_sign == "negative"), "field_sign must be positive or negative"

    return f"Fourier Domain Fields - {field_sign}", f"{field_name}{field_component}"
