import openpmd_api as opmd
import numpy as np

from picongpuanalysis.utils.units import unit_m


def load_vector_field_component(path: str, iteration: int, field_name: str, field_component: str) -> dict:
    """
    Loads a vector field component from an openPMD file.

    Parameters:
    path (str): The path to the openPMD file.
    iteration (int): The iteration number of the data to load.
    field_name (str): The name of the field to load.
    field_component (str): The component of the field to load.

    Returns:
    dict: A dictionary containing the loaded vector field data.
    """
    series = opmd.Series(path, opmd.Access.read_only)
    i = series.iterations[iteration]

    chunkdata = i.meshes[field_name][field_component].load_chunk()
    unit = i.meshes[field_name][field_component].get_attribute("unitSI")

    ret_dict = {"data": np.swapaxes(chunkdata, 0, 2) * unit}

    ret_dict["axis_labels"] = ["x_position", "y_position", "z_position"]
    ret_dict["axis_units"] = [unit_m, unit_m, unit_m]

    if "unitDimension" in i.meshes[field_name].attributes:
        unit_dimension = i.meshes[field_name].get_attribute("unitDimension")
        ret_dict["unit_dimension"] = unit_dimension

    series.flush()
    series.close()

    del i
    del series

    x_space, y_space, z_space = load_vector_grid(path, iteration, field_name, field_component)

    ret_dict["x_space"] = x_space
    ret_dict["y_space"] = y_space
    ret_dict["z_space"] = z_space

    return ret_dict


def load_vector_field(path: str, iteration: int, field_name: str) -> dict:
    components = ["x", "y", "z"]

    ret_dict = {}
    for component in components:
        ret_dict = {**ret_dict, **load_vector_field_component(path, iteration, field_name, component)}

    return ret_dict


def load_vector_grid(path: str, iteration: int, field_name: str, field_component: str) -> tuple:
    series = opmd.Series(path, opmd.Access.read_only)
    i = series.iterations[iteration]

    tmp_field = i.meshes[field_name][field_component].load_chunk()
    series.flush()
    shape = tmp_field.shape

    grid_spacing = i.meshes[field_name].grid_spacing
    grid_unit_si = i.meshes[field_name].grid_unit_SI
    grid_global_offset = i.meshes[field_name].grid_global_offset

    series.flush()
    series.close()

    del i
    del series

    x_space = ((np.arange(shape[2]) - shape[2] // 2) * grid_spacing[2] + grid_global_offset[2]) * grid_unit_si
    y_space = (np.arange(shape[1]) * grid_spacing[1] + grid_global_offset[1]) * grid_unit_si
    z_space = ((np.arange(shape[0]) - shape[0] // 2) * grid_spacing[0] + grid_global_offset[0]) * grid_unit_si

    return x_space, y_space, z_space


def load_scalar_grid(path: str, iteration: int, field_name: str) -> tuple:
    series = opmd.Series(path, opmd.Access.read_only)
    i = series.iterations[iteration]

    tmp_field = i.meshes[field_name][opmd.Mesh_Record_Component.SCALAR].load_chunk()
    series.flush()
    shape = tmp_field.shape

    grid_spacing = i.meshes[field_name].grid_spacing
    grid_unit_si = i.meshes[field_name].grid_unit_SI
    grid_global_offset = i.meshes[field_name].grid_global_offset

    series.flush()
    series.close()

    del i
    del series

    x_space = ((np.arange(shape[2]) - shape[2] // 2) * grid_spacing[2] + grid_global_offset[2]) * grid_unit_si
    y_space = (np.arange(shape[1]) * grid_spacing[1] + grid_global_offset[1]) * grid_unit_si
    z_space = ((np.arange(shape[0]) - shape[0] // 2) * grid_spacing[0] + grid_global_offset[0]) * grid_unit_si

    return x_space, y_space, z_space
