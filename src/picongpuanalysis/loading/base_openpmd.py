import openpmd_api as opmd
import numpy as np


def load_vector_field_component(path: str, iteration: int, fieldname: str, fieldcomponent: str) -> dict:
    """
    Loads a vector field component from an openPMD file.

    Parameters:
    path (str): The path to the openPMD file.
    iteration (int): The iteration number of the data to load.
    fieldname (str): The name of the field to load.
    fieldcomponent (str): The component of the field to load.

    Returns:
    dict: A dictionary containing the loaded vector field data.
    """
    series = opmd.Series(path, opmd.Access.read_only)
    i = series.iterations[iteration]

    chunkdata = i.meshes[fieldname][fieldcomponent].load_chunk()
    unit = i.meshes[fieldname][fieldcomponent].get_attribute("unitSI")

    ret_dict = {"data": np.swapaxes(chunkdata, 0, 2) * unit, "axis_labels": ["x_position", "y_position", "z_position"]}

    if "unitDimension" in i.meshes[fieldname].attributes:
        unit_dimension = i.meshes[fieldname].get_attribute("unitDimension")
        ret_dict["unit_dimension"] = unit_dimension

    series.flush()
    series.close()

    del i
    del series

    x_space, y_space, z_space = load_vector_grid(path, iteration, fieldname, fieldcomponent)

    ret_dict["x_space"] = x_space
    ret_dict["y_space"] = y_space
    ret_dict["z_space"] = z_space

    return ret_dict


def load_vector_field(path: str, iteration: int, fieldname: str) -> dict:
    components = ["x", "y", "z"]

    ret_dict = {}
    for component in components:
        ret_dict = {**ret_dict, **load_vector_field_component(path, iteration, fieldname, component)}

    return ret_dict


def load_vector_grid(path: str, iteration: int, fieldname: str, fieldcomponent: str) -> tuple:
    series = opmd.Series(path, opmd.Access.read_only)
    i = series.iterations[iteration]

    tmp_field = i.meshes[fieldname][fieldcomponent].load_chunk()
    series.flush()
    shape = tmp_field.shape

    gridSpacing = i.meshes[fieldname].grid_spacing
    gridUnitSI = i.meshes[fieldname].grid_unit_SI
    gridGlobalOffset = i.meshes[fieldname].grid_global_offset

    series.flush()
    series.close()

    del i
    del series

    x_space = ((np.arange(shape[2]) - shape[2] // 2) * gridSpacing[2] + gridGlobalOffset[2]) * gridUnitSI
    y_space = (np.arange(shape[1]) * gridSpacing[1] + gridGlobalOffset[1]) * gridUnitSI
    z_space = ((np.arange(shape[0]) - shape[0] // 2) * gridSpacing[0] + gridGlobalOffset[0]) * gridUnitSI

    return x_space, y_space, z_space


def load_scalar_grid(path: str, iteration: int, fieldname: str) -> tuple:
    series = opmd.Series(path, opmd.Access.read_only)
    i = series.iterations[iteration]

    tmp_field = i.meshes[fieldname][opmd.Mesh_Record_Component.SCALAR].load_chunk()
    series.flush()
    shape = tmp_field.shape

    gridSpacing = i.meshes[fieldname].grid_spacing
    gridUnitSI = i.meshes[fieldname].grid_unit_SI
    gridGlobalOffset = i.meshes[fieldname].grid_global_offset

    series.flush()
    series.close()

    del i
    del series

    x_space = ((np.arange(shape[2]) - shape[2] // 2) * gridSpacing[2] + gridGlobalOffset[2]) * gridUnitSI
    y_space = (np.arange(shape[1]) * gridSpacing[1] + gridGlobalOffset[1]) * gridUnitSI
    z_space = ((np.arange(shape[0]) - shape[0] // 2) * gridSpacing[0] + gridGlobalOffset[0]) * gridUnitSI

    return x_space, y_space, z_space
