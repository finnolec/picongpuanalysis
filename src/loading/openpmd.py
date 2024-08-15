import openpmd_api as opmd


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
    series.flush()
    series.close()

    del i
    del series

    ret_dict = {}
    ret_dict[fieldname] = {
        fieldcomponent: {
            "data": chunkdata * unit,
        }
    }

    return ret_dict
