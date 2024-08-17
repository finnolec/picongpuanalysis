import itertools
import openpmd_api as opmd


def load_shadowgraphy_fourier(path: str, iteration: int, use_si_units: bool = True) -> dict:
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


def load_shadowgraphy_fourier_component(
    path: str, iteration: int, field_name: str, field_component: str, field_sign: str, use_si_units: bool = True
) -> dict:
    assert (field_name == "E") or (field_name == "B"), "field_name must be E or B"
    assert (field_component == "x") or (field_component == "y"), "field_component must be x or y"
    assert (field_sign == "positive") or (field_sign == "negative"), "field_sign must be positive or negative"

    opmd_name, opmd_component = _get_openpmd_field_component_name(field_name, field_component, field_sign)

    series = opmd.Series(path, opmd.Access.read_only)
    i = series.iterations[iteration]

    chunkdata = i.meshes[opmd_name][opmd_component].load_chunk()
    unit = i.meshes[opmd_name][opmd_component].get_attribute("unitSI")

    ret_dict = {"data": chunkdata * unit}
    del chunkdata

    if use_si_units:
        ret_dict["axis_labels"] = ["x_position", "y_position", "omega_frequency"]

    return ret_dict


def _get_openpmd_field_component_name(field_name: str, field_component: str, field_sign: str) -> tuple:
    assert (field_name == "E") or (field_name == "B"), "field_name must be E or B"
    assert (field_component == "x") or (field_component == "y"), "field_component must be x or y"
    assert (field_sign == "positive") or (field_sign == "negative"), "field_sign must be positive or negative"

    return f"Fourier Domain Fields - {field_sign}", f"{field_name}{field_component}"
