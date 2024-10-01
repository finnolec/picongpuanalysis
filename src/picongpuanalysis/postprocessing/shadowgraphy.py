import copy
import itertools
import numpy as np
import scipy.constants as const
import typeguard

from picongpuanalysis.utils.units import unit_k, unit_m, unit_omega, unit_t


@typeguard.typechecked
def apply_band_pass_filter(
    fields: dict, lower_cutoff: float, upper_cutoff: float, override_fields: bool = True
) -> dict:
    """
    Applies a band-pass filter to the fields in the k-omega or xy-omega space.

    Parameters:
    fields (dict): A dictionary with the fields as keys and a dictionary containing the
        data and omega_space of the field as values.
    lower_cutoff (float): The lower angular frequency cutoff of the band-pass filter in SI units.
    upper_cutoff (float ): The upper angular frequency cutoff of the band-pass filter in SI units.
    override_fields (bool, optional): If True, the fields are overwritten. If False, the fields are copied
        before applying the band-pass filter. Default is True.

    Returns:
        dict: A dictionary with the filtered fields.
    """
    assert lower_cutoff < upper_cutoff, "lower_cutoff must be smaller than upper_cutoff"
    assert lower_cutoff > 0, "lower_cutoff must be positive"
    assert "omega_space" in fields[list(fields.keys())[0]].keys(), "omega_space not found in fields"

    # Check if band-pass filter is equal or smaller than truncated k-omega space arrays
    if "Ex - positive" in fields.keys():
        assert lower_cutoff >= np.min(np.abs(fields["Ex - positive"]["omega_space"])), "lower_cutoff too small"
        assert upper_cutoff <= np.max(np.abs(fields["Ex - positive"]["omega_space"])), "upper_cutoff too large"
    else:
        assert upper_cutoff <= np.max(np.abs(fields[list(fields.keys())[0]]["omega_space"])), "upper_cutoff too large"

    if not override_fields:
        fields = copy.deepcopy(fields)

    for field_name in fields.keys():
        # Set band-pass filter
        omega_space = np.abs(fields[field_name]["omega_space"])
        mask_upper = np.where(omega_space > upper_cutoff, 0, 1)
        mask_lower = np.where(omega_space < lower_cutoff, 0, 1)
        mask = mask_upper * mask_lower

        masked_fields = fields[field_name]["data"] * mask

        # Truncate arrays that are previously truncated
        if " - " in field_name:
            if "positive" in field_name:
                min_idx = np.searchsorted(
                    fields[field_name]["omega_space"],
                    lower_cutoff,
                    sorter=np.argsort(fields[field_name]["omega_space"]),
                )
                max_idx = np.searchsorted(
                    fields[field_name]["omega_space"],
                    upper_cutoff,
                    sorter=np.argsort(fields[field_name]["omega_space"]),
                )
            elif "negative" in field_name:
                min_idx = np.searchsorted(
                    fields[field_name]["omega_space"],
                    -upper_cutoff,
                    sorter=np.argsort(fields[field_name]["omega_space"]),
                )
                max_idx = np.searchsorted(
                    fields[field_name]["omega_space"],
                    -lower_cutoff,
                    sorter=np.argsort(fields[field_name]["omega_space"]),
                )
            else:
                raise ValueError("field_name must be positive or negative")

            fields[field_name]["data"] = masked_fields[:, :, min_idx:max_idx]
            fields[field_name]["omega_space"] = fields[field_name]["omega_space"][min_idx:max_idx]
        else:
            fields[field_name]["data"] = masked_fields

        fields[field_name]["band-pass_mask"] = mask
        fields[field_name]["upper_cutoff"] = upper_cutoff
        fields[field_name]["lower_cutoff"] = lower_cutoff

        del masked_fields

    return fields


@typeguard.typechecked
def apply_numerical_aperture(fields: dict, numerical_aperture: float, override_fields: bool = True) -> dict:
    """
    Applies a nuemrical aperture to the fields in the k-omega space.

    Parameters:
    fields (dict): A dictionary with the fields as keys and a dictionary containing the
        data and omega_space of the field as values.
    numerical_aperture (float):
        The numerical aperture to apply
    override_fields (bool, optional):
        If True, the original fields will be overwritten. If False, a copy of the fields will be made and the numerical aperture will be applied on the copy.

    Returns:
        dict: The fields with the numerical aperture applied
    """
    assert numerical_aperture > 0, "numerical_aperture must be positive"
    assert "omega_space" in fields[list(fields.keys())[0]].keys(), "omega_space not found in fields"
    assert "kx_space" in fields[list(fields.keys())[0]].keys(), "kx_space not found in fields"
    assert "ky_space" in fields[list(fields.keys())[0]].keys(), "ky_space not found in fields"

    if not override_fields:
        fields = copy.deepcopy(fields)

    for field_name in fields.keys():
        kx = fields[field_name]["kx_space"]
        ky = fields[field_name]["ky_space"]
        omega = fields[field_name]["omega_space"]

        kxm, kym, omegam = np.meshgrid(kx, ky, omega, indexing="ij")

        mask = np.where(kxm**2 + kym**2 > (numerical_aperture * omegam / const.c) ** 2, 0, 1)

        fields[field_name]["data"] *= mask
        fields[field_name]["numerical_aperture"] = numerical_aperture
        fields[field_name]["numerical_aperture_mask"] = mask

    return fields


@typeguard.typechecked
def compute_shadowgram(fields: dict) -> dict:
    """
    Compute a shadowgram in z direction from the given electric and magnetic fields.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values. The fields must be in (x, y, t) space.

    Returns:
        dict: A dictionary of the shadowgram data, axis labels, and axis units.
    """
    assert (
        "Ex" in fields.keys() and "Ey" in fields.keys() and "Bx" in fields.keys() and "By" in fields.keys()
    ), "Fields must contain Ex, Ey, Bx, and By"
    assert (
        fields["Ex"]["axis_units"]
        == fields["Ey"]["axis_units"]
        == fields["Bx"]["axis_units"]
        == fields["By"]["axis_units"]
        == [unit_m, unit_m, unit_t]
    ), "Field units must be [unit_m, unit_m, unit_t]"

    delta_t = fields["Ex"]["t_space"][1] - fields["Ex"]["t_space"][0]

    poynting_vectors = fields["Ex"]["data"] * fields["By"]["data"] - fields["Ey"]["data"] * fields["Bx"]["data"]
    data = np.sum(np.real(poynting_vectors), axis=2) * delta_t / const.mu_0

    ret_dict = {}
    ret_dict["data"] = data
    ret_dict["delta_t"] = delta_t
    ret_dict["axis_labels"] = ["x_position", "y_position"]
    ret_dict["axis_units"] = [unit_m, unit_m]
    ret_dict["x_space"] = fields["Ex"]["x_space"]
    ret_dict["y_space"] = fields["Ex"]["y_space"]

    return ret_dict


@typeguard.typechecked
def fft_to_kko(fields: dict) -> dict:
    """
    Fourier transform fields in k-position space to fields in k-omega space.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values.

    Returns:
        dict: A dictionary with the same keys as the input, but with the field data and axis units
            transformed to k-omega space.
    """
    field_names = list(fields.keys())

    ret_dict = {}

    for field_name in field_names:
        assert fields[field_name]["axis_units"] == [
            unit_m,
            unit_m,
            unit_omega,
        ], "Field units must be [unit_m, unit_m, unit_omega]"

        data_kko = np.fft.fftshift(np.fft.fft2(fields[field_name]["data"], axes=(0, 1)), axes=(0, 1))
        ret_dict.setdefault(field_name, {"data": data_kko})

        ret_dict[field_name]["axis_labels"] = ["kx_wavevector", "ky_wavevector", "omega_frequency"]
        ret_dict[field_name]["axis_units"] = [unit_k, unit_k, unit_omega]

        ret_dict[field_name]["kx_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["x_space"].shape[0],
                np.abs(fields[field_name]["x_space"][1] - fields[field_name]["x_space"][0]),
            )
        )
        ret_dict[field_name]["ky_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["y_space"].shape[0],
                np.abs(fields[field_name]["y_space"][1] - fields[field_name]["y_space"][0]),
            )
        )

        ret_dict[field_name]["omega_space"] = fields[field_name]["omega_space"]

    return ret_dict


@typeguard.typechecked
def ifft_to_xyt(fields: dict) -> dict:
    """
    Transforms fields from k-omega space to x-y-t space.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values.

    Returns:
        dict: A dictionary with the same keys as the input, but with the field data and axis units
            transformed to x-y-t space.
    """
    field_names = list(fields.keys())

    ret_dict = {}

    for field_name in field_names:
        assert fields[field_name]["axis_units"] == [
            unit_k,
            unit_k,
            unit_omega,
        ], "Field units must be [unit_k, unit_k, unit_omega]"

        data_xyt = np.fft.ifftn(np.fft.ifftshift(fields[field_name]["data"], axes=(0, 1)), axes=(0, 1))
        # TODO check if the following still works with propagators
        # Otherwise np.fft.ifft(data_xyt, axis=2, norm="backward") might be correct.
        # It is weird that there is no fftshift anymore
        data_xyt = np.fft.fft(data_xyt, axis=2, norm="forward")
        ret_dict.setdefault(field_name, {"data": data_xyt})

        ret_dict[field_name]["axis_labels"] = ["x_position", "y_position", "t_time"]
        ret_dict[field_name]["axis_units"] = [unit_m, unit_m, unit_t]

        ret_dict[field_name]["x_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["kx_space"].shape[0],
                np.abs(fields[field_name]["kx_space"][1] - fields[field_name]["kx_space"][0]),
            )
        )
        ret_dict[field_name]["y_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["ky_space"].shape[0],
                np.abs(fields[field_name]["ky_space"][1] - fields[field_name]["ky_space"][0]),
            )
        )
        # TODO figure out start time of plugin and use it here
        ret_dict[field_name]["t_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["omega_space"].shape[0],
                np.abs(fields[field_name]["omega_space"][1] - fields[field_name]["omega_space"][0]) / 2 / np.pi,
            )
        )

    return ret_dict


@typeguard.typechecked
def propagate_fields(
    fields: dict, delta_z: float, propagation_method: str = "angular_spectrum", override_fields: bool = True
) -> dict:
    """
    Propagates fields in k-omega space along the z-axis by a distance of delta_z.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values.
        delta_z (float): The distance to propagate the fields along the z-axis. Units are meters.
        propagation_method (str): The method to use for propagation. Can be either "angular_spectrum" or "fresnel".
            Defaults to "angular_spectrum".
        override_fields (bool): If True, the input dictionary will be modified. If False, a deep copy of the dictionary
            will be made before modification. Defaults to True.

    Returns:
        dict: The input dictionary with the fields propagated along the z-axis.
    """
    field_names = list(fields.keys())

    if not override_fields:
        fields = copy.deepcopy(fields)

    for field_name in field_names:
        kx = fields[field_name]["kx_space"]
        ky = fields[field_name]["ky_space"]
        omega = fields[field_name]["omega_space"] / const.c
        kxm, kym, km = np.meshgrid(kx, ky, omega, indexing="ij")

        return kxm, kym, km

        if propagation_method == "angular_spectrum":
            sqrt_content = 1 - (kxm / km) ** 2 - (kym / km) ** 2
            mask = np.where(sqrt_content > 0, 1, 0)
            phase = np.where(km == 0, 0, delta_z * km * np.emath.sqrt(sqrt_content))
            propagator = mask * np.exp(1j * phase)
        elif propagation_method == "fresnel":
            # TODO check if correct
            phase = np.exp(-1j * 2 * np.pi * km * delta_z)
        else:
            raise ValueError("Unknown propagation method")

        fields[field_name]["data"] = fields[field_name]["data"] * propagator
        fields[field_name]["delta_z"] = delta_z
        fields[field_name]["propagation_method"] = propagation_method

    return fields


@typeguard.typechecked
def restore_fields_kko(fields: dict, delta_t: float) -> dict:
    """
    Pad the truncated k-omega space fields to the original size for 3D FFTs.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values.
        delta_t (float): The time step to use for padding.

    Returns:
        dict: A dictionary with the same keys as the input, but with the field data and axis units
            transformed to k-omega space and padded to original size.
    """
    field_names = list(fields.keys())

    ret_dict = {}

    field_components = ["x", "y"]
    field_names = ["E", "B"]

    for field_name, field_component in itertools.product(field_names, field_components):
        # Load positive field
        write_name = f"{field_name}{field_component}"
        read_name_pos = f"{field_name}{field_component} - positive"
        assert fields[read_name_pos]["axis_units"] == [
            unit_k,
            unit_k,
            unit_omega,
        ], "Field units must be [unit_m, unit_m, unit_omega]"

        # Load truncated omega space
        truncated_omega_space_pos = fields[read_name_pos]["omega_space"]
        delta_omega = np.abs(truncated_omega_space_pos[1] - truncated_omega_space_pos[0])

        # Calculate final size of array
        n_t = int(round(2 * np.pi / (delta_t * delta_omega)))
        padded_omega_space = 2 * np.pi * (np.arange(n_t) - n_t / 2) / n_t / delta_t

        padded_array = np.zeros(fields[read_name_pos]["data"].shape[:-1] + (n_t,), dtype=np.complex128)

        # Insert truncated data into padded array
        start_idx = np.searchsorted(padded_omega_space, truncated_omega_space_pos[0])
        end_idx = np.searchsorted(padded_omega_space, truncated_omega_space_pos[-1])
        padded_array[:, :, start_idx:end_idx] = fields[read_name_pos]["data"]

        # Load negative field
        read_name_neg = f"{field_name}{field_component} - negative"
        truncated_omega_space_neg = fields[read_name_neg]["omega_space"]

        # Insert truncated data into padded array
        start_idx = np.searchsorted(padded_omega_space, truncated_omega_space_neg[0])
        end_idx = np.searchsorted(padded_omega_space, truncated_omega_space_neg[-1])
        padded_array[:, :, start_idx:end_idx] = fields[read_name_neg]["data"]

        ret_dict.setdefault(write_name, {"data": padded_array})

        ret_dict[write_name]["axis_labels"] = ["kx_wavevector", "ky_wavevector", "omega_frequency"]
        ret_dict[write_name]["axis_units"] = [unit_k, unit_k, unit_omega]
        ret_dict[write_name]["kx_space"] = fields[read_name_pos]["kx_space"]
        ret_dict[write_name]["ky_space"] = fields[read_name_pos]["ky_space"]
        ret_dict[write_name]["omega_space"] = padded_omega_space

    return ret_dict
