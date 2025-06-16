import copy
import itertools
import numpy as np
import os
import pickle
import scipy.constants as const
import typeguard

from picongpuanalysis.utils.units import unit_k, unit_m, unit_omega, unit_t


@typeguard.typechecked
def apply_band_pass_filter(
    fields: dict, lower_cutoff: float, upper_cutoff: float, overwrite_fields: bool = True
) -> dict:
    """
    Applies a band-pass filter to the fields in the k-omega or xy-omega space.

    Parameters:
    fields (dict): A dictionary with the fields as keys and a dictionary containing the
        data and omega_space of the field as values.
    lower_cutoff (float): The lower angular frequency cutoff of the band-pass filter in SI units.
    upper_cutoff (float ): The upper angular frequency cutoff of the band-pass filter in SI units.
    overwrite_fields (bool, optional): If True, the fields are overwritten. If False, the fields are copied
        before applying the band-pass filter. Default is True.

    Returns:
        dict: A dictionary with the filtered fields.
    """
    assert lower_cutoff < upper_cutoff, "lower_cutoff must be smaller than upper_cutoff"
    assert lower_cutoff > 0, "lower_cutoff must be positive"

    if not overwrite_fields:
        fields = copy.deepcopy(fields)

    for field_name in fields.keys():
        assert fields[field_name]["axis_units"][2] == unit_omega, "Field units must be [arb., arb., unit_omega]"

        omega_space = np.abs(fields[field_name]["omega_space"])
        # Round cutoffs to nearest omega_space value
        lower_cutoff = float(omega_space[_find_closest_idx(omega_space, lower_cutoff)])
        upper_cutoff = float(omega_space[_find_closest_idx(omega_space, upper_cutoff)])

        # Set band-pass filter
        mask_upper = np.where(omega_space > upper_cutoff, 0, 1)
        mask_lower = np.where(omega_space < lower_cutoff, 0, 1)
        mask = mask_upper * mask_lower

        masked_fields = fields[field_name]["data"] * mask

        # Truncate arrays that are previously truncated
        if " - " in field_name:
            if "positive" in field_name:
                min_idx = _find_closest_idx(fields[field_name]["omega_space"], lower_cutoff)
                max_idx = _find_closest_idx(fields[field_name]["omega_space"], upper_cutoff)
            elif "negative" in field_name:
                min_idx = _find_closest_idx(fields[field_name]["omega_space"], -upper_cutoff) + 1
                max_idx = _find_closest_idx(fields[field_name]["omega_space"], -lower_cutoff) + 1
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
def apply_numerical_aperture(
    fields: dict,
    numerical_aperture: float,
    overwrite_fields: bool = True,
    window_function=None,
) -> dict:
    """
    Applies a numerical aperture to the fields in the k-omega space, with optional window function.

    Parameters:
        fields (dict): A dictionary with the fields as keys and a dictionary containing the
            data and omega_space of the field as values.
        numerical_aperture (float): The numerical aperture to apply.
        overwrite_fields (bool, optional): If True, the original fields will be overwritten.
            If False, a copy of the fields will be made and the numerical aperture will be applied on the copy.
        window_function (callable, optional): A window function from scipy.signal.windows. If provided,
            it will be applied radially in k_perp, centered at k_perp=0, with the window's support
            extending to k_perp = NA * omega / c. The window function should accept an integer (number of points)
            and return a 1D array.

    Returns:
        dict: The fields with the numerical aperture applied.
    """
    assert numerical_aperture > 0, "numerical_aperture must be positive"

    if not overwrite_fields:
        fields = copy.deepcopy(fields)

    for field_name in fields.keys():
        assert fields[field_name]["axis_units"] == [
            unit_k,
            unit_k,
            unit_omega,
        ], "Field units must be [unit_k, unit_k, unit_omega]"

        kx = fields[field_name]["kx_space"]
        ky = fields[field_name]["ky_space"]
        omega = fields[field_name]["omega_space"]

        kxm, kym, omegam = np.meshgrid(kx, ky, omega, indexing="ij")
        k_perp = np.sqrt(kxm**2 + kym**2)
        k_aperture = np.abs(numerical_aperture * omegam / const.c)

        if window_function is not None:
            # For each omega slice, apply the window function radially in k_perp
            mask = np.zeros_like(k_perp)
            for idx in range(omegam.shape[2]):
                k_ap = k_aperture[:, :, idx][0, 0]  # scalar for this omega
                if k_ap == 0:
                    continue
                # Compute normalized k_perp for this omega slice
                k_perp_slice = k_perp[:, :, idx]
                # Only apply window inside aperture
                inside = k_perp_slice <= k_ap
                # Number of points for window: use the max k_perp index inside aperture
                n_points = np.count_nonzero(inside)
                if n_points == 0:
                    continue
                # Sort k_perp values inside aperture for window mapping
                k_perp_flat = k_perp_slice[inside]
                # Map k_perp from 0 to k_ap to window indices
                window_vals = window_function(n_points)
                # Assign window values to mask
                # Sort k_perp_flat and assign window values in order of increasing k_perp
                sort_idx = np.argsort(k_perp_flat)
                mask_slice = np.zeros_like(k_perp_slice)
                mask_indices = np.argwhere(inside)
                for i, idx_pair in enumerate(mask_indices[sort_idx]):
                    mask_slice[tuple(idx_pair)] = window_vals[i]
                mask[:, :, idx] = mask_slice
            # Hard cutoff outside aperture
            mask[k_perp > k_aperture] = 0
        else:
            # Hard mask
            mask = np.where(k_perp > k_aperture, 0, 1)

        fields[field_name]["data"] *= mask
        fields[field_name]["numerical_aperture"] = numerical_aperture
        fields[field_name]["numerical_aperture_mask"] = mask
        if window_function is not None:
            fields[field_name]["numerical_aperture_window"] = window_function.__name__
        else:
            fields[field_name]["numerical_aperture_window"] = None

    return fields


@typeguard.typechecked
def apply_custom_mask(fields: dict, mask: np.ndarray, overwrite_fields: bool = True) -> dict:
    if not overwrite_fields:
        fields = copy.deepcopy(fields)

    for field_name in fields.keys():
        assert fields[field_name]["axis_units"] == [
            unit_k,
            unit_k,
            unit_omega,
        ], "Field units must be [unit_k, unit_k, unit_omega]"

        if field_name.endswith("positive"):
            fields[field_name]["data"] *= mask[:, :, : mask.shape[2] // 2]
        elif field_name.endswith("negative"):
            fields[field_name]["data"] *= mask[:, :, mask.shape[2] // 2 :]
        # else:
        #    fields[field_name]["data"] *= mask
        # fields[field_name]["data"] *= mask

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

    if "numerical_aperture" in fields["Ex"].keys():
        ret_dict["numerical_aperture"] = fields["Ex"]["numerical_aperture"]
    else:
        ret_dict["numerical_aperture"] = None

    if "upper_cutoff" in fields["Ex"].keys():
        ret_dict["upper_cutoff"] = fields["Ex"]["upper_cutoff"]
        ret_dict["lower_cutoff"] = fields["Ex"]["lower_cutoff"]
    else:
        ret_dict["upper_cutoff"] = None
        ret_dict["lower_cutoff"] = None

    if "propagation_method" in fields["Ex"].keys():
        ret_dict["delta_z"] = fields["Ex"]["delta_z"]
        ret_dict["propagation_method"] = fields["Ex"]["propagation_method"]
    else:
        ret_dict["delta_z"] = None
        ret_dict["propagation_method"] = None

    return ret_dict


@typeguard.typechecked
def fft_xyo_to_kko(fields: dict) -> dict:
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
                np.abs(fields[field_name]["x_space"][1] - fields[field_name]["x_space"][0]) / (2 * np.pi),
            )
        )
        ret_dict[field_name]["ky_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["y_space"].shape[0],
                np.abs(fields[field_name]["y_space"][1] - fields[field_name]["y_space"][0]) / (2 * np.pi),
            )
        )

        ret_dict[field_name]["omega_space"] = fields[field_name]["omega_space"]

    return ret_dict


@typeguard.typechecked
def fft_xyt_to_xyo(fields: dict) -> dict:
    """
    Fourier transform fields in x-time space to fields in x-omega space.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values.

    Returns:
        dict: A dictionary with the same keys as the input, but with the field data and axis units
            transformed to x-omega space.
    """
    field_names = list(fields.keys())

    ret_dict = {}

    for field_name in field_names:
        assert fields[field_name]["axis_units"] == [
            unit_m,
            unit_m,
            unit_t,
        ], "Field units must be [unit_m, unit_m, unit_t]"

        data_xyo = np.fft.fft(fields[field_name]["data"], axis=2, norm="backward")

        ret_dict.setdefault(field_name, {"data": data_xyo})

        ret_dict[field_name]["axis_labels"] = ["x_position", "y_position", "omega_frequency"]
        ret_dict[field_name]["axis_units"] = [unit_m, unit_m, unit_omega]

        ret_dict[field_name]["x_space"] = fields[field_name]["x_space"]
        ret_dict[field_name]["y_space"] = fields[field_name]["y_space"]

        ret_dict[field_name]["omega_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["t_space"].shape[0],
                np.abs(fields[field_name]["t_space"][1] - fields[field_name]["t_space"][0]) / (2 * np.pi),
            )
        )

    return ret_dict


@typeguard.typechecked
def ifft_kko_to_xyt(fields: dict) -> dict:
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
                np.abs(fields[field_name]["kx_space"][1] - fields[field_name]["kx_space"][0]) / (2 * np.pi),
            )
        )
        ret_dict[field_name]["y_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["ky_space"].shape[0],
                np.abs(fields[field_name]["ky_space"][1] - fields[field_name]["ky_space"][0]) / (2 * np.pi),
            )
        )
        # TODO figure out start time of plugin and use it here
        ret_dict[field_name]["t_space"] = np.fft.fftshift(
            np.fft.fftfreq(
                fields[field_name]["omega_space"].shape[0],
                np.abs(fields[field_name]["omega_space"][1] - fields[field_name]["omega_space"][0]) / (2 * np.pi),
            )
        )

        if "numerical_aperture" in fields[field_name].keys():
            ret_dict[field_name]["numerical_aperture"] = fields[field_name]["numerical_aperture"]

        if "upper_cutoff" in fields[field_name].keys():
            ret_dict[field_name]["upper_cutoff"] = fields[field_name]["upper_cutoff"]
            ret_dict[field_name]["lower_cutoff"] = fields[field_name]["lower_cutoff"]

        if "propagation_method" in fields[field_name].keys():
            ret_dict[field_name]["propagation_method"] = fields[field_name]["propagation_method"]
            ret_dict[field_name]["delta_z"] = fields[field_name]["delta_z"]

    return ret_dict


@typeguard.typechecked
def propagate_fields(
    fields: dict, delta_z: float, propagation_method: str = "angular_spectrum", overwrite_fields: bool = True
) -> dict:
    """
    Propagates fields in k-omega space along the z-axis by a distance of delta_z.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values.
        delta_z (float): The distance to propagate the fields along the z-axis. Units are meters.
        propagation_method (str): The method to use for propagation. Can be either "angular_spectrum" or "fresnel".
            Defaults to "angular_spectrum".
        overwrite_fields (bool): If True, the input dictionary will be modified. If False, a deep copy of the dictionary
            will be made before modification. Defaults to True.

    Returns:
        dict: The input dictionary with the fields propagated along the z-axis.
    """
    field_names = list(fields.keys())

    if not overwrite_fields:
        fields = copy.deepcopy(fields)

    for field_name in field_names:
        assert fields[field_name]["axis_units"] == [
            unit_k,
            unit_k,
            unit_omega,
        ], "Field units must be [unit_k, unit_k, unit_omega]"

        kx = fields[field_name]["kx_space"]
        ky = fields[field_name]["ky_space"]
        omega = fields[field_name]["omega_space"] / const.c

        kxm, kym, km = np.meshgrid(kx, ky, omega, indexing="ij")

        if propagation_method == "angular_spectrum":
            sqrt_content = 1 - (kxm / km) ** 2 - (kym / km) ** 2
            # Clipping to avoid negative square roots
            sqrt_content = np.clip(sqrt_content, 0, None)
            # Masking to remove evanescent fields
            mask = np.where(sqrt_content > 0, 1, 0)
            # Angular spectrum waves
            phase = np.where(km == 0, 0, delta_z * km * np.sqrt(sqrt_content))
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
def restore_fields_kko(fields: dict, delta_t: float, field_components=["x", "y"], field_names=["E", "B"]) -> dict:
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
    ret_dict = {}

    for field_name, field_component in itertools.product(field_names, field_components):
        # Load positive field
        write_name = f"{field_name}{field_component}"
        read_name_pos = f"{field_name}{field_component} - positive"
        assert fields[read_name_pos]["axis_units"] == [
            unit_k,
            unit_k,
            unit_omega,
        ], "Field units must be [unit_k, unit_k, unit_omega]"

        # Load truncated omega space
        truncated_omega_space_pos = fields[read_name_pos]["omega_space"]
        delta_omega = np.abs(truncated_omega_space_pos[1] - truncated_omega_space_pos[0])

        # Calculate final size of array
        n_t = int(round(2 * np.pi / (delta_t * delta_omega)))

        padded_omega_space = 2 * np.pi * (np.arange(n_t) - n_t / 2) / n_t / delta_t

        padded_array = np.zeros(fields[read_name_pos]["data"].shape[:-1] + (n_t,), dtype=np.complex128)

        # Insert truncated data into padded array
        start_idx = _find_closest_idx(padded_omega_space, truncated_omega_space_pos[0])
        end_idx = _find_closest_idx(padded_omega_space, truncated_omega_space_pos[-1]) + 1

        padded_array[:, :, start_idx:end_idx] = fields[read_name_pos]["data"]

        # Load negative field
        read_name_neg = f"{field_name}{field_component} - negative"
        truncated_omega_space_neg = fields[read_name_neg]["omega_space"]

        # Insert truncated data into padded array
        start_idx = _find_closest_idx(padded_omega_space, truncated_omega_space_neg[0])
        end_idx = _find_closest_idx(padded_omega_space, truncated_omega_space_neg[-1]) + 1

        padded_array[:, :, start_idx:end_idx] = fields[read_name_neg]["data"]

        ret_dict.setdefault(write_name, {"data": padded_array})

        ret_dict[write_name]["axis_labels"] = ["kx_wavevector", "ky_wavevector", "omega_frequency"]
        ret_dict[write_name]["axis_units"] = [unit_k, unit_k, unit_omega]
        ret_dict[write_name]["kx_space"] = fields[read_name_pos]["kx_space"]
        ret_dict[write_name]["ky_space"] = fields[read_name_pos]["ky_space"]
        ret_dict[write_name]["omega_space"] = padded_omega_space

        if "propagation_method" in fields[read_name_pos].keys():
            ret_dict[write_name]["propagation_method"] = fields[read_name_pos]["propagation_method"]
            ret_dict[write_name]["delta_z"] = fields[read_name_pos]["delta_z"]

        if "numerical_aperture" in fields[read_name_pos].keys():
            ret_dict[write_name]["numerical_aperture"] = fields[read_name_pos]["numerical_aperture"]

        if "upper_cutoff" in fields[read_name_pos].keys():
            ret_dict[write_name]["lower_cutoff"] = fields[read_name_pos]["lower_cutoff"]
            ret_dict[write_name]["upper_cutoff"] = fields[read_name_pos]["upper_cutoff"]

    return ret_dict


@typeguard.typechecked
def split_fields_kko(fields: dict) -> dict:
    """
    Split the fields into positive and negative omega components from a full k-omega space.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values.

    Returns:
        dict: A dictionary with separate positive and negative components for each field in the input.
    """
    field_names = ["E", "B"]
    field_components = ["x", "y"]

    ret_dict = {}

    for field_name, field_component in itertools.product(field_names, field_components):
        read_name = f"{field_name}{field_component}"
        assert fields[read_name]["axis_units"] == [
            unit_k,
            unit_k,
            unit_omega,
        ], "Field units must be [unit_m, unit_m, unit_omega]"

        full_omega_space = fields[read_name]["omega_space"]
        data = fields[read_name]["data"]

        # Find the zero index in the omega space
        zero_idx = _find_closest_idx(full_omega_space, 0.0)

        # Split the omega space and data into positive and negative parts
        positive_omega_space = full_omega_space[zero_idx:]
        negative_omega_space = full_omega_space[:zero_idx]

        positive_data = data[:, :, zero_idx:]
        negative_data = data[:, :, :zero_idx]

        # Create entries for positive and negative components
        write_name_pos = f"{read_name} - positive"
        write_name_neg = f"{read_name} - negative"

        ret_dict[write_name_pos] = {
            "data": positive_data,
            "axis_labels": fields[read_name]["axis_labels"],
            "axis_units": fields[read_name]["axis_units"],
            "kx_space": fields[read_name]["kx_space"],
            "ky_space": fields[read_name]["ky_space"],
            "omega_space": positive_omega_space,
        }

        ret_dict[write_name_neg] = {
            "data": negative_data,
            "axis_labels": fields[read_name]["axis_labels"],
            "axis_units": fields[read_name]["axis_units"],
            "kx_space": fields[read_name]["kx_space"],
            "ky_space": fields[read_name]["ky_space"],
            "omega_space": negative_omega_space,
        }

    return ret_dict


@typeguard.typechecked
def split_fields_xyo(fields: dict) -> dict:
    """
    Split the fields into positive and negative omega components from a full position-omega space.

    Parameters:
        fields (dict): A dictionary with field names as keys and dictionaries containing the field data,
            axis labels, and axis units as values.

    Returns:
        dict: A dictionary with separate positive and negative components for each field in the input.
    """
    field_names = ["E", "B"]
    field_components = ["x", "y"]

    ret_dict = {}

    for field_name, field_component in itertools.product(field_names, field_components):
        read_name = f"{field_name}{field_component}"
        assert fields[read_name]["axis_units"] == [
            unit_m,
            unit_m,
            unit_omega,
        ], "Field units must be [unit_m, unit_m, unit_omega]"

        full_omega_space = fields[read_name]["omega_space"]
        data = fields[read_name]["data"]

        # Find the zero index in the omega space
        zero_idx = _find_closest_idx(full_omega_space, 0.0)

        # Split the omega space and data into positive and negative parts
        positive_omega_space = full_omega_space[zero_idx:]
        negative_omega_space = full_omega_space[:zero_idx]

        positive_data = data[:, :, zero_idx:]
        negative_data = data[:, :, :zero_idx]

        # Create entries for positive and negative components
        write_name_pos = f"{read_name} - positive"
        write_name_neg = f"{read_name} - negative"

        ret_dict[write_name_pos] = {
            "data": positive_data,
            "axis_labels": fields[read_name]["axis_labels"],
            "axis_units": fields[read_name]["axis_units"],
            "x_space": fields[read_name]["x_space"],
            "y_space": fields[read_name]["y_space"],
            "omega_space": positive_omega_space,
        }

        ret_dict[write_name_neg] = {
            "data": negative_data,
            "axis_labels": fields[read_name]["axis_labels"],
            "axis_units": fields[read_name]["axis_units"],
            "x_space": fields[read_name]["x_space"],
            "y_space": fields[read_name]["y_space"],
            "omega_space": negative_omega_space,
        }

    return ret_dict


def save_fields_xyt(fields: dict, filename: str, overwrite: bool = False) -> None:
    """
    Saves the fields dictionary to a file in binary format.

    Parameters:
        fields (dict): A dictionary containing the field data to be saved.
        filename (str): The name of the file to save the fields to. The file must not already exist.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.


    Raises:
        AssertionError: If the file already exists.

    Notes:
        The fields are saved using the pickle module.
    """
    if not overwrite:
        assert not os.path.exists(filename), f"File {filename} already exists."
    else:
        if os.path.exists(filename):
            print(f"Overwriting {filename}.")

    with open(filename, "wb") as f:
        pickle.dump(fields, f)

    print(f"Saved fields to {filename}.")


def load_fields_xyt(filename: str) -> dict:
    """
    Loads field data from a file in binary format.

    Parameters:
        filename (str): The name of the file to load the fields from.

    Returns:
        dict: A dictionary containing the loaded field data.

    Raises:
        AssertionError: If the file does not exist.

    Notes:
        The fields are loaded using the pickle module.
    """
    assert os.path.exists(filename), f"File {filename} does not exist."

    with open(filename, "rb") as f:
        fields = pickle.load(f)

    return fields


def save_fields_kko(fields: dict, filename: str, overwrite: bool = False) -> None:
    """
    Saves the k-omega space fields dictionary to a file in binary format.

    Parameters:
        fields (dict): A dictionary containing the field data to be saved.
        filename (str): The name of the file to save the fields to. The file must not already exist.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.

    Raises:
        AssertionError: If the file already exists.

    Notes:
        The fields are saved using the pickle module.
    """
    if not overwrite:
        assert not os.path.exists(filename), f"File {filename} already exists."
    else:
        if os.path.exists(filename):
            print(f"Overwriting {filename}.")

    with open(filename, "wb") as f:
        pickle.dump(fields, f)

    print(f"Saved fields to {filename}.")


def load_fields_kko(filename: str) -> dict:
    """
    Loads field data from a file in binary format.

    Parameters:
        filename (str): The name of the file to load the fields from.

    Returns:
        dict: A dictionary containing the loaded field data.

    Raises:
        AssertionError: If the file does not exist.

    Notes:
        The fields are loaded using the pickle module.
    """
    assert os.path.exists(filename), f"File {filename} does not exist."

    with open(filename, "rb") as f:
        fields = pickle.load(f)

    return fields


def save_shadowgram(shadowgram: dict, filename: str, overwrite: bool = False) -> None:
    """
    Saves a shadowgram dictionary to a file.

    Parameters:
        shadowgram (dict): Shadowgram dictionary containing the data, axis labels, and axis units.
        filename (str): Filename to save the shadowgram to.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.

    Notes:
        The file is saved in binary format using pickle.
    """
    if not overwrite:
        assert not os.path.exists(filename), f"File {filename} already exists."
    else:
        if os.path.exists(filename):
            print(f"Overwriting {filename}.")

    with open(filename, "wb") as f:
        pickle.dump(shadowgram, f)

    print(f"Saved shadowgram to {filename}.")


def load_shadowgram(filename: str) -> dict:
    """
    Loads a shadowgram from a file.

    Parameters:
        filename (str): Filename to load the shadowgram from.

    Returns:
        dict: Shadowgram dictionary containing the data, axis labels, and axis units.

    Notes:
        The file is loaded in binary format using pickle.
    """
    assert os.path.exists(filename), f"File {filename} does not exist."

    with open(filename, "rb") as f:
        shadowgram = pickle.load(f)

    return shadowgram


def _find_closest_idx(arr, target_value):
    """
    Find the index of the value in arr that is closest to target_value.

    Parameters:
        arr (numpy.ndarray): The array to search.
        target_value (float): The value to search for.

    Returns:
        int: The index of the closest value.
    """
    return np.argmin(np.abs(arr - target_value))
