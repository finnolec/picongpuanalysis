import itertools
import numpy as np
import typeguard

from picongpuanalysis.utils.units import unit_k, unit_m, unit_omega, unit_t


def compute_shadowgram(fields: dict) -> dict:
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
    data = np.real(np.sum(poynting_vectors, axis=2)) * delta_t

    ret_dict = {}
    ret_dict["data"] = data
    ret_dict["delta_t"] = delta_t
    ret_dict["axis_labels"] = ["x_position", "y_position", "t_time"]
    ret_dict["axis_units"] = [unit_m, unit_m, unit_t]
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
    field_names = list(fields.keys())

    ret_dict = {}

    for field_name in field_names:
        assert fields[field_name]["axis_units"] == [
            unit_k,
            unit_k,
            unit_omega,
        ], "Field units must be [unit_k, unit_k, unit_omega]"

        data_xyt = np.fft.ifftn(np.fft.ifftshift(fields[field_name]["data"], axes=(0, 1, 2)), axes=(0, 1, 2))
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
