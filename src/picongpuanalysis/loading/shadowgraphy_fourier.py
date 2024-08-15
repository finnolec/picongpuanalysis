import base_openpmd

import itertools


def load_electric_field():
    fields = ["Fourier Domain Fields - positive", "Fourier Domain Fields - negative"]
    components = ["Ex", "Ey"]

    ret_dict = {}

    for field, component in itertools.product(fields, components):
        ret_dict = {**ret_dict, **base_openpmd.load_vector_field_component(field, component)}
