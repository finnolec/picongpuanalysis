# import ../base_openpmd as base_openpmd
# import .base_openpmd as base_openpmd
from picongpuanalysis.loading.base_openpmd import load_vector_field_component

import itertools


def load_electric_field(path, iteration) -> dict:
    fields = ["Fourier Domain Fields - positive", "Fourier Domain Fields - negative"]
    components = ["Ex", "Ey"]

    ret_dict = {}

    for field, component in itertools.product(fields, components):
        ret_dict = {**ret_dict, field: {component: load_vector_field_component(path, iteration, field, component)}}

    return ret_dict
