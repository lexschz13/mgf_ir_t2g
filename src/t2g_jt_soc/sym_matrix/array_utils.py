import numpy as np
from .sym_matrix import _constructor_dict


def matrixsum(M, **kwargs):
    return _constructor_dict[type(M)](*[np.sum(c, **kwargs) for c in M.coefs])

def matrixcopy(M, **kwargs):
    return _constructor_dict[type(M)](*[np.copy(c, **kwargs) for c in M.coefs])