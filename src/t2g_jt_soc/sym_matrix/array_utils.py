from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from .sym_matrix import _constructor_dict


if TYPE_CHECKING:
    from typing import Any
    from .sym_matrix import _AbstractMatrix



def matrixsum(M: _AbstractMatrix, **kwargs: Any) -> _AbstractMatrix:
    """
    Emulates :func:"np.sum" for symetric matrices.
    Applies the sum for each coefficient.

    Parameters
    ----------
    M : _AbstractMatrix
        Matrix whose elements are summed.
    **kwargs : Any
        :func:"np.sum" kwargs.

    Returns
    -------
    _AbstractMatrix

    """
    
    if not isinstance(M, _AbstractMatrix):
        raise TypeError("Object to sum must be a symmetric matrix.")
    return _constructor_dict[type(M)](*[np.sum(c, **kwargs) for c in M.coefs])

def matrixcopy(M: _AbstractMatrix, **kwargs: Any) -> _AbstractMatrix:
    """
    Emulates :func:"np.copy" for symetric matrices.
    Copies each coefficient.

    Parameters
    ----------
    M : _AbstractMatrix
        Matrix whose elements are copied.
    **kwargs : Any
        :func:"np.copy" kwargs.

    Returns
    -------
    _AbstractMatrix

    """
    
    if not isinstance(M, _AbstractMatrix):
        raise TypeError("Object to copy must be a symmetric matrix.")
    return _constructor_dict[type(M)](*[np.copy(c, **kwargs) for c in M.coefs])