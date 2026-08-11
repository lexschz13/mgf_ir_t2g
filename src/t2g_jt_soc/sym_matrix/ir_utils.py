from __future__ import annotations

from typing import TYPE_CHECKING
import sparse_ir as ir
from .sym_matrix import _constructor_dict

if TYPE_CHECKING:
    from typing import Any
    from .sym_matrix import _AbstractMatrix
    from ..__utils.__new_types import Sampling


def matrixfit(sampling: Sampling, M: _AbstractMatrix, **kwargs: Any) -> _AbstractMatrix:
    """
    Fits coefitients of a symmetric matrix with a sampling from sparse_ir.

    Parameters
    ----------
    sampling : Sampling
    M : _AbstractMatrix
        Matrix to fit.
    **kwargs : Any
        Sampling kwargs.

    Returns
    -------
    _AbstractMatrix

    """
    
    if not isinstance(M, _AbstractMatrix):
        raise TypeError("Object to fit must be a symmetric matrix.")
    if not isinstance(sampling, (ir.MatsubaraSampling, ir.TauSampling)):
        raise TypeError("Sampling must be a sampling objetc from sparse_ir.")
    return _constructor_dict[type(M)](*[sampling.fit(c, **kwargs) for c in M.coefs])

def matrixevaluate(sampling: Sampling, M: _AbstractMatrix, **kwargs: Any) -> _AbstractMatrix:
    """
    Evaluates coefitients of a symmetric matrix with a sampling from sparse_ir.

    Parameters
    ----------
    sampling : Sampling
    M : _AbstractMatrix
        Matrix to evaluate.
    **kwargs : Any
        Sampling kwargs.

    Returns
    -------
    _AbstractMatrix

    """
    
    if not isinstance(M, _AbstractMatrix):
        raise TypeError("Object to evaluate must be a symmetric matrix.")
    if not isinstance(sampling, (ir.MatsubaraSampling, ir.TauSampling)):
        raise TypeError("Sampling must be a sampling objetc from sparse_ir.")
    return _constructor_dict[type(M)](*[sampling.evaluate(c, **kwargs) for c in M.coefs])