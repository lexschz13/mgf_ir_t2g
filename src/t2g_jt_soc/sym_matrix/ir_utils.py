from sparse_ir import MatsubaraSampling, TauSampling
Sampling = MatsubaraSampling | TauSampling
from .sym_matrix import _constructor_dict, _AbstractMatrix

from typing import Any


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
    if not isinstance(sampling, (MatsubaraSampling, TauSampling)):
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
    if not isinstance(sampling, (MatsubaraSampling, TauSampling)):
        raise TypeError("Sampling must be a sampling objetc from sparse_ir.")
    return _constructor_dict[type(M)](*[sampling.evaluate(c, **kwargs) for c in M.coefs])