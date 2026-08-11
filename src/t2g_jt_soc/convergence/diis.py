from __future__ import annotations

import numpy as np
from warnings import warn
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Callable
    from ..sym_matrix.sym_matrix import _AbstractMatrix
    from ..__utils.__new_types import Scalar


def diis(vec: _AbstractMatrix[NDArray[float]], err: _AbstractMatrix[NDArray[float]],
         inner: Callable[[_AbstractMatrix,_AbstractMatrix],Scalar|NDArray[Scalar]],
         sum_op: Callable[[_AbstractMatrix | Scalar],_AbstractMatrix | Scalar] = np.sum,
         eps_reg: float = 1e-8) -> _AbstractMatrix:
    # Computation of direct inversion in the iterative subspace algorithm
    if abs(eps_reg) > 1e-5:
        warn("Too big matrix inverse regularizer can induce numerical errors.", UserWarning)
    
    mem = vec.shape[0]
    if err.shape[0] != mem:
        raise ValueError("Error vector and values vector must have same size")
    
    B = np.zeros((mem,)*2)
    for i in range(mem):
        for j in range(i,mem):
            B[i,j] = inner(err[i],err[j])
            if i != j:
                B[j,i] = np.copy(B[i,j])
    B /= np.mean(B) # Normalization
    try:
        cp = np.linalg.inv(B) @ np.ones((mem,))
    except np.linalg.LinAlgError:
        cp = np.linalg.inv(B+eps_reg*np.eye(mem)) @ np.ones((mem,))
    c = cp / np.sum(cp)
    vext = sum_op(c.reshape((mem,)+(1,)*(vec.ndim-1)) * vec, axis=0)
    vec[-1] = vext
    return vec