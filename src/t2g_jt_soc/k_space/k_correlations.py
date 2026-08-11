from __future__ import annotations

import numpy as np
from .convolution import k_convolution
from ..sym_matrix import matrixsum

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..dyson import DysonSolver
    from ..__utils.__new_types import RealScalar



def correl_BZ(dy_solver: DysonSolver) -> NDArray[RealScalar]:
    """
    Computes basic meafield correlations on k-space of a solve DysonSolver.

    Parameters
    ----------
    dy_solver : DysonSolver
        It must be solved.

    Returns
    -------
    NDArray[RealScalar]
        Array with shape (2,2,...).
        [0,0,...] <gk.a(0) gk.a(beta)>
        [0,1,...] <gk.a(0) gk.b(beta)>
        [1,0,...] <gk.b(0) gk.a(beta)>
        [1,1,...] <gk.b(0) gk.b(beta)>
        Coeffitioents corresponding to :class:"..sym_matrix.OhMatrix".

    """
    
    from ..dyson import DysonSolver
    if not isinstance(dy_solver, DysonSolver):# type(dy_solver) != DysonSolver:
        raise TypeError("Correlations on irreducible BZ are computed from a Dyson solver")
    if not dy_solver.is_solved:
        print("No solution computed, no correlations available")
        return
    
    gkzero = matrixsum(dy_solver.irbf.u(0)[:,None,None,None] * dy_solver.gkl, axis=0)
    gkbeta = matrixsum(dy_solver.irbf.u(dy_solver.beta)[:,None,None,None] * dy_solver.gkl, axis=0)
    return np.array([[k_convolution(gkzero.a, gkbeta.a), k_convolution(gkzero.a, gkbeta.b)],
                     [k_convolution(gkzero.b, gkbeta.a), k_convolution(gkzero.b, gkbeta.b)]])