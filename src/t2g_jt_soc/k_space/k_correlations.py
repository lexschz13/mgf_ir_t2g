import numpy as np
from .convolution import k_convolution
from ..sym_matrix import matrixsum



def correl_BZ(dy_solver):
    from ..dyson import DysonSolver
    if type(dy_solver) != DysonSolver:
        raise TypeError("Correlations on irreducible BZ are computed from a Dyson solver")
    if not dy_solver.is_solved:
        print("No solution computed, no correlations available")
        return
    
    gkzero = matrixsum(dy_solver.irbf.u(0)[:,None,None,None] * dy_solver.gkl, axis=0)
    gkbeta = matrixsum(dy_solver.irbf.u(dy_solver.beta)[:,None,None,None] * dy_solver.gkl, axis=0)
    return np.array([[k_convolution(gkzero.a, gkbeta.a), k_convolution(gkzero.a, gkbeta.b)],
                     [k_convolution(gkzero.b, gkbeta.a), k_convolution(gkzero.b, gkbeta.b)]])