import numpy as np
from numpy.typing import NDArray
from sparse_ir import FiniteTempBasis
from scipy.integrate import quad_vec
from .__utils import axis_push, axis_pull
from.__continuation_solvers import _fermion_solvers, _boson_solvers, __quasi_inv






def fermion_continuation(fl: NDArray[float], irb: FiniteTempBasis, alpha: int | float = 10**-1.1,
                         axis: int = -1, guess: None | NDArray[float] = None, solver: str = "lsql2") -> NDArray[float]:
    """
    Analytical continuation for fermionic functions in ir-basis.

    Parameters
    ----------
    fl : NDArray[float]
        Green's function matrix element. This does not support sym_matrix features.
    irb : FiniteTempBasis
        Fermionic basis from package sparse_ir.
    alpha : int | float, optional
        Regularization parameter for both, L1 and L2.
        The default is 10**-1.1.
    axis : int, optional
        Axis where ir-basis is encoded on fl.
        The default is -1.
    guess : None | NDArray[float], optional
        Initial guess of result for admm kind solvers.
        This is ignored for other solvers.
        It must have the same shape.
        None initialize the guess at 0.
        The default is None.
    solver : str, optional
        Kind of optimization problem. The implemented are:
            · lsql1: Least squares problem and LASSO (L1) regularization. Solved algebraicly.
            · lsql2: Least squares problem and Tikhonov (L2) regularization. Solved algebraicly.
            · admml1: Least squares, L1 regularization, sum rule restriction and non-negativity restriction. Uses guess. Solved with scipy minimize.
            · admml2: Least squares, L2 regularization, sum rule restriction and non-negativity restriction. Uses guess. Solved with scipy minimize.
        The default is "lsql2".

    Returns
    -------
    NDArray[float]
        Analytical continuation result. This array have the same shape than fl.
        It is assumed that user wants spectral function matrix element. To obtain imaginary parto of retarded function multiply by -1/pi.

    """
    if irb.statistics != 'F':
        raise ValueError("Continuation specific for fermions")
    return _fermion_solvers[solver](fl, irb, alpha, axis, guess)


def boson_continuation(fl: NDArray[float], irb: FiniteTempBasis, alpha: int | float = 10**-1.1,
                         axis: int = -1, guess: None | NDArray[float] = None, solver: str = "lsql2") -> NDArray[float]:
    """
    Analytical continuation for bosonic functions in ir-basis.
    This analytical continuation is done with regularized kernel, K_reg = w*K.

    Parameters
    ----------
    fl : NDArray[float]
        Correlation matrix element. This does not support sym_matrix features.
    irb : FiniteTempBasis
        Bosonic basis from package sparse_ir.
    alpha : int | float, optional
        Regularization parameter for both, L1 and L2.
        The default is 10**-1.1.
    axis : int, optional
        Axis where ir-basis is encoded on fl.
        The default is -1.
    guess : None | NDArray[float], optional
        Initial guess of result for admm kind solvers (not implemented for bosons yet).
        This is ignored for other solvers.
        It must have the same shape.
        None initialize the guess at 0.
        The default is None.
    solver : str, optional
        Kind of optimization problem. The implemented are:
            · lsql2: Least squares problem and Tikhonov (L2) regularization. Solved algebraicly.
        The default is "lsql2".

    Returns
    -------
    NDArray[float]
        Analytical continuation result. This array have the same shape than fl.
        The result is the real part of auxiliar correlator due tu the use of regularized kernel.

    """
    if irb.statistics != 'B':
        raise ValueError("Continuation specific for bosons")
    return _boson_solvers[solver](fl, irb, alpha, axis, guess)


def hilbert_ir(fl: NDArray[float], irb: FiniteTempBasis, eta: float = 1e-8, axis: int = -1) -> NDArray[float]:
    """
    Hilbert transformation in ir-basis.
    Intimately related with Kramers-Kronig relations.
    For complex funtion f(z) = f_1(z)+if_2(z),
    f_2(z) = H[f_1](z)
    f_1(z) = -H[f_2](z)

    Parameters
    ----------
    fl : NDArray[float]
        Green's object matrix element. This does not support sym_matrix features.
    irb : FiniteTempBasis
        Basis from package sparse_ir.
    eta : (int,float), optional
        Parameter to stabilize numerically the denominator of Hilbert transform integrand.
        The default is 1e-8.
    axis : int, optional
        Axis where ir-basis is encoded on fl.
        The default is -1.

    Returns
    -------
    NDArray[float]
        Hilbert transform of original function on ir-basis. This array have the same shape than fl.

    """
    fl = axis_push(fl)
    integrand = lambda w,x: np.sum(fl*irb.v(x)) * irb.v(w) * __quasi_inv(w-x)
    
    def inner_integral(x):
        return quad_vec(lambda w: integrand(w,x), -irb.wm, irb.wm)[0]
    return axis_pull(quad_vec(inner_integral, -irb.wm, irb.wm)[0])
