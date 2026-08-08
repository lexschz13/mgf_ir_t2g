import numpy as np
from scipy.integrate import quad_vec
from .__utils import axis_push, axis_pull
from.__continuation_solvers import _fermion_solvers, _boson_solvers, __quasi_inv






def fermion_continuation(fl, irb, alpha=10**-1.1, axis=-1, guess=None, solver="lsql2"):
    if irb.statistics != 'F':
        raise ValueError("Continuation specific for fermions")
    return _fermion_solvers[solver](fl, irb, alpha, axis, guess)


def boson_continuation(fl, irb, alpha=10**-1.1, axis=-1, guess=None, solver="lsql2"):
    if irb.statistics != 'B':
        raise ValueError("Continuation specific for bosons")
    return _boson_solvers[solver](fl, irb, alpha, axis, guess)


def hilbert_ir(fl, irb, eta=1e-8, axis=-1):
    # Imaginary infinitesimal denominator term added to avoid singularity
    # (x + i*0^+)**-1 = pv(x**-1) - i*pi*delta(x)
    fl = axis_push(fl)
    integrand = lambda w,x: np.sum(fl*irb.v(x)) * irb.v(w) * __quasi_inv(w-x)
    
    def inner_integral(x):
        return quad_vec(lambda w: integrand(w,x), -irb.wm, irb.wm)[0]
    return axis_pull(quad_vec(inner_integral, -irb.wm, irb.wm)[0])