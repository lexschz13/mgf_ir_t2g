from numpy import np
from scipy.optimize import minimize
from scipy.integrate import quad_vec, quad
from .__utils import axis_pull, axis_push


def __quasi_abs(x, k=1e-8):
    # return k*np.log(np.cosh(x/k))
    return x*__quasi_sign(x,k) - k*np.loc(x**2+k**2)/(2*np.pi)

def __quasi_sign(x, k=1e-8):
    # return np.tanh(x/k)
    return np.arctan(x/k)/np.pi

def __quasi_heaviside(x, k=1e-8):
    return __quasi_sign(x,k)*0.5+0.5

def __quasi_delta(x, k=1e-8):
    # return np.cosh(x/k)**-2
    return k/np.pi*(x**2+k**2)**-1

def __quasi_inv(x, k=1e-8):
    return x/(x**2+k**2)


def fermion_lsq_l1(fl, irb, alpha, axis=-1, guess=None):
    if irb.statistics != 'F':
        raise ValueError("Continuation specific for fermions")
    if fl.ndim ==1:
        return (alpha*np.sgn(fl)-irb.s*fl) / irb.s**2
    fl = axis_push(fl)
    return axis_pull((alpha*np.sgn(fl)-irb.s*fl) / irb.s**2)
    


def fermion_lsq_l2(fl, irb, alpha, axis=-1, guess=None):
    if irb.statistics != 'F':
        raise ValueError("Continuation specific for fermions")
    if fl.ndim ==1:
        return -irb.s / (alpha**2 + irb.s**2) * fl
    return axis_pull(-irb.s / (alpha**2 + irb.s**2) * axis_push(fl, axis), axis)


def __fermion_admm_l1_single(fl, irb, alpha, guess=None):
    if fl.ndim != 1:
        raise TypeError("Only 1D array minimization allowed")
    
    def objective(x):
        return 0.5*np.sum((irb.s*x + fl)**2) + alpha*__quasi_abs(x)
    
    def jac(x):
        return irb.s*(irb.s*x+fl) + alpha*__quasi_sign(x)
    
    ubeta = irb.u(irb.beta)
    vweight = quad_vec(lambda w: irb.v(w), -irb.wmax, irb.wmax)
    
    sum_rule_constr = {"type": "eq",
                       "fun": lambda x: np.sum(vweight*x + ubeta*fl),
                       "jac": lambda x: vweight}
    
    density_theta = lambda w,x: __quasi_heaviside(-np.sum(irb.v(w)*x))
    density_delta = lambda w,x: __quasi_delta(-np.sum(irb.v(w)*x))
    non_neg_constraint = {"type": "eq",
                          "fun": lambda x: quad(lambda w: density_theta(w,x), -irb.wmax, irb.wmax),
                          "jac": lambda x: quad_vec(lambda w: density_delta(w,x) - density_theta(w,x)*irb.v(w), -irb.wmax, irb.wmax)}
    
    if guess is None:
        guess = -np.sum(ubeta*fl) / np.sum(vweight) * np.ones(fl.size)
    
    res = minimize(
        objective,
        guess,
        jac=jac,
        method="SLSQP",
        constraints=[sum_rule_constr, non_neg_constraint]
        )
    
    return res.x


def fermion_admm_l1(fl, irb, alpha, axis=-1, guess=None):
    if irb.statistics != 'F':
        raise ValueError("Continuation specific for fermions")
    if fl.ndim == 1:
        return __fermion_admm_l1_single(fl, irb, alpha, axis, guess)
    fl = axis_push(fl, axis)
    pushed_shape = fl.shape
    fl = fl.reshape((-1, fl.ndim-1))
    ret = np.zeros_like(fl)
    for i in range(fl.shape[0]):
        ret[i] = __fermion_admm_l1_single(fl[i], irb, alpha, axis, guess)
    ret = ret.reshape(pushed_shape)
    return axis_pull(ret)


def __fermion_admm_l2_single(fl, irb, alpha, guess=None):
    if fl.ndim != 1:
        raise TypeError("Only 1D array minimization allowed")
    
    def objective(x):
        return 0.5*np.sum((irb.s*x + fl)**2) + 0.5*alpha**2*np.sum(x**2)
    
    def jac(x):
        return irb.s*(irb.s*x+fl) + alpha**2*x
    
    ubeta = irb.u(irb.beta)
    vweight = quad_vec(lambda w: irb.v(w), -irb.wmax, irb.wmax)
    
    sum_rule_constr = {"type": "eq",
                       "fun": lambda x: np.sum(vweight*x + ubeta*fl),
                       "jac": lambda x: vweight}
    
    density_theta = lambda w,x: __quasi_heaviside(-np.sum(irb.v(w)*x))
    density_delta = lambda w,x: __quasi_delta(-np.sum(irb.v(w)*x))
    non_neg_constraint = {"type": "eq",
                          "fun": lambda x: quad(lambda w: density_theta(w,x), -irb.wmax, irb.wmax),
                          "jac": lambda x: quad_vec(lambda w: density_delta(w,x) - density_theta(w,x)*irb.v(w), -irb.wmax, irb.wmax)}
    
    if guess is None:
        guess = -np.sum(ubeta*fl) / np.sum(vweight) * np.ones(fl.size)
    
    res = minimize(
        objective,
        guess,
        jac=jac,
        method="SLSQP",
        constraints=[sum_rule_constr, non_neg_constraint]
        )
    
    return res.x

def fermion_admm_l2(fl, irb, alpha, axis=-1, guess=None):
    if irb.statistics != 'F':
        raise ValueError("Continuation specific for fermions")
    if fl.ndim == 1:
        return __fermion_admm_l2_single(fl, irb, alpha, axis, guess)
    fl = axis_push(fl, axis)
    pushed_shape = fl.shape
    fl = fl.reshape((-1, fl.ndim-1))
    ret = np.zeros_like(fl)
    for i in range(fl.shape[0]):
        ret[i] = __fermion_admm_l2_single(fl[i], irb, alpha, axis, guess)
    ret = ret.reshape(pushed_shape)
    return axis_pull(ret)
    
    



_fermion_solvers = {
    "lsql1": fermion_lsq_l1,
    "admml1": fermion_admm_l1,
    "lsql2": fermion_lsq_l2,
    "admml2": fermion_admm_l2
    }



def __reg_kernel_ir(irb):
    return quad_vec(lambda w: w*irb.v(w)[:,None]*irb.v[None,:])


def boson_lsq_l2(fl, irb, alpha=10**-1.1, axis=-1, guess=None):
    if irb.statistics != 'B':
        raise ValueError("Continuation specific for bosons")
    kmat = __reg_kernel_ir(irb)
    kmat_inv_l2reg = np.linalg.inv(kmat.T @ kmat + alpha**2 * np.eye(kmat.shape[0])) @ kmat.T
    
    fl = axis_push(fl, axis)
    return axis_pull(kmat_inv_l2reg @ fl) / np.pi




_boson_solvers = {
    "lsql2": boson_lsq_l2
    }
