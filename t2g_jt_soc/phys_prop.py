import numpy as np
# import sparse_ir as ir
from scipy.signal import hilbert


def _push_axis_to_zero(array, axis):
    if axis < -array.ndim or axis >= array.ndim:
        raise np.exceptions.AxisError()
    axis %= array.ndim
    if axis==0:
        return array
    return array.transpose([axis] + [i for i in range(axis)] + [i for i in range(axis+1,array.ndim)])

def _correct_axis_position(array, axis):
    if axis < -array.ndim or axis >= array.ndim:
        raise np.exceptions.AxisError()
    axis %= array.ndim
    if axis==0:
        return array
    return array.transpose([i for i in range(1,axis+1)] + [0] + [i for i in range(axis+1,array.ndim)])


def conductivity_real_ir(irb, stau, smat, optsusctau, alpha=10**-1.1, axis=-1):
    iwn = np.pi/irb.beta * smat.wn
    ker_reg = lambda x: x**2/(x**2 + iwn**2)
    ker_reg_mat = smat.fit(irb.v.overlap(ker_reg).T[:,::2].real, axis=0)[::2,:].real
    ker_pinv_reg = np.linalg.inv(ker_reg_mat.T @ ker_reg_mat + alpha**2*np.eye(ker_reg_mat.shape[0])) @ ker_reg_mat.T
    if optsusctau.ndim == 1:
        optsuscl = stau.fit(optsusctau)[::2]
        return -ker_pinv_reg @ optsuscl / np.pi
   
    optsuscl = stau.fit(_push_axis_to_zero(optsusctau, axis), axis=0)[::2]
    condrel = -_correct_axis_position(np.einsum('ij,j...->i...', ker_pinv_reg, optsuscl), axis) / np.pi
    return condrel



def conductivity_from_ir(condrel, irb, omega_sample_points=1001, axis=-1, wlim=None):
    if wlim is None:
        wlim = irb.wmax
    wr = np.linspace(-wlim, wlim, omega_sample_points)
    if condrel.ndim==1:
        condrew = np.einsum('li,l->i', irb.v[::2](wr), condrel)
    else:
        condrew = _correct_axis_position(np.einsum('li,l...->i...', irb.v[::2](wr), _push_axis_to_zero(condrel, axis)), axis)
    return condrew + 1j*hilbert(condrew).imag