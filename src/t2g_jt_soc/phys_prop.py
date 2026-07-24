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


def conductivity_real_ir(irb, stau, smat, optsusctau, alpha=10**-1.1, axis=-1, get_err=False, eps=None):
    iwn = np.pi/irb.beta * smat.wn
    ker_reg = lambda x: x**2/(x**2 + iwn**2)
    ker_reg_mat = smat.fit(irb.v.overlap(ker_reg).T[:,::2].real, axis=0)[::2,:].real
    # if get_err:
    #     sl = np.linalg.eigvals(ker_reg_mat)
    ker_pinv_reg = np.linalg.inv(ker_reg_mat.T @ ker_reg_mat + alpha**2*np.eye(ker_reg_mat.shape[0])) @ ker_reg_mat.T
    if optsusctau.ndim == 1:
        optsuscl = stau.fit(optsusctau)[::2]
        condrel = -ker_pinv_reg @ optsuscl / np.pi
    else:
        optsuscl = stau.fit(_push_axis_to_zero(optsusctau, axis), axis=0)[::2]
        condrel = -_correct_axis_position(np.einsum('ij,j...->i...', ker_pinv_reg, optsuscl), axis) / np.pi
    
    if get_err:
        if eps is None:
            eps = alpha * 1e-1
        # errl = np.abs((alpha**2 - sl**2)/(alpha**2 + sl**2)) * eps * np.abs(sl)
        if optsusctau.ndim == 1:
            errl = np.sqrt((ker_pinv_reg)**2 @ (eps*optsuscl)**2)/np.pi
        else:
            errl = np.sqrt(_correct_axis_position(np.einsum('ij,j...->i...', ker_pinv_reg**2, (eps*optsuscl)**2), axis))/np.pi
        return condrel, errl
    
    else:
        return condrel


def conductivity_imag_ir(irb, stau, smat, susctau, alpha=10**-1.1, axis=-1, get_err=False, eps=None):
    iwn = np.pi/irb.beta * smat.wn
    # ker_reg = w/(iwn - w) = -w**2(wn**2 + w**2) - iwn*w/(wn**2 + w**2)
    ker_reg_odd  = lambda x: x*iwn/(x**2 + iwn**2)
    ker_reg_mat = np.zeros((irb.size, irb.size))
    # ker_reg_mat[::2,::2] = smat.fit(irb.v.overlap(ker_reg_even).T[:,::2].real, axis=0)[::2,:].real
    ker_reg_mat = -smat.fit(irb.v.overlap(ker_reg_odd).T[:,1::2].real, axis=0)[1::2,:].imag
    if get_err:
        sl = np.linalg.eigvals(ker_reg_mat)
    ker_pinv_reg = np.linalg.inv(ker_reg_mat.T @ ker_reg_mat + alpha**2*np.eye(ker_reg_mat.shape[0])) @ ker_reg_mat.T
    if susctau.ndim == 1:
        suscl = stau.fit(susctau)[1::2]
        corrl = -ker_pinv_reg @ suscl / np.pi
    else:
        suscl = stau.fit(_push_axis_to_zero(susctau, axis), axis=0)[1::2]
        corrl = -_correct_axis_position(np.einsum('ij,j...->i...', ker_pinv_reg, suscl), axis) / np.pi
    
    if get_err:
        if eps is None:
            eps = alpha * 1e-1 * np.abs(sl)
        errl = np.abs((alpha**2 - sl**2)/(alpha**2 + sl**2)) * eps
        return corrl, errl
    
    else:
        return corrl


def correl_cplx_ir(irb, stau, smat, susctau, alpha=10**-1.1, axis=-1, get_err=False, eps=None):
    iwn = np.pi/irb.beta * smat.wn
    # ker_reg = w/(iwn - w) = -w**2(wn**2 + w**2) - iwn*w/(wn**2 + w**2)
    ker_reg_even = lambda x: x**2/(x**2 + iwn**2)
    ker_reg_odd  = lambda x: x*iwn/(x**2 + iwn**2)
    ker_reg_mat = np.zeros((irb.size, irb.size))
    ker_reg_mat[::2,::2] = smat.fit(irb.v.overlap(ker_reg_even).T[:,::2].real, axis=0)[::2,:].real
    ker_reg_mat[1::2,1::2] = -smat.fit(irb.v.overlap(ker_reg_odd).T[:,1::2].real, axis=0)[1::2,:].imag
    if get_err:
        sl = np.linalg.eigvals(ker_reg_mat)
    ker_pinv_reg = np.linalg.inv(ker_reg_mat.T @ ker_reg_mat + alpha**2*np.eye(ker_reg_mat.shape[0])) @ ker_reg_mat.T
    if susctau.ndim == 1:
        suscl = stau.fit(susctau)
        corrl = -ker_pinv_reg @ suscl / np.pi
    else:
        suscl = stau.fit(_push_axis_to_zero(susctau, axis), axis=0)
        corrl = -_correct_axis_position(np.einsum('ij,j...->i...', ker_pinv_reg, suscl), axis) / np.pi
    
    if get_err:
        if eps is None:
            eps = alpha * 1e-1 * np.abs(sl)
        errl = np.abs((alpha**2 - sl**2)/(alpha**2 + sl**2)) * eps
        return corrl, errl
    
    else:
        return corrl



def conductivity_from_ir(condrel, irb, omega_sample_points=1001, axis=-1, wlim=None, parity=None):
    if wlim is None:
        wlim = irb.wmax
    wr = np.linspace(-wlim, wlim, omega_sample_points)
    if condrel.ndim==1:
        if parity in ['e',"even"]:
            condrew = np.einsum('li,l->i', irb.v[::2](wr), condrel)
        elif parity in ['o',"odd"]:
            condrew = np.einsum('li,l->i', irb.v[1::2](wr), condrel)
        else:
            condrew = np.einsum('li,l->i', irb.v(wr), condrel)
    else:
        if parity in ['e',"even"]:
            condrew = _correct_axis_position(np.einsum('li,l...->i...', irb.v[::2](wr), _push_axis_to_zero(condrel, axis)), axis)
        elif parity in ['o',"odd"]:
            condrew = _correct_axis_position(np.einsum('li,l...->i...', irb.v[1::2](wr), _push_axis_to_zero(condrel, axis)), axis)
        else:
            condrew = _correct_axis_position(np.einsum('li,l...->i...', irb.v(wr), _push_axis_to_zero(condrel, axis)), axis)
    return condrew + 1j*hilbert(condrew, axis=axis).imag