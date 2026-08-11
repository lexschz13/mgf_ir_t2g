import numpy as np
import sparse_ir as ir
from .magneto import s0, gell_mann, Gspin_proj
from .reciprocal_space import k_convolution
from .phys_prop import correl_cplx_ir
from .bz_interpolators import interpBZ2d




def gell_mann_corr_maps(G, G1, la, lb, irbb):
    ksz = G.shape[-1]
    
    la = np.kron(la, s0)
    lb = np.kron(lb, s0)
    
    minus_k = (-np.arange(ksz)) % ksz
    A = np.einsum("ab,bc...,cd->ad...", la, G, lb, optimize=True)
    Aflipp = A[...,minus_k.reshape((ksz,1,1)),
                   minus_k.reshape((1,ksz,1)),
                   minus_k.reshape((1,1,ksz))]
    
    return k_convolution(Aflipp, -G1[:,:,::-1,...], "ab...,ba...->...")


def coef_mat(gamma, delta):
    cp = np.array([(2-4*np.cos(2*gamma))/3, #0
                   np.sin(gamma)*np.sin(delta), #1
                   -1j*np.cos(gamma)*np.cos(delta), #2
                   (np.cos(2*gamma)+np.cos(2*delta)-2)/4, #3
                   -np.sin(gamma)*np.cos(delta), #4
                   -1j*np.cos(gamma)*np.sin(delta), #5
                   0.5*np.sin(2*delta), #6
                   0.5j*np.sin(2*gamma), #7
                   (np.cos(2*gamma)-3*np.cos(2*delta)-2)/4/np.sqrt(3), #8
                   ])
    
    cm = np.array([(2-4*np.cos(2*gamma))/3, #0
                   -np.sin(gamma)*np.sin(delta), #1
                   1j*np.cos(gamma)*np.cos(delta), #2
                   (np.cos(2*gamma)+np.cos(2*delta)-2)/4, #3
                   np.sin(gamma)*np.cos(delta), #4
                   1j*np.cos(gamma)*np.sin(delta), #5
                   0.5*np.sin(2*delta), #6
                   0.5j*np.sin(2*gamma), #7
                   (np.cos(2*gamma)-3*np.cos(2*delta)-2)/4/np.sqrt(3), #8
                   ])
    
    return cp.conj()[:,None]*cp[None,:] - cm.conj()[:,None]*cm[None,:]



def create_gm_map(irbb, Gkl, beta, t, angle, ejt, irbf, jt_mode):
    Gktau, G1ktau = Gspin_proj(Gkl, beta, t, angle, ejt, irbf, jt_mode)
    gm_map = []
    for a in range(9):
        for b in range(a+1,9):
            gm_map.append(gell_mann_corr_maps(Gktau, G1ktau, gell_mann[a], gell_mann[b], irbb)[:,0,:,:]) # Removed kx axis
    
    return np.array(gm_map)


def rixs_calc(irbb, gm_map, alpha_i, alpha_f, rwl, wlim=1, life_time=0, w_sample_points=501,
              interp_func=interpBZ2d):
    staub = ir.TauSampling(irbb)
    smatb = ir.MatsubaraSampling(irbb)
    
    f = lambda a,b: np.sum(np.arange(8,8-a,-1)) + (b-a) - 1 # map idx
    
    gamma = 0.5*(alpha_i + alpha_f)
    delta = 0.5*(alpha_i - alpha_f)
    qy = -4/rwl * np.sin(gamma) * np.cos(delta) % 2
    qz = 4/rwl * np.sin(gamma) * np.sin(delta) % 2
    
    C = coef_mat(gamma, delta)
    if C.ndim == 2:
        C = C.reshape(C.shape + (1,))
    corrqtau = np.zeros((C.shape[-1],)+gm_map[0].shape, dtype=gm_map.dtype)
    print("Computing RIXS from gell-mann maps")
    for a in range(9):
        for b in range(9):
            if a < b:
                corrqtau += np.einsum("...,tkl->...tkl", C[b,a], gm_map[f(a,b)], optimize=True) + np.einsum("...,tkl->...tkl", C[a,b].conj(), gm_map[f(a,b),::-1], optimize=True)
            elif a > b:
                corrqtau += np.einsum("...,tkl->...tkl", C[b,a], gm_map[f(b,a)].conj(), optimize=True) + np.einsum("...,tkl->...tkl", C[a,b].conj(), gm_map[f(b,a),::-1].conj(), optimize=True)
    
    corrtau = np.einsum("iji->ji", interp_func(np.squeeze(corrqtau.real))(qy, qz), optimize=True)
    print("Transforming to frequency")
    corrl = correl_cplx_ir(irbb, staub, smatb, corrtau, axis=0).real
    wr = np.linspace(0, wlim, w_sample_points)
    corrw = (irbb.v(wr).T @ corrl) * wr[:,None] / ((wr[:,None]*life_time)**2 + 1)
    return wr, gamma, corrw