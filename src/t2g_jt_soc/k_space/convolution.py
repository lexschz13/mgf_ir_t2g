import numpy as np
from .__utils import __handle_mem_error



def k_convolution(gk, fk, einidxs=None):
    if gk.shape[-3:] != fk.shape[-3:]:
        raise ValueError("Shapes on k-space must be equal")
    
    ksh = gk.shape[-3:]
    Nk = np.prod(ksh)
    
    g_fft = np.fft.fftn(gk, axes=(-3,-2,-1))
    f_fft = np.fft.fftn(fk, axes=(-3,-2,-1))
    
    if einidxs:
        h_fft = np.einsum(einidxs, g_fft, f_fft, optimize=True)
        try:
            hk = np.fft.ifftn(h_fft, axes=(-3,-2,-1)) / Nk
        except MemoryError:
            hk = __handle_mem_error(h_fft) / Nk
    else:
        h_fft = g_fft * f_fft
        hk = np.fft.ifftn(h_fft, axes=(-3,-2,-1)) / Nk
    return hk