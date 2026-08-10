import numpy as np
from ..__utils.__utils import handle_mem_error, axes_push

from collections.abc import Iterable





def k_convolution(gk: np.ndarray, fk: np.ndarray, kdim: int = 3,
                  g_axes: int | Iterable[int] = -1, f_axes: int | Iterable[int] = -1,
                  einidxs: None | str = None) -> np.ndarray:
    """
    Uses fast fourier transform (integrated on numpy) to perform convolutions on k-space.

    Parameters
    ----------
    gk : np.ndarray
        Function to convolve.
        It is assumed that k dimensions are de lasts.
    fk : np.ndarray
        Function to convolve.
        It is assumed that k dimensions are de lasts.
    kdim : int, optional
        Dimension of reciprocal space.
        Greater than 0.
        The default is 3.
    g_axes : int | Iterable[int]
        Axes where k-dimensions are encoded at first argument.
        The default is tuple(d for d in range(-kdim,0)).
    f_axes : int | Iterable[int]
        Axes where k-dimensions are encoded at second argument.
        The default is tuple(d for d in range(-kdim,0)).
    einidxs : None | str, optional
        Indices passed to :func:"np.einsum" if there is a matrix or tensor structure to be multiplied on convolution.
        If None element-wise multiplication is used.
        The default is None.

    Returns
    -------
    hk : np.ndarray
        Convolved function.
        Axes aren't pull, all k-axes are on last dimensions.

    """
    if kdim < 1:
        raise ValueError("Dimension of reciprocal space must be positive and non-zero")
    
    if g_axes==-1 and kdim!=1:
        g_axes = tuple(d for d in range(-kdim,0))
    if f_axes==-1 and kdim!=1:
        f_axes = tuple(d for d in range(-kdim,0))
    
    gk = axes_push(gk, g_axes)
    fk = axes_push(fk, f_axes)
    if gk.shape[-kdim:] != fk.shape[-kdim:]:
        raise ValueError("Shapes on k-space must be equal")
    
    ksh = gk.shape[-kdim:]
    Nk = np.prod(ksh)
    
    k_axes = tuple(i for i in range(-kdim,0))
    g_fft = np.fft.fftn(gk, axes=k_axes)
    f_fft = np.fft.fftn(fk, axes=k_axes)
    
    if einidxs:
        h_fft = np.einsum(einidxs, g_fft, f_fft, optimize=True)
        try:
            hk = np.fft.ifftn(h_fft, axes=k_axes) / Nk
        except MemoryError:
            hk = handle_mem_error(h_fft) / Nk
    else:
        h_fft = g_fft * f_fft
        hk = np.fft.ifftn(h_fft, axes=k_axes) / Nk
    return hk