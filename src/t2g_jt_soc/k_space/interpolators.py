import numpy as np
from scipy.interpolate import RegularGridInterpolator






def interpBZ(f: np.ndarray, kdim: int = 3, superdomain: int = 1, gamma: str = "bl", method: int = "cubic") -> callable:
    """
    

    Parameters
    ----------
    f : np.ndarray
        Function to interpolate.
        It is assumed that k-dimensions are the lasts.
    kdim : int, optional
        Dimension of reciprocal space.
        Greater than 0.
        The default is 3.
    superdomain : int, optional
        Number of cells expanden at every direction to simulate periodicity.
        It must be odd, so it is corrected with -1 if it is even.
        Greater-equal 0.
        The default is 1.
    gamma : str, optional
        Position of Gamma point, i.e. k=(0,0,0,...).
        Only allowed border left "bl" and center "c".
        The default is "bl".
    method : str, optional
        Interpolation mode for :class:"RegularGridInterpolator".
        The default is "cubic".

    Returns
    -------
    callable
        Interpolated function.
        It accepts kdim arguments being all scalar or collection of scalars and arrays as lists of k1,k2,k3,... coordinates.

    """
    if kdim < 1:
        raise ValueError("Dimension of reciprocal space must be positive and non-zero")
    struct_shape = f.shape[:-kdim]
    k_shape = f.shape[-kdim:]
    if gamma == "bl":
        x = [np.arange(0,2,2/k_shape[i]) for i in range(kdim)]
    elif gamma == "c":
        x = [np.arange(0,2,2/k_shape[i])-1 for i in range(kdim)]
    else:
        raise ValueError("Position of Gamma only allows border left and center")
    
    superdomain = superdomain-1 if superdomain%2==0 else superdomain
    expand_k_shape = tuple([s*superdomain for s in k_shape])
    expand_idxs = np.indices(expand_k_shape)
    # expanded_f = f[expand_idxs[0]%3, expand_idxs[1]%3, expand_idxs[2]%3]
    expanded_f = f[...,*tuple(expand_idxs[i]%superdomain for i in range(kdim))] # np.zeros(struct_shape + expand_k_shape, dtype=f.dtype)
    ex = [2*(np.arange(expand_k_shape[i])//k_shape[i]-1) + np.concatenate((x[i],)*superdomain) for i in range(kdim)]
    
    # Normalization to avoid interpolation errors
    # It is corrected in interpolated funciton
    expanded_f = np.transpose(expanded_f, [i for i in range(-kdim,0)]+[i for i in range(len(struct_shape))]) / np.max(np.abs(f))
    
    interp = RegularGridInterpolator(ex, expanded_f, method=method)
    
    def func(*x):
        if np.all([isinstance(xi, (int,float)) for xi in x]):
            raise TypeError("All coordinate list must be scalar or array")
        if len(x) != kdim:
            raise TypeError("Expected %i arguments" % kdim)
        points = np.array(x)//2
        if gamma=='c':
            points -= 1
        if np.all([isinstance(xi, (int,float)) for xi in x]):
            points = np.array(x)
            f_int = interp(points)
            return f_int[0] * np.max(np.abs(f))
        else:
            dims = [xi.ndim if isinstance(xi, np.ndarray) else 0 for xi in x]
            reference_shape = x[dims.index(np.max(dims))].shape
            try:
                x = [xi*np.ones(reference_shape) for xi in x] # Correct shape
            except ValueError:
                raise ValueError("Some shapes cannot be broadcasted.")
            points = np.array(x)
            points = np.transpose(points, tuple([i for i in range(1,points.ndim)]) + (0,))
            f_int = interp(points)
            return np.transpose(f_int, [i for i in range(-len(struct_shape),0)] + [i for i in range(points.ndim-1)]) * np.max(np.abs(f))
    
    return func