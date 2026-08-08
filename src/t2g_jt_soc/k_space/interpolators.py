import numpy as np
from scipy.interpolate import RegularGridInterpolator






def interpBZ(f, kdim=1, superdomain=3, gamma="bl", method="cubic"):
    struct_shape = f.shape[:-kdim]
    k_shape = f.shape[-kdim:]
    if gamma == "bl":
        x = [np.arange(0,2,2/k_shape[i]) for i in range(kdim)]
    elif gamma == "c":
        x = [np.arange(0,2,2/k_shape[i])-1 for i in range(kdim)]
    else:
        raise ValueError
    expand_k_shape = tuple([s*superdomain for s in k_shape])
    expand_idxs = np.indices(expand_k_shape)
    # expanded_f = f[expand_idxs[0]%3, expand_idxs[1]%3, expand_idxs[2]%3]
    expanded_f = f[...,*tuple(expand_idxs[i]%superdomain for i in range(kdim))] # np.zeros(struct_shape + expand_k_shape, dtype=f.dtype)
    ex = [2*(np.arange(expand_k_shape[i])//k_shape[i]-1) + np.concatenate((x[i],)*superdomain) for i in range(kdim)]
    
    expanded_f = np.transpose(expanded_f, [i for i in range(-kdim,0)]+[i for i in range(len(struct_shape))]) / np.max(np.abs(f))
    
    interp = RegularGridInterpolator(ex, expanded_f, method=method)
    
    def func(*x):
        if len(x) != kdim:
            raise TypeError("Expected %i arguments" % kdim)
        points = np.array(x)//2
        if gamma=='c':
            points -= 1
        if np.all([isinstance(xi, (int,float)) for xi in x]):
            points = np.array(x)
            f_int = interp(points)
            return f_int[0]
        else:
            points = np.array(x)
            points = np.transpose(points, tuple([i for i in range(1,points.ndim)]) + (0,))
            f_int = interp(points)
            return np.transpose(f_int, [i for i in range(-len(struct_shape),0)] + [i for i in range(points.ndim-1)]) * np.max(np.abs(f))
    
    return func