import numpy as np
from scipy.interpolate import RegularGridInterpolator



def interpBZ2d(z, gamma="bl", method="cubic"):
    struct_shape = z.shape[:-2]
    k_shape = z.shape[-2:]
    if gamma == "bl":
        x = np.arange(0,2,2/k_shape[0])
        y = np.arange(0,2,2/k_shape[1])
    elif gamma == "c":
        x = np.arange(0,2,2/k_shape[0]) -1
        y = np.arange(0,2,2/k_shape[1]) -1
    else:
        raise ValueError
    expand_k_shape = tuple([s*3 for s in k_shape])
    expanded_z = np.zeros(struct_shape + expand_k_shape, dtype=z.dtype)
    ex = np.zeros(expand_k_shape[0])
    ey = np.zeros(expand_k_shape[1])
    for i in range(3):
        ex[i*k_shape[0]:(i+1)*k_shape[0]] = x + 2*(i-1)
        ey[i*k_shape[1]:(i+1)*k_shape[1]] = y + 2*(i-1)
        for j in range(3):
            expanded_z[...,i*k_shape[0]:(i+1)*k_shape[0], j*k_shape[1]:(j+1)*k_shape[1]] = np.copy(z)
    
    expanded_z = np.transpose(expanded_z, [-2,-1]+[i for i in range(expanded_z.ndim-2)]) / np.max(np.abs(z))
    
    interp = RegularGridInterpolator((ex, ey), expanded_z, method=method)
    
    def func(x, y):
        points = np.transpose(np.array([x,y]), tuple([i for i in range(1,x.ndim+1)]) + (0,))
        z_int = interp(points)
        return np.transpose(z_int, [i+1 for i in range(z_int.ndim-1)] + [0,]) * np.max(np.abs(z))
    
    return func