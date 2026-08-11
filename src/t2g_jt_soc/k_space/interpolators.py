from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from ..__utils.__utils import axes_push, axes_pull



if TYPE_CHECKING:
    from typing import Iterable, Unpack, Callable
    from numpy.typing import NDArray
    from scipy.interpolate import RegularGridInterpolator
    from ..__utils.__new_types import RealScalar



def interpBZ(f: NDArray[RealScalar], kdim: int = 3, axes: int | Iterable[int] = (-3,-2,-1),
             superdomain: int = 1, gamma: str = "bl",
             method: str = "cubic") -> Callable[[Unpack[tuple[RealScalar,...]]],NDArray[float]]:
    """
    Interpolates an array which defines a function on a reciprocal space grid.

    Parameters
    ----------
    f : NDArray[RealScalar]
        Data to interpolate.
    kdim : int, optional
        Dimension of reciprocal space.
        Greater than 0.
        The default is 3.
    axes : int | Iterable[int]
        Axes where k-dimensions are encoded.
        The default is tuple(d for d in range(-kdim,0)).
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
    if axes==-1 and kdim!=1:
        axes = tuple(d for d in range(-kdim,0))
    f = axes_push(f, axes)
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
    
    def func(*k: RealScalar | NDArray[RealScalar],
             axes: int | Iterable[int] = -1) -> NDArray[float]:
        """
        Interpolated function.

        Parameters
        ----------
        *x : RealScalar | NDArray[RealScalar]
            Lists of coordinates of evaluation points on k-space.
        axes : int | Iterable[int], optional
            Axes where k-dims must be returned.
            The default is -1.

        Returns
        -------
        NDArray[float]
            Evaluated function.

        """
        if np.all([isinstance(xi, (int,float)) for xi in k]):
            raise TypeError("All coordinate list must be scalar or array")
        if len(k) != kdim:
            raise TypeError("Expected %i arguments" % kdim)
        points = np.array(k)//2
        if gamma=='c':
            points -= 1
        if np.all([isinstance(xi, (int,float)) for xi in k]):
            points = np.array(k)
            f_int = interp(points)
            return f_int[0] * np.max(np.abs(f))
        else:
            dims = [xi.ndim if isinstance(xi, np.ndarray) else 0 for xi in k]
            reference_shape = k[dims.index(np.max(dims))].shape
            try:
                k = [xi*np.ones(reference_shape) for xi in k] # Correct shape
            except ValueError:
                raise ValueError("Some shapes cannot be broadcasted.")
            points = np.array(k)
            points = np.transpose(points, tuple([i for i in range(1,points.ndim)]) + (0,))
            f_int = interp(points)
            # This reorder k-dims to end and then these are pulled to desired axes
            return axes_pull(np.transpose(f_int,
                                          [i for i in range(-len(struct_shape),0)]
                                          + [i for i in range(points.ndim-1)]) * np.max(np.abs(f)),
                             axes)
    
    return func