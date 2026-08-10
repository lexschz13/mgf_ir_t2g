from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING
from collections.abc import Iterable
from io import TextIOWrapper

if TYPE_CHECKING:
    from ..sym_matrix.sym_matrix import _AbstractMatrix



###############################################################################
# Axes accomodation

def axis_push(array: NDArray, axis: int) -> NDArray:
    # Pushes axis to -1, compatible with np funcionalities
    if axis < -array.ndim or axis >= array.ndim:
        raise np.exceptions.AxisError()
    axis %= array.ndim
    if axis==array.ndim-1:
        return array
    return array.transpose([i for i in range(axis)] + [i for i in range(axis+1,array.ndim)] + [axis])


def axis_pull(array: NDArray, axis: int) -> NDArray:
    # Returns to original axis position
    if axis < -array.ndim or axis >= array.ndim:
        raise np.exceptions.AxisError()
    axis %= array.ndim
    if axis==array.ndim-1:
        return array
    return array.transpose([i for i in range(axis)] + [-1] + [i for i in range(axis,array.ndim-1)])


def axes_push(array: NDArray, axes: int | Iterable[int]) -> NDArray:
    # Push axes to end
    if isinstance(axes, int):
        return axis_push(array, axes)
    if len(axes) == 1:
        return axis_push(array, axes[0])
    
    if tuple(a%array.ndim for a in axes) == tuple(i for i in range(array.ndim-len(axes),array.ndim)):
        return array
    
    axes = tuple(a%array.ndim for a in axes)
    struct_axes = tuple(a for a in range(array.ndim) if not a in axes)
    
    return np.transpose(array, struct_axes+axes)


def axes_pull(array: NDArray, axes: int | Iterable[int]) -> NDArray:
    # Push axes to end
    if isinstance(axes, int):
        return axis_pull(array, axes)
    if len(axes) == 1:
        return axis_pull(array, axes[0])
    
    if tuple(a%array.ndim for a in axes) == tuple(i for i in range(array.ndim-len(axes),array.ndim)):
        return array
    
    axes = tuple(a%array.ndim for a in axes)
    # struct_axes = tuple(a for a in range(array.ndim) if not a in axes)
    transposition = []
    k = 0
    for a in range(array.ndim):
        if a in axes:
            transposition += [k-len(axes)]
            k += 1
        else:
            transposition += [a-k]
    
    return np.transpose(array, transposition)


###############################################################################
# Check funcs

def check_physical_param(p: int | float,
                         minp: int | float = 0,
                         maxp: int | float = np.inf,
                         text_value_error: str = "",
                         text_type_error: str = "") -> int | float:
    if not isinstance(p, (int,float)): raise TypeError(text_type_error)
    if p < minp or p > maxp: raise ValueError(text_value_error)
    
    return p


def check_discrete_parameter(p: int,
                             minp: int = 1,
                             maxp: int | float = np.inf,
                             text_value_error: str = "",
                             text_type_error: str = "") -> int:
    if not isinstance(p, int): raise TypeError(text_type_error)
    if p < minp or p > maxp: raise ValueError(text_value_error)
    
    return p


def check_shape(p: Iterable,
                l: int = 3,
                text_value_error: str = "",
                text_type_error: str = "") -> Iterable:
    if isinstance(p, Iterable):
        if isinstance(p, np.ndarray): p = p.flatten()
        if len(p)==l: pass
        elif len(p)==1: p *= l
        else: raise ValueError(text_value_error)
    elif isinstance(p, int): p = (p,)*l
    else: raise TypeError(text_type_error)
    return p

###############################################################################
# Other

def fprint(string: str, file: TextIOWrapper, **kwargs: Any) -> None:
    # Print on output file
    print(string, **kwargs)
    print(string, file=file, **kwargs)


def frobenius_inner(X: _AbstractMatrix, Y: _AbstractMatrix) -> float | complex | NDArray[float] | NDArray[complex]:
    return (X*Y).trace


def handle_mem_error(h_fft: NDArray) -> NDArray:
    # This function is to avoid memory errors
    if h_fft.ndim <= 3:
        raise MemoryError("Unable to allocate array for fft")
    
    ak = []
    for i in range(h_fft.shape[0]):
        try:
            ak.append(np.fft.ifftn(h_fft[i], axes=(-3,-2,-1)))
        except MemoryError:
            ak.append(handle_mem_error(h_fft[i]))
    return np.array(ak)