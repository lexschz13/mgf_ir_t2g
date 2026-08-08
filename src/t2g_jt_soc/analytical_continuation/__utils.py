import numpy as np



def axis_push(array, axis):
    # Pushes axis to -1, compatible with np funcionalities
    if axis < -array.ndim or axis >= array.ndim:
        raise np.exceptions.AxisError()
    axis %= array.ndim
    if axis==array.ndim-1:
        return array
    return array.transpose([i for i in range(axis)] + [i for i in range(axis+1,array.ndim)] + [axis])


def axis_pull(array, axis):
    # Returns to original axis position
    if axis < -array.ndim or axis >= array.ndim:
        raise np.exceptions.AxisError()
    axis %= array.ndim
    if axis==array.ndim-1:
        return array
    return array.transpose([i for i in range(axis)] + [-1] + [i for i in range(axis,array.ndim-1)])