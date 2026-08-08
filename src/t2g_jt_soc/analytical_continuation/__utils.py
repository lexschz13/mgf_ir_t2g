import numpy as np


def push_axis_to_zero(array, axis):
    if axis < -array.ndim or axis >= array.ndim:
        raise np.exceptions.AxisError()
    axis %= array.ndim
    if axis==0:
        return array
    return array.transpose([axis] + [i for i in range(axis)] + [i for i in range(axis+1,array.ndim)])

def correct_axis_position(array, axis):
    if axis < -array.ndim or axis >= array.ndim:
        raise np.exceptions.AxisError()
    axis %= array.ndim
    if axis==0:
        return array
    return array.transpose([i for i in range(1,axis+1)] + [0] + [i for i in range(axis+1,array.ndim)])


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