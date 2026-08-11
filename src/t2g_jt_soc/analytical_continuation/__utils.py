# Deprecated

# import numpy as np
# from numpy.typing import NDArray



# def axis_push(array: NDArray, axis: int) -> NDArray:
#     # Pushes axis to -1, compatible with np funcionalities
#     if axis < -array.ndim or axis >= array.ndim:
#         raise np.exceptions.AxisError()
#     axis %= array.ndim
#     if axis==array.ndim-1:
#         return array
#     return array.transpose([i for i in range(axis)] + [i for i in range(axis+1,array.ndim)] + [axis])


# def axis_pull(array: NDArray, axis: int) -> NDArray:
#     # Returns to original axis position
#     if axis < -array.ndim or axis >= array.ndim:
#         raise np.exceptions.AxisError()
#     axis %= array.ndim
#     if axis==array.ndim-1:
#         return array
#     return array.transpose([i for i in range(axis)] + [-1] + [i for i in range(axis,array.ndim-1)])