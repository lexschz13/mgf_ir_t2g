# Deprecated

# import numpy as np
# from npt import NDArray
# from collections.abc import Iterable





# def __handle_mem_error(h_fft):
#     # This function is to avoid memory errors
#     if h_fft.ndim <= 3:
#         raise MemoryError("Unable to allocate array for fft")
    
#     ak = []
#     for i in range(h_fft.shape[0]):
#         try:
#             ak.append(np.fft.ifftn(h_fft[i], axes=(-3,-2,-1)))
#         except MemoryError:
#             ak.append(__handle_mem_error(h_fft[i]))
#     return np.array(ak)



# def axes_push(fk: NDArray, axes: int | Iterable[int]) -> NDArray:
#     # Push k-axes to end
#     if isinstance(axes, int):
#         pass