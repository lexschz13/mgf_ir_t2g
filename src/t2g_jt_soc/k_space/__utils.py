import numpy as np





def __handle_mem_error(h_fft):
    if h_fft.ndim <= 3:
        raise MemoryError("Unable to allocate array for fft")
    
    ak = []
    for i in range(h_fft.shape[0]):
        try:
            ak.append(np.fft.ifftn(h_fft[i], axes=(-3,-2,-1)))
        except MemoryError:
            ak.append(__handle_mem_error(h_fft[i]))
    return np.array(ak)
