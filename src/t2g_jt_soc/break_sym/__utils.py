import numpy as np


def tau_triple_conv(smat, stau, a,b,c, sum_idx, **kwargs):
    # Arguments on freq space
    d = np.einsum(sum_idx, a, b, c, optimize=True)
    return stau.evaluate(smat.fit(d, **kwargs).real, **kwargs)