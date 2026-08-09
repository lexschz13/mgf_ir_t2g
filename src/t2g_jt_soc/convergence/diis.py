import numpy as np


def diis(vec, err, inner, sum_op = np.sum, eps_reg=1e-8):
    # Computation of direct inversion in the iterative subspace algorithm
    mem = vec.shape[0]
    if err.shape[0] != mem:
        raise ValueError("Error vector and values vector must have same size")
    
    B = np.zeros((mem,)*2)
    for i in range(mem):
        for j in range(i,mem):
            B[i,j] = inner(err[i],err[j])
            if i != j:
                B[j,i] = np.copy(B[i,j])
    B /= np.mean(B) # Normalization
    try:
        cp = np.linalg.inv(B) @ np.ones((mem,))
    except np.linalg.LinAlgError:
        cp = np.linalg.inv(B+eps_reg*np.eye(mem)) @ np.ones((mem,))
    c = cp / np.sum(cp)
    vext = sum_op(c.reshape((mem,)+(1,)*(vec.ndim-1)) * vec, axis=0)
    vec[-1] = vext
    return vec