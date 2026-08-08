from .sym_matrix import _constructor_dict


def matrixfit(sampling, M, **kwargs):
    return _constructor_dict[type(M)](*[sampling.fit(c, **kwargs) for c in M.coefs])

def matrixevaluate(sampling, M, **kwargs):
    return _constructor_dict[type(M)](*[sampling.evaluate(c, **kwargs) for c in M.coefs])