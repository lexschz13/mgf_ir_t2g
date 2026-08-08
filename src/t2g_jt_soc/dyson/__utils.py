import numpy as np


def fprint(string, file, **kwargs):
    print(string, **kwargs)
    print(string, file=file, **kwargs)


def frobenius_inner(X, Y):
    return (X*Y).trace


def check_physical_param(p,
                         minp=0,
                         maxp=np.inf,
                         text_value_error = "",
                         text_type_error = ""):
    if not isinstance(p, (int,float)): raise TypeError(text_type_error)
    if p < minp or p > maxp: raise ValueError(text_value_error)
    
    return p


def check_discrete_parameter(p,
                             minp=1,
                             maxp=np.inf,
                             text_value_error = "",
                             text_type_error = ""):
    if not isinstance(p, int): raise TypeError(text_type_error)
    if p < minp or p > maxp: raise ValueError(text_value_error)
    
    return p


def check_shape(p,
                l = 3,
                text_value_error = "",
                text_type_error = ""):
    if isinstance(p, (list,tuple,np.ndarray)):
        if isinstance(p, np.ndarray): p = p.flatten()
        if len(p)==l: pass
        elif len(p)==1: p *= l
        else: raise ValueError(text_value_error)
    elif isinstance(p, int): p = (p,)*3
    else: raise TypeError(text_type_error)
    return p
