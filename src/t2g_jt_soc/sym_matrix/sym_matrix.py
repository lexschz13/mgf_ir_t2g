import numpy as np


class _AbstractMatrix:
    def __init__(self, *args):
        self._coefs = args
        if np.all([isinstance(c, (int,float,complex)) for c in self._coefs]):
            self._is_array = False
        
        else:
            try:
                _coefs = [np.array(c) for c in self._coefs]
            except:
                raise TypeError
            
            if np.all([type(c.flatten()[0].item()) in [int,float,complex] for c in _coefs]):
                try:
                    _coefs = [c*np.ones_like(d) for i,c in enumerate(_coefs) for j,d in enumerate(_coefs) if i!=j]
                except:
                    raise ValueError
                self._coefs = _coefs
                self._is_array = True
            else:
                raise TypeError
    
    ###############################################
    #Auxiliar
    def __array_proprtey_check(self, p_name):
        if self._is_array:
            return
        else:
            raise AttributeError("This Matrix is a scalar, %s holds for an array" % p_name)
    
    def _apply_func(self, func):
        return
    
    def __set_coefs(self, *args):
        self._coefs = args
    
    ###############################################
    #Properties
    @property
    def coefs(self): return self._coefs
    @property
    def shape(self):
        self.__array_proprtey_check("shape")
        return self.coefs[0].shape
    @property
    def size(self):
        self.__array_proprtey_check("size")
        return self.coefs[0].size
    @property
    def ndim(self):
        self.__array_proprtey_check("ndim")
        return self.coefs[0].ndim
    @property
    def real(self): return _constructor_dict[type(self)](*[c.real for c in self.coefs])
    @property
    def imag(self): return _constructor_dict[type(self)](*[c.imag for c in self.coefs])
    @property
    def eigvals(self): return
    @property
    def trace(self): return
    
    #################################################
    #Functions
    def inv(self):
        return
        
    def exp(self):
        return self._apply_func(np.exp)
    
    def cos(self):
        return self._apply_func(np.cos)
    
    def sin(self):
        return self._apply_func(np.sin)
    
    def tan(self):
        return self._apply_func(np.tan)
    
    def log(self):
        return self._apply_func(np.log)
    
    def sqrt(self):
        return self._apply_func(np.sqrt)
    
    def cbrt(self):
        return self._apply_func(np.cbrt)
    
    ##########################################################
    #Array methods
    def reshape(self, newsh, **kwargs):
        self.__array_proprtey_check("reshape")
        return _constructor_dict[type(self)](*[c.reshape(newsh, **kwargs) for c in self.coefs])
    
    def __getitem__(self, key):
        if self._is_array:
            return _constructor_dict[type(self)](*[c[key] for c in self.coefs])
        else:
            raise IndexError("Non-array OhMatrix does not accept indexing")
    
    def __setitem__(self, key, item):
        if not self._is_array:
            raise IndexError("Non-array OhMatrix does not accept indexing")
        if type(item) != type(self):
            raise TypeError
        for i in range(len(self.coefs)):
            self._coefs[i][key] = item.coefs[i]
    
    #######################################################
    #Magic methods
    def __pos__(self):
        return _constructor_dict[type(self)](*self.coefs)
    
    def __neg__(self):
        return _constructor_dict[type(self)](*[-c for c in self.coefs])
    
    def __add__(self, other):
        return
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return
    
    def __rsub__(self, other):
        return
    
    def __isub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        return
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return
    
    def __rtruediv__(self, other):
        return
    
    def __itruediv__(self, other):
        return self.__truediv__(other)
    
    def __pow__(self, other):
        if isinstance(other, int):
            if other==1:
                return self
            elif other==0:
                return _identity_dict[type(self)](self.shape if self._is_array else None)
            elif other>1:
                return self.__pow__(other-1) * self
            elif other<0:
                return self.__pow__(-other).inv()
        elif isinstance(other, (float,complex,np.ndarray)):
            return self._apply_func(lambda x: x**other)
        else:
            raise NotImplementedError
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return
    
    def __repr__(self):
        return



class OhMatrix(_AbstractMatrix):
    def __init__(self, a, b):
        """
        Two coeffitient representation for matrices representing orbital-spin systems conserved under Oh symmetries.

        Parameters
        ----------
        a : int,float,complex,np.ndarray[int,float,complex]
            Identity coefficient.
        b : int,float,complex,np.ndarray[int,float,complex]
            SOC coefficient.

        """
        
        super().__init__(a, b)
    
    ###############################################
    #Auxiliar
    
    def _apply_func(self, func):
        c = (func(self.a+2*self.b) + 2*func(self.a-self.b))/3
        d = (func(self.a+2*self.b) - func(self.a-self.b))/3
        return ohmatrix(c, d)
    
    ###############################################
    #Properties
    @property
    def a(self): return self._coefs[0]
    @property
    def b(self): return self._coefs[1]
    @property
    def coefs(self): return [self.a, self.b]
    @property
    def eigvals(self): return self.a+2*self.b, self.a-self.b
    @property
    def trace(self): return 6*self.a
    
    #################################################
    #Functions
    def inv(self):
        denom = self.a*self.a-2*self.b*self.b+self.a*self.b
        c = (self.a+self.b)/denom
        d = -self.b/denom
        return ohmatrix(c,d)
    
    ##########################################################
    #Array methods
    
    
    #######################################################
    #Magic methods
    def __add__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a+other.a, self.b+other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a+other, self.b)
        else:
            raise TypeError
    
    def __sub__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a-other.a, self.b-other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a-other, self.b)
        else:
            raise TypeError
    
    def __rsub__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(-self.a+other.a, -self.b+other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(-self.a+other, -self.b)
        else:
            raise TypeError
    
    def __mul__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a*other.a + 2*self.b*other.b, self.b*other.a + (self.a+self.b)*other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a*other, self.b*other)
        else:
            raise TypeError
    
    def __truediv__(self, other):
        if isinstance(other, OhMatrix):
            return self * other**-1
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a/other, self.b/other)
        else:
            raise TypeError
    
    def __rtruediv__(self, other):
        if isinstance(other, OhMatrix):
            return self.inv() * other
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return self.inv() * other
        else:
            raise TypeError
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.nin == 1:
            return self._apply_func(ufunc)
        if ufunc in [np.add, np.subtract]:
            arr, ohm = inputs
            a = ufunc(arr, ohm.a, **kwargs)
            return ohmatrix(a, ohm.b)
        elif ufunc in [np.multiply, np.divide]:
            arr, ohm = inputs
            ohm = ohm.inv() if ufunc == np.divide else ohm
            a = np.multiply(arr, ohm.a, **kwargs)
            b = np.multiply(arr, ohm.b, **kwargs)
            return ohmatrix(a, b)
        else:
            raise NotImplementedError
    
    def __repr__(self):
        return f"OhMatrix({self.a}, {self.b})"


# Constructors

def ohmatrix(a, b):
    """
    Two coeffitient representation for matrices representing orbital-spin systems conserved under Oh symmetries. 
    Basic constructor function

    Parameters
    ----------
    a : int,float,complex,np.ndarray[int,float,complex]
        Identity coefficient.
    b : int,float,complex,np.ndarray[int,float,complex]
        SOC coefficient.
    
    Returns
    -------
    OhMatrix

    """
    return OhMatrix(a,b)


def ohidentity(shape=None):
    """
    Makes an OhMatrix that is an identity

    Parameters
    ----------
    shape : iterable,None
        If None returns scarlar.
        Defailt is None.
    
    Returns
    -------
    OhMatrix

    """
    if shape is None:
        return OhMatrix(1,0)
    else:
        return OhMatrix(np.ones(shape),0)


def ohzeros(shape):
    """
    Makes an array of zero OhMatrix.

    Parameters
    ----------
    shape : iterable,None
        If None returns scarlar.
        Defailt is None.
    
    Returns
    -------
    OhMatrix

    """
    if shape is None:
        return OhMatrix(0,0)
    else:
        return OhMatrix(np.zeros(shape), np.zeros(shape))


def ohrandom(shape):
    """
    Makes a random array of OhMatrix.

    Parameters
    ----------
    shape : iterable
    
    Returns
    -------
    OhMatrix

    """
    ar = np.random.random(shape)
    ap = 2*np.pi*np.random.random(shape)
    br = np.random.random(shape)
    bp = 2*np.pi*np.random.random(shape)
    a = ar*np.exp(1j*ap)
    b = br*np.exp(1j*bp)
    return OhMatrix(a,b)


# Constructor-Type dictionaries

_constructor_dict = {OhMatrix:ohmatrix}
_identity_dict    = {OhMatrix:ohidentity}
_zeros_dict       = {OhMatrix:ohzeros}
_random_dict      = {OhMatrix:ohrandom}