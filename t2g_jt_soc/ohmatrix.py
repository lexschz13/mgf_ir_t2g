import numpy as np


class OhMatrix:
    def __init__(self, a, b):
        if isinstance(a, (int,float,complex)) and isinstance(b, (int,float,complex)):
            self.__a = a
            self.__b = b
            self.__is_array = False
        try:
            _a = np.array(a)
            _b = np.array(b)
        except:
            raise TypeError
        if (type(_a.flatten()[0].item()) in [int,float,complex]) and (type(_b.flatten()[0].item()) in [int,float,complex]):
            try:
                _a = _a*np.ones_like(_b)
                _b = _b*np.ones_like(_a)
            except:
                raise ValueError
            self.__a = _a
            self.__b = _b
            self.__is_array = True
        else:
            raise TypeError
    
    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b
    
    @property
    def shape(self):
        return np.asarray(self.a).shape
    
    @property
    def size(self):
        return np.asarray(self.a).size
    
    @property
    def ndim(self):
        return np.asarray(self.a).ndim
    
    @property
    def real(self):
        return ohmatrix(self.a.real, self.b.real)
    
    @property
    def imag(self):
        return ohmatrix(self.a.imag, self.b.imag)
    
    @property
    def eigvals(self):
        return self.a+2*self.b, self.a-self.b
    
    @property
    def trace(self):
        return 6*self.a
    
    def inv(self):
        denom = self.a*self.a-2*self.b*self.b+self.a*self.b
        c = (self.a+self.b)/denom
        d = -self.b/denom
        return ohmatrix(c,d)
        
    def __apply_func(self, func):
        c = (func(self.a+2*self.b) + 2*func(self.a-self.b))/3
        d = (func(self.a+2*self.b) - func(self.a-self.b))/3
        return ohmatrix(c, d)
    
    def exp(self):
        return self.__apply_func(np.exp)
    
    def cos(self):
        return self.__apply_func(np.cos)
    
    def sin(self):
        return self.__apply_func(np.sin)
    
    def tan(self):
        return self.__apply_func(np.tan)
    
    def log(self):
        return self.__apply_func(np.log)
    
    def sqrt(self):
        return self.__apply_func(np.sqrt)
    
    def cbrt(self):
        return self.__apply_func(np.cbrt)
    
    def reshape(self, newsh):
        a = np.asarray(self.a)
        b = np.asarray(self.b)
        return ohmatrix(a.reshape(newsh), b.reshape(newsh))
    
    def __getitem__(self, key):
        return ohmatrix(self.a[key], self.b[key])
    
    def __setitem__(self, key, item):
        if not isinstance(item, OhMatrix):
            raise TypeError
        self.__a[key] = item.a
        self.__b[key] = item.b
    
    def __pos__(self):
        return OhMatrix(self.a,self.b)
    
    def __neg__(self):
        return OhMatrix(-self.a,-self.b)
    
    def __add__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a+other.a, self.b+other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a+other, self.b)
        else:
            raise TypeError
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        return self.__add__(other)
    
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
    
    def __isub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a*other.a + 2*self.b*other.b, self.b*other.a + (self.a+self.b)*other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a*other, self.b*other)
        else:
            raise TypeError
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        return self.__mul__(other)
    
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
    
    def __itruediv__(self, other):
        return self.__truediv__(other)
    
    def __pow__(self, other):
        if isinstance(other, int):
            if other==1:
                return self
            elif other==0:
                return OhMatrix(1,0)
            elif other>1:
                return self.__pow__(other-1) * self
            elif other<0:
                return self.__pow__(-other).inv()
        elif isinstance(other, (float,complex,np.ndarray)):
            return self.__apply_func(lambda x: x**other)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.nin == 1:
            return self.__apply_func(ufunc)
        if ufunc in [np.add, np.subtract]:
            arr, ohm = inputs
            a = ufunc(arr, ohm.a, **kwargs)
            return ohmatrix(a, ohm.b)
        elif ufunc in [np.multiply, np.divide]:
            arr, ohm = inputs
            ohm = ohm.inv() if ufunc == np.divide else ohm
            a = arr*ohm.a
            b = arr*ohm.b
            return ohmatrix(a, b)
        else:
            raise NotImplementedError
    
    def __repr__(self):
        return f"OhMatrix({self.a}, {self.b})"



# Constructors

def ohmatrix(a, b):
    return OhMatrix(a,b)

def ohzeros(shape):
    return OhMatrix(np.zeros(shape), np.zeros(shape))

def ohrandom(shape):
    ar = np.random.random(shape)
    ap = 2*np.pi*np.random.random(shape)
    br = np.random.random(shape)
    bp = 2*np.pi*np.random.random(shape)
    a = ar*np.exp(1j*ap)
    b = br*np.exp(1j*bp)
    return OhMatrix(a,b)



# Auxiliars

def ohsum(M, **kwargs):
    return ohmatrix(np.sum(M.a, **kwargs), np.sum(M.b, **kwargs))

def ohcopy(M, **kwargs):
    return ohmatrix(np.copy(M.a), np.copy(M.b))