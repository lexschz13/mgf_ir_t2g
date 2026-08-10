import numpy as np
from numpy.typing import NDArray

from typing import Any, Self, Iterable


Scalar = int | float | complex
ScalarArray = NDArray[int] | NDArray[float] | NDArray[complex]
ArrayKey = int | slice | tuple[int | slice]


class _AbstractMatrix:
    def __init__(self, *args: Scalar | ScalarArray) -> None:
        """
        Basic abstract class to construct symmetric matrices classes.

        Parameters
        ----------
        *args : Scalar | ScalarArray
            Coefitients.

        """
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
    def __array_proprtey_check(self, p_name: str) -> None:
        if self._is_array:
            return
        else:
            raise AttributeError("This Matrix is a scalar, %s holds for an array" % p_name)
    
    def _apply_func(self, func: callable) -> Self:
        # Applies a funciton over the matrix by eigen-decomposition
        return _constructor_dict[type(self)](*[func(c) for c in self.coefs])
    
    def __set_coefs(self, *args: Scalar | ScalarArray) -> None:
        self._coefs = args
    
    ###############################################
    #Properties
    @property
    def coefs(self) -> Scalar | ScalarArray: return self._coefs
    @property
    def shape(self) -> tuple[int]:
        self.__array_proprtey_check("shape")
        return self.coefs[0].shape
    @property
    def size(self) -> int:
        self.__array_proprtey_check("size")
        return self.coefs[0].size
    @property
    def ndim(self) -> int:
        self.__array_proprtey_check("ndim")
        return self.coefs[0].ndim
    @property
    def real(self) -> Self: return _constructor_dict[type(self)](*[c.real for c in self.coefs])
    @property
    def imag(self) -> Self: return _constructor_dict[type(self)](*[c.imag for c in self.coefs])
    @property
    def eigvals(self) -> tuple[Scalar | ScalarArray]: return (self.coefs[0],)
    @property
    def trace(self) -> Scalar | ScalarArray: return 0.0
    
    #################################################
    #Functions
    def inv(self) -> Self:
        """
        Iverse of the matrix.

        Returns
        -------
        Self

        """
        return _constructor_dict[type(self)](*[1/c for c in self.coefs])
        
    def exp(self) -> Self:
        """
        Exponentia of the matrixl.

        Returns
        -------
        Self

        """
        return self._apply_func(np.exp)
    
    def cos(self) -> Self:
        """
        Cosine of the matrix.

        Returns
        -------
        Self

        """
        return self._apply_func(np.cos)
    
    def sin(self) -> Self:
        """
        Sine of the matrix.

        Returns
        -------
        Self

        """
        return self._apply_func(np.sin)
    
    def tan(self) -> Self:
        """
        Tangent of the matrix.

        Returns
        -------
        Self

        """
        return self._apply_func(np.tan)
    
    def log(self) -> Self:
        """
        Natural logarithm of the matrix.

        Returns
        -------
        Self

        """
        return self._apply_func(np.log)
    
    def sqrt(self) -> Self:
        """
        Squared root of the matrix.

        Returns
        -------
        Self

        """
        return self._apply_func(np.sqrt)
    
    def cbrt(self) -> Self:
        """
        Cubic root of the matrix.

        Returns
        -------
        Self

        """
        return self._apply_func(np.cbrt)
    
    ##########################################################
    #Array methods
    def reshape(self, newsh: Iterable[int], **kwargs: Any) -> Self:
        """
        Reshapes matrix coeficients only if they are arrays.

        Parameters
        ----------
        newsh : Iterable[int]
            New shape.
        **kwargs : Any
            :func:np.reshape keargs.

        Returns
        -------
        Self

        """
        self.__array_proprtey_check("reshape")
        return _constructor_dict[type(self)](*[c.reshape(newsh, **kwargs) for c in self.coefs])
    
    def __getitem__(self, key: ArrayKey) -> Self:
        if self._is_array:
            return _constructor_dict[type(self)](*[c[key] for c in self.coefs])
        else:
            raise IndexError("Non-array OhMatrix does not accept indexing")
    
    def __setitem__(self, key: ArrayKey, item: Self) -> Self:
        if not self._is_array:
            raise IndexError("Non-array OhMatrix does not accept indexing")
        if type(item) != type(self):
            raise TypeError("Cannot set this objetc elements on this matrix")
        for i in range(len(self.coefs)):
            self._coefs[i][key] = item.coefs[i]
    
    #######################################################
    #Magic methods
    def __pos__(self) -> Self:
        return _constructor_dict[type(self)](*self.coefs)
    
    def __neg__(self) -> Self:
        return _constructor_dict[type(self)](*[-c for c in self.coefs])
    
    def __add__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self
    
    def __radd__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self.__add__(other)
    
    def __iadd__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self.__add__(other)
    
    def __sub__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self
    
    def __rsub__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self
    
    def __isub__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self.__sub__(other)
    
    def __mul__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self
    
    def __rmul__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self.__mul__(other)
    
    def __imul__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self.__mul__(other)
    
    def __truediv__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self
    
    def __rtruediv__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self
    
    def __itruediv__(self, other: Self | Scalar | ScalarArray) -> Self:
        return self.__truediv__(other)
    
    def __pow__(self, other: Scalar | ScalarArray) -> Self:
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
    
    def __array_ufunc__(self, ufunc: callable, method: str, *inputs: Any, **kwargs: Any) -> Self:
        return self
    
    def __repr__(self) -> str:
        return "_Abstract"



class OhMatrix(_AbstractMatrix):
    def __init__(self, a: Scalar | ScalarArray, b: Scalar | ScalarArray) -> None:
        """
        Two coeffitient representation for matrices representing orbital-spin systems conserved under Oh symmetries.

        Parameters
        ----------
        a : Scalar | ScalarArray
            Identity coefficient.
        b : Scalar | ScalarArray
            SOC coefficient.

        """
        
        super().__init__(a, b)
    
    ###############################################
    #Auxiliar
    
    def _apply_func(self, func: callable) -> Self:
        c = (func(self.a+2*self.b) + 2*func(self.a-self.b))/3
        d = (func(self.a+2*self.b) - func(self.a-self.b))/3
        return ohmatrix(c, d)
    
    ###############################################
    #Properties
    @property
    def a(self) -> Scalar | ScalarArray: return self._coefs[0]
    @property
    def b(self) -> Scalar | ScalarArray: return self._coefs[1]
    @property
    def coefs(self) -> list[Scalar | ScalarArray]: return [self.a, self.b]
    @property
    def eigvals(self) -> tuple[Scalar | ScalarArray]: return self.a+2*self.b, self.a-self.b
    @property
    def trace(self) -> Scalar | ScalarArray: return 6*self.a
    
    #################################################
    #Functions
    def inv(self) -> Self:
        """
        Iverse of the matrix.

        Returns
        -------
        Self

        """
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
    
    def __array_ufunc__(self, ufunc: callable, method: str, *inputs: Any, **kwargs: Any)-> Self:
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