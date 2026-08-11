from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, TypeVar, Generic
from ..__utils.__matrices import I,V


if TYPE_CHECKING:
    from typing import Any, Self, Iterable
    from numpy.typing import NDArray
    from ..__utils.__new_types import Scalar, ArrayKey

GeTy = TypeVar("GeTy") # Allows classes have typing parameters


class _AbstractMatrix(Generic[GeTy]):
    def __init__(self, *args: Scalar | NDArray[Scalar]) -> None:
        """
        Basic abstract class to construct symmetric matrices classes.

        Parameters
        ----------
        *args : Scalar | NDArray[Scalar]
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
            
            if np.all(isinstance(c.flatten()[0].item(), (int,float,complex)) for c in _coefs):
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
    
    def __set_coefs(self, *args: Scalar | NDArray[Scalar]) -> None:
        self._coefs = args
    
    ###############################################
    #Properties
    @property
    def coefs(self) -> Scalar | NDArray[Scalar]: return self._coefs
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
    def eigvals(self) -> tuple[Scalar | NDArray[Scalar]]: return (self.coefs[0],)
    @property
    def trace(self) -> Scalar | NDArray[Scalar]: return 0.0
    
    #################################################
    # Numpy matrix display
    def numpy_matrix(self) -> NDArray[Scalar]:
        """
        Numpy array representation of matrix.

        Returns
        -------
        NDArray[Scalar]

        """
        return np.eye(len(self.coefs)) * np.array(self.coefs[...,None,None])
    
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
    
    def __add__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self
    
    def __radd__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self.__add__(other)
    
    def __iadd__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self.__add__(other)
    
    def __sub__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self
    
    def __rsub__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self
    
    def __isub__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self.__sub__(other)
    
    def __mul__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self
    
    def __rmul__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self.__mul__(other)
    
    def __imul__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self.__mul__(other)
    
    def __truediv__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self
    
    def __rtruediv__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self
    
    def __itruediv__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        return self.__truediv__(other)
    
    def __pow__(self, other: Scalar | NDArray[Scalar]) -> Self:
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
    
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Self:
        return self
    
    def __repr__(self) -> str:
        return "_Abstract"



class OhMatrix(_AbstractMatrix[GeTy]):
    def __init__(self, a: Scalar | NDArray[Scalar], b: Scalar | NDArray[Scalar]) -> None:
        """
        Two coeffitient representation for matrices representing orbital-spin systems conserved under Oh symmetries.

        Parameters
        ----------
        a : Scalar | NDArray[Scalar]
            Identity coefficient.
        b : Scalar | NDArray[Scalar]
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
    def a(self) -> Scalar | NDArray[Scalar]: return self._coefs[0]
    @property
    def b(self) -> Scalar | NDArray[Scalar]: return self._coefs[1]
    @property
    def coefs(self) -> list[Scalar | NDArray[Scalar]]: return [self.a, self.b]
    @property
    def eigvals(self) -> tuple[Scalar | NDArray[Scalar]]: return self.a+2*self.b, self.a-self.b
    @property
    def trace(self) -> Scalar | NDArray[Scalar]: return 6*self.a
    
    #################################################
    # Numpy matrix display
    def numpy_matrix(self) -> NDArray[Scalar]:
        """
        Numpy array representation of matrix.

        Returns
        -------
        NDArray[Scalar]

        """
        if self._is_array:
            return self.a[...,None,None]*I + self.b[...,None,None]*V
        else:
            return self.a*I + self.b*V
    
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
    def __add__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a+other.a, self.b+other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a+other, self.b)
        else:
            raise TypeError
    
    def __sub__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a-other.a, self.b-other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a-other, self.b)
        else:
            raise TypeError
    
    def __rsub__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        if isinstance(other, OhMatrix):
            return ohmatrix(-self.a+other.a, -self.b+other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(-self.a+other, -self.b)
        else:
            raise TypeError
    
    def __mul__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        if isinstance(other, OhMatrix):
            return ohmatrix(self.a*other.a + 2*self.b*other.b, self.b*other.a + (self.a+self.b)*other.b)
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a*other, self.b*other)
        else:
            raise TypeError
    
    def __truediv__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        if isinstance(other, OhMatrix):
            return self * other**-1
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return ohmatrix(self.a/other, self.b/other)
        else:
            raise TypeError
    
    def __rtruediv__(self, other: Self | Scalar | NDArray[Scalar]) -> Self:
        if isinstance(other, OhMatrix):
            return self.inv() * other
        elif isinstance(other, (int,float,complex,np.ndarray)):
            return self.inv() * other
        else:
            raise TypeError
    
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any)-> Self:
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
    
    def __repr__(self) -> str:
        return f"OhMatrix({self.a}, {self.b})"


# Constructors

def ohmatrix(a: Scalar | NDArray[Scalar], b: Scalar | NDArray[Scalar]) -> OhMatrix:
    """
    Two coeffitient representation for matrices representing orbital-spin systems conserved under Oh symmetries. 
    Basic constructor function

    Parameters
    ----------
    a : Scalar | NDArray[Scalar]
        Identity coefficient.
    b : Scalar | NDArray[Scalar]
        SOC coefficient.
    
    Returns
    -------
    OhMatrix

    """
    return OhMatrix(a,b)


def ohidentity(shape: None | Iterable[int] = None) -> OhMatrix:
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


def ohzeros(shape: None | Iterable[int] = None) -> OhMatrix:
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
        return OhMatrix(0,0)
    else:
        return OhMatrix(np.zeros(shape), np.zeros(shape))


def ohrandom(shape: None | Iterable[int] = None) -> OhMatrix:
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