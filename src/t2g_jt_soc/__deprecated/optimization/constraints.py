import numpy as np

from .problems import Problem



class Constraint(Problem):
    def __init__(self):
        super().__init__()



class LinearConstraint(Constraint):
    def __init__(self, a, d):
        # Ax = d
        self.__a = a
        self.__d = d
    
    
    # Properties
    @property
    def a(self): return self.__a
    @property
    def d(self): return self.__d
    @property
    def size(self): return self.a.shape[0]
    
    
    # Objective function
    def f(self, x):
        super().f(x)
        return np.dot(self.a, x) - self.d
    
    # Gradient
    def grad(self, x):
        super().grad(x)
        return self.a