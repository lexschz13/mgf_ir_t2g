import numpy as np

from .problems import Problem



class Regularization(Problem):
    def __init__(self, size):
        super().__init__()
        self.__size = size
    
    
    # Properties
    @property
    def size(self): return self.__size



class L2Regularization(Regularization):
    def __init__(self, alpha, size):
        super().__init__(size)
        self.__alpha = alpha
    
    
    # Properties
    @property
    def alpha(self): return self.__alpha
    
    
    # Objective function
    def f(self,x):
        super().f(x)
        return 0.5 * self.alpha**2 * np.dot(x,x)
    
    # Gradient
    def grad(self, x):
        super().grad(x)
        return self.alpha**2 * x
    
    
    # Gradient dependent matrix
    def grad_dep_mat(self):
        return self.alpha**2 * np.eye(self.size)


class L1Regularization(Regularization):
    def __init__(self, alpha, size):
        super().__init__(size)
        self.__alpha = alpha
    
    
    # Properties
    @property
    def is_linear_gradient(self): return False
    
    
    # Objective function
    def f(self, x):
        super().f(x)
        return self.alpha * np.sum(np.abs(x))
    
    
    # Gradient
    def grad(self, x):
        super().grad(x)
        return self.alpha * np.sign(x)