import numpy as np


class Problem:
    def __init__(self):
        pass
    
    
    # Properties
    @property
    def size(self): return
    @property # Sets as default that gradient is linear for inverse matrix method
    def is_linear_gradient(self): return True
    
    # Auxiliar
    def __check_size(self, x):
        if self.size is None:
            raise AttributeError("This problem has no size")
        else:
            if x.shape[0] != self.size:
                raise ValueError("Objective vector must match the problem's size")
    
    
    # Objective function
    def f(self, x):
        self.__check_size(x)
        pass
    
    # Gradient
    def grad(self, x):
        self.__check_size(x)
        pass
    
    
    # Gradient dependent matrix
    def grad_dep_mat(self):
        if not self.is_linear_gradient:
            raise AttributeError("This problem has not a linear gradient")
        return np.zeros((self.size,self.size))
    
    # Gradient independent vector
    def grad_ind_vec(self):
        if not self.is_linear_gradient:
            raise AttributeError("This problem has not a linear gradient")
        return np.zeros(self.size)


class LinearProblem(Problem):
    def __init__(self, c):
        # Lag = c^T x
        self.__c = c # 1-dim array
    
    
    # Properties
    @property
    def c(self): return self.__c
    @property
    def size(self): return self.c.shape[0]
    
    
    # Objective function
    def f(self, x):
        super().f(x)
        return np.dot(self.c, x)
    
    # Gradient
    def grad(self, x):
        super().grad(x)
        return self.c
    
    
    # Gradient independent vector
    def grad_ind_vec(self):
        return self.c


class QuadraticProblem(Problem):
    def __init__(self, Q, c):
        # Lag = 0.5 * x^T Q x + c^T x
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be an squared matrix")
        if c.shape[0] != Q.shape[0]:
            raise ValueError("Q and c must have the same dimensions being Q a matrix and c a vector")
        self.__Q = Q # 2-dim matrix
        self.__c = c # 1-dim array
    
    
    # Properties
    @property
    def Q(self): return self.__Q
    @property
    def c(self): return self.__c
    @property
    def size(self): return self.Q.shape[0]
    
    
    # Objective function
    def f(self, x):
        super().f(x)
        return 0.5*np.dot(x, self.Q @ x) + np.dot(self.c, x)
    
    # Gradient
    def grad(self, x):
        super().grad(x)
        return 0.5*np.dot(self.Q+self.Q.T, x) + self.c
    
    
    # Gradient dependent matrix
    def grad_dep_mat(self):
        return 0.5 * (self.Q+self.Q.T)
    
    # Gradient independent vector
    def grad_ind_vec(self):
        return self.c


class LeastSquaresProblem(QuadraticProblem):
    def __init__(self, A, b):
        # Lag = 0.5*|Ax-b|^2
        if b.shape[0] != A.shape[1]:
            raise ValueError("A must be a matrix with the same coulumns that dimensions of vector b")
        self.__A = A # 2-dim matrix
        self.__b = b # 1-dim array
        super().__init__(A.T @ A, -A.T @ b)
    


if __name__ == "__main__":
    # Q = np.random.random((5,5))
    # c = np.random.random(5)
    # x = np.random.random(5)
    # prob = QuadraticProblem(Q, c)
    # L = prob.f(x)
    
    A = np.random.random((5,5))
    b = np.random.random(5)
    x = np.random.random(5)
    prob = LeastSquaresProblem(A, b)
    # L = prob.f(x)