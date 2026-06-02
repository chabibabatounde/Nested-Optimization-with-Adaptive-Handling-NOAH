import numpy as np
from Benchmark.Functions.__Function import Function


class CECFunction(Function):
    def __init__(self, dim, domain, name):
        x_start = np.zeros(dim)
        f_x_start = 0
        max_dimension = None
        super().__init__(dim, domain, x_start, f_x_start, max_dimension, name)
        self.o = np.random.uniform(domain[0], domain[1], dim)  # décalage aléatoire
        # matrice de rotation orthogonale
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        self.M = Q

    def shift_rotate(self, x):
        z = np.dot(self.M, (x - self.o))
        return z
