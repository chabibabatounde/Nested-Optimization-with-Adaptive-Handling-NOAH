from Benchmark.Functions.__Function import Function
import numpy as np


class RW9(Function):
    def __init__(self, dim):
        x_start = np.zeros(3)
        f_x_start = 0
        domain = [
            (0, 3),   # x1 (continu)
            (0, 3),   # x2 (continu)
            (0, 1),   # x3 (entier : 0 ou 1)
        ]
        name = "RW9"
        max_dimension = 3

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            f_x_start,
            max_dimension,
            name
        )
        self.types(['CEC2020RW'])

    def eval(self, variables_values):
        x = np.array(variables_values).reshape(1, -1)[0]

        x1 = x[0]
        x2 = x[1]
        x3 = round(x[2])   # variable entière (0 ou 1)

        # ===== OBJECTIVE FUNCTION =====
        f = -x3 + 2 * x1 + x2

        # ===== EQUALITY CONSTRAINTS (h = 0) =====
        h = np.zeros(1)
        h[0] = x1 - 2 * np.exp(-x2)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(1)
        g[0] = -x1 + x2 + x3

        # ===== PENALTY =====
        tol = 1e-4
        penalty_h = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * (penalty_h + penalty_g)

        return fitness
