from Benchmark.Functions.__Function import Function
import numpy as np


class RW8(Function):
    def __init__(self, dim):
        x_start = np.zeros(2)
        f_x_start = 0
        domain = [
            (0, 1.6),   # x1 (continu)
            (0, 1),     # x2 (entier : 0 ou 1)
        ]
        name = "RW8"
        max_dimension = 2

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
        x2 = round(x[1])   # variable entière (0 ou 1)

        # ===== OBJECTIVE FUNCTION =====
        f = 2 * x1 + x2

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(2)
        g[0] = 1.25 - x1**2 - x2
        g[1] = x1 + x2 - 1.6

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
