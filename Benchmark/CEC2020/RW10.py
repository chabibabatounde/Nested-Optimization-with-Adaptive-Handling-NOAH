from Benchmark.__Function import Function
import numpy as np


class RW10(Function):
    def __init__(self, dim):
        x_start = np.zeros(3)
        f_x_start = 0
        domain = [
            (0, 1),    # x1 (continu)
            (-1, 1),   # x2 (continu)
            (0, 1),    # x3 (entier : 0 ou 1)
        ]
        name = "RW10"
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
        f = -0.7 * x3 + 5 * (x1 - 0.5)**2 + 0.8

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(3)
        g[0] = -np.exp(x1 - 0.2) - x2
        g[1] = x2 + 1.1 * x3 + 1
        g[2] = x1 - x3 - 0.2

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
