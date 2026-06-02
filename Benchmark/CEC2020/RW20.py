from Benchmark.Functions.__Function import Function
import numpy as np


class RW20(Function):
    def __init__(self, dim):
        x_start = np.zeros(2)
        f_x_start = 0
        domain = [
            (0.0, 10.0),  # x1 : cross-sectional area of diagonal bars
            (0.0, 10.0),  # x2 : cross-sectional area of horizontal bar
        ]
        name = "RW20"
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

        # ===== VARIABLES =====
        x1 = x[0]  # cross-sectional area of diagonal bars
        x2 = x[1]  # cross-sectional area of horizontal bar

        # ===== OBJECTIVE FUNCTION =====
        # Minimize total weight
        f = (2 * np.sqrt(2) * x1 + x2) * 100

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(3)

        # Stress constraint for diagonal bars
        g[0] = ((np.sqrt(2) * x1 + x2) / (np.sqrt(2) * x1 ** 2 + 2 * x1 * x2) * 2) - 2

        # Stress constraint for horizontal bar
        g[1] = (x2 / (np.sqrt(2) * x1 ** 2 + 2 * x1 * x2) * 2) - 2

        # Buckling constraint
        g[2] = (1 / (np.sqrt(2) * x2 + x1) * 2) - 2

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
