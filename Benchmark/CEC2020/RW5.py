from Benchmark.Functions.__Function import Function
import numpy as np


class RW5(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        # Bornes pour les 9 variables du problème RC05 (Haverly's Pooling Problem)
        domain = [
            (0, 100),    # x1
            (0, 200),    # x2
            (0, 100),    # x3
            (0, 100),    # x4
            (0, 100),    # x5
            (0, 100),    # x6
            (0, 200),    # x7
            (0, 100),    # x8
            (0, 200)     # x9
        ]
        name = "RW5"
        max_dimension = 9  # IMPORTANT: ce problème utilise 9 variables

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
        x = np.array(variables_values).reshape(1, -1)

        # Variables (plus lisible)
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = x[0]

        # ===== OBJECTIVE FUNCTION =====
        # Minimisation : -profit
        f = -(9 * x1 + 15 * x2 - 6 * x3 - 16 * x4 - 10 * (x5 + x6))

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(2)
        g[0] = x9 * x7 + 2 * x5 - 2.5 * x1
        g[1] = x9 * x8 + 2 * x6 - 1.5 * x2

        # ===== EQUALITY CONSTRAINTS (h = 0) =====
        h = np.zeros(4)
        h[0] = x7 + x8 - x3 - x4
        h[1] = x1 - x7 - x5
        h[2] = x2 - x8 - x6
        h[3] = x9 * x7 + x9 * x8 - 3 * x3 - x4

        tol = 1e-4
        penalty_h = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * (penalty_h + penalty_g)  # plus souple
        # fitness = f + 1e8 * (penalty_h + penalty_g)  # plus strict

        return fitness
