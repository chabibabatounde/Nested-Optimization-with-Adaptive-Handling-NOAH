from Benchmark.__Function import Function
import numpy as np


class RW1(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        domain = [
                (0, 10),           # x1
                (0, 200),          # x2
                (0, 100),          # x3
                (0, 200),          # x4
                (1000, 2000000),   # x5
                (0, 600),          # x6
                (100, 600),        # x7
                (100, 600),        # x8
                (100, 900)         # x9
            ]
        name = "RW1"
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
        f = 35 * x1 ** 0.6 + 35 * x2 ** 0.6

        # ===== INEQUALITY CONSTRAINTS =====
        g = np.array([0.0])  # aucune contrainte d'inégalité

        # ===== EQUALITY CONSTRAINTS =====
        h = np.zeros(8)

        h[0] = 200 * x1 * x4 - x3
        h[1] = 200 * x2 * x6 - x5
        h[2] = x3 - 10000 * (x7 - 100)
        h[3] = x5 - 10000 * (300 - x7)
        h[4] = x3 - 10000 * (600 - x8)
        h[5] = x5 - 10000 * (900 - x9)
        h[6] = x4 * np.log(abs(x8 - 100) + 1e-8) - x4 * np.log((600 - x7) + 1e-8) - x8 + x7 + 500
        h[7] = x6 * np.log(abs(x9 - x7) + 1e-8) - x6 * np.log(600) - x9 + x7 + 600

        tol = 1e-4
        penalty = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        fitness = f + 1e5 * penalty  # plus souple
        # fitness = f + 1e8 * penalty  # plus strict

        return fitness
