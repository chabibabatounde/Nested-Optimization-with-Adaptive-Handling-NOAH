from Benchmark.Functions.__Function import Function
import numpy as np


class RW3(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        domain = [
            (0, 100),   # x1
            (0, 200),   # x2
            (0, 100),   # x3
            (0, 100),   # x4
            (0, 100),   # x5
            (0, 100),   # x6
            (0, 200)    # x7
        ]
        name = "RW3"
        max_dimension = 7

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
        x1, x2, x3, x4, x5, x6, x7 = x[0]

        # ===== OBJECTIVE FUNCTION =====
        # Minimisation : -profit
        f = 9 * x1 + 15 * x2 - 6 * x3 - 16 * x4 - 10 * x5

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(2)
        g[0] = x6 * x7 + 2 * x5 - 2.5 * x1
        g[1] = x6 * x7 + 2 * x5 - 1.5 * x2

        # ===== EQUALITY CONSTRAINTS (h = 0) =====
        h = np.zeros(4)
        h[0] = x7 * x6 + 2 * x5 - 2.5 * x1   # qualité produit 1
        h[1] = x6 * x7 + 2 * x5 - 1.5 * x2   # qualité produit 2
        h[2] = x3 + x4 - x6 - x7              # bilan matière pool
        h[3] = x1 - x3                        # bilan x1 (selon formulation)

        # Reformulation officielle Haverly Pooling Problem (CEC2020):
        h[0] = x1 + x2 - x3 - x4              # bilan total entrée/sortie
        h[1] = x1 - x6 - x3                   # bilan produit 1
        h[2] = x2 - x7 - x4                   # bilan produit 2
        h[3] = x6 * x7 + x4 * x5 - 3 * x3 - x4 * x5  # qualité au pool

        tol = 1e-4
        penalty_h = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * (penalty_h + penalty_g)

        return fitness
