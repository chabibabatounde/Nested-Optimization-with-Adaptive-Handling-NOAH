from Benchmark.Functions.__Function import Function
import numpy as np


class RW2(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        # Bornes pour les 11 variables du problème RC02
        # x1, x2, x3 : flux de chaleur (E4 à E7 typiquement)
        # x4, x5, x6 : surfaces / coefficients
        # x7, x8 : températures intermédiaires
        # x9, x10, x11 : températures de sortie
        domain = [
            (1e4, 0.819 * 1e6),   # x1
            (1e4, 1.131 * 1e6),   # x2
            (1e4, 2.05 * 1e6),    # x3
            (0, 0.05074),         # x4
            (0, 0.05074),         # x5
            (0, 0.05074),         # x6
            (100, 200),           # x7
            (100, 300),           # x8
            (100, 300),           # x9
            (100, 400),           # x10
            (100, 600)            # x11
        ]
        name = "RW2"
        max_dimension = 11  # IMPORTANT: ce problème utilise 11 variables

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
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x[0]

        # ===== OBJECTIVE FUNCTION =====
        f = (x1 / (120 * x4)) ** 0.6 \
            + (x2 / (80 * x5)) ** 0.6 \
            + (x3 / (40 * x6)) ** 0.6

        # ===== INEQUALITY CONSTRAINTS =====
        g = np.array([0.0])  # aucune contrainte d'inégalité

        # ===== EQUALITY CONSTRAINTS =====
        h = np.zeros(9)

        h[0] = x1 - 1e4 * (x7 - 100)
        h[1] = x2 - 1e4 * (x8 - x7)
        h[2] = x3 - 1e4 * (500 - x8)
        h[3] = x1 - 1e4 * (300 - x9)
        h[4] = x2 - 1e4 * (400 - x10)
        h[5] = x3 - 1e4 * (600 - x11)
        h[6] = x4 * np.log(abs(x9 - 100) + 1e-8) \
               - x4 * np.log(300 - x7 + 1e-8) \
               - x9 - x7 + 400
        h[7] = x5 * np.log(abs(x10 - x7) + 1e-8) \
               - x5 * np.log(abs(400 - x8) + 1e-8) \
               - x10 + x7 - x8 + 400
        h[8] = x6 * np.log(abs(x11 - x8) + 1e-8) \
               - x6 * np.log(100) \
               - x11 + x8 + 100

        tol = 1e-4
        penalty = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        fitness = f + 1e5 * penalty  # plus souple
        # fitness = f + 1e8 * penalty  # plus strict

        return fitness
