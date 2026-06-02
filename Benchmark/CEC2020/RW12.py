from Benchmark.Functions.__Function import Function
import numpy as np


class RW12(Function):
    def __init__(self, dim):
        x_start = np.zeros(7)
        f_x_start = 0
        domain = [
            (0, 3),   # x1 (continu)
            (0, 3),   # x2 (continu)
            (0, 3),   # x3 (continu)
            (0, 1),   # y1 (entier : 0 ou 1)
            (0, 1),   # y2 (entier : 0 ou 1)
            (0, 1),   # y3 (entier : 0 ou 1)
            (0, 3),   # y4 (entier : 0, 1, 2 ou 3)
        ]
        name = "RW12"
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
        x = np.array(variables_values).reshape(1, -1)[0]

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        y1 = round(x[3])   # variable entière (0 ou 1)
        y2 = round(x[4])   # variable entière (0 ou 1)
        y3 = round(x[5])   # variable entière (0 ou 1)
        y4 = round(x[6])   # variable entière (0, 1, 2 ou 3)

        # ===== OBJECTIVE FUNCTION =====
        f = ((y1 - 1)**2 + (y2 - 1)**2 + (y3 - 1)**2
             - np.log(y4 + 1)
             + (x1 - 1)**2 + (x2 - 2)**2 + (x3 - 3)**2)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(9)
        g[0] = x1 + x2 + x3 + y1 + y2 + y3 - 5
        g[1] = y3**2 + x1**2 + x2**2 + x3**2 - 5.5
        g[2] = x1 + y1 - 1.2
        g[3] = x2 + y2 - 1.8
        g[4] = x3 + y3 - 2.5
        g[5] = x1 + y4 - 1.2
        g[6] = y2**2 + x2**2 - 1.64
        g[7] = y3**2 + x3**2 - 4.25
        g[8] = y2**2 + x3**2 - 4.64

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
