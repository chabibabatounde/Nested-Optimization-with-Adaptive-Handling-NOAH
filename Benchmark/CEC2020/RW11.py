from Benchmark.Functions.__Function import Function
import numpy as np


class RW11(Function):
    def __init__(self, dim):
        x_start = np.zeros(7)
        f_x_start = 0
        domain = [
            (0, 20),   # x1 (continu)
            (0, 20),   # x2 (continu)
            (0, 10),   # v1 (continu)
            (0, 10),   # v2 (continu)
            (0, 1),    # y1 (entier : 0 ou 1)
            (0, 1),    # y2 (entier : 0 ou 1)
            (0, 20),   # x_ (continu)
        ]
        name = "RW11"
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
        v1 = x[2]
        v2 = x[3]
        y1 = round(x[4])   # variable entière (0 ou 1)
        y2 = round(x[5])   # variable entière (0 ou 1)
        x_ = x[6]

        # ===== TERMES INTERMÉDIAIRES =====
        z1 = 0.9 * (1 - np.exp(-0.5 * v1)) * x1
        z2 = 0.8 * (1 - np.exp(-0.4 * v2)) * x2

        # ===== OBJECTIVE FUNCTION =====
        f = 7.5 * y1 + 5.5 * y2 + 7 * v1 + 6 * v2 + 5 * x_

        # ===== EQUALITY CONSTRAINTS (h = 0) =====
        h = np.zeros(4)
        h[0] = y1 + y2 - 1
        h[1] = z1 + z2 - 10
        h[2] = x1 + x2 - x_
        h[3] = z1 * y1 + z2 * y2 - 10

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(4)
        g[0] = v1 - 10 * y1
        g[1] = v2 - 10 * y2
        g[2] = x1 - 20 * y1
        g[3] = x2 - 20 * y2

        # ===== PENALTY =====
        tol = 1e-4
        penalty_h = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * (penalty_h + penalty_g)

        return fitness
