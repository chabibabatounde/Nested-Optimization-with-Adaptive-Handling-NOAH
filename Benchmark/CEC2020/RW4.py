from Benchmark.Functions.__Function import Function
import numpy as np


class RW4(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        domain = [
            (0, 1),  # x1
            (0, 1),  # x2
            (0, 1),  # x3
            (0, 1),  # x4
            (0.00001, 16),  # x5
            (0.00001, 16)  # x6
        ]
        name = "RW4"
        max_dimension = 6

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
        x = np.array(variables_values)
        x1, x2, x3, x4, x5, x6 = x

        # ===== OBJECTIVE =====
        f = (x1 - 1) ** 2 + (x2 - 2) ** 2 + (x3 - 3) ** 2 \
            + (x4 - 4) ** 2 + (x5 - 5) ** 2 + (x6 - 6) ** 2

        # ===== INEQUALITY CONSTRAINTS g(x) <= 0 =====
        g = np.zeros(6)

        g[0] = x1 + x2 + x3 + x4 + x5 + x6 - 2
        g[1] = x1 + x2 + x3 + x4 + x5 + x6 - 3
        g[2] = x1 + x2 - 1
        g[3] = x3 + x4 - 1
        g[4] = x5 + x6 - 1
        g[5] = x1 * x6 - 0.5

        # ===== PENALTY =====
        penalty = np.sum(np.maximum(0, g) ** 2)

        # ===== FITNESS =====
        fitness = f + 1e6 * penalty

        return fitness
