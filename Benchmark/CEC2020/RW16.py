from Benchmark.Functions.__Function import Function
import numpy as np


class RW16(Function):
    def __init__(self, dim):
        x_start = np.zeros(14)
        f_x_start = 0
        domain = [
            (0.5, 1.5),    # x1
            (0.5, 1.5),    # x2
            (0.5, 1.5),    # x3
            (0.5, 1.5),    # x4
            (0.5, 4.0),    # x5
            (0.5, 4.0),    # x6
            (0.5, 2.0),    # x7
            (0.5, 2.0),    # x8
            (0.5, 4.0),    # x9
            (0.5, 2.0),    # x10
            (0.5, 4.0),    # x11
            (0.5, 4.0),    # x12
            (0.5, 2.0),    # x13
            (0.5, 2.0),    # x14
        ]
        name = "RW16"
        max_dimension = 14

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
        x1  = x[0]
        x2  = x[1]
        x3  = x[2]
        x4  = x[3]
        x5  = x[4]
        x6  = x[5]
        x7  = x[6]
        x8  = x[7]
        x9  = x[8]
        x10 = x[9]
        x11 = x[10]
        x12 = x[11]
        x13 = x[12]
        x14 = x[13]

        # ===== OBJECTIVE FUNCTION =====
        f = (63098.88 * x2 * x4 * x12 + 5441.5 * x2**2 * x12 + 115055.5 * x2**1.664 * x6
             + 6172.27 * x2**2 * x6 + 63098.88 * x1 * x3 * x11 + 5441.5 * x1**2 * x11
             + 115055.5 * x1**1.664 * x5 + 6172.27 * x1**2 * x5 + 140.53 * x1 * x11
             + 281.29 * x3 * x11 + 70.26 * x1**2 + 281.29 * x1 * x3 + 281.29 * x3**2
             + 14437 * x8**1.8812 * x12**0.3424 * x10 * x14**(-1) * x1**2 * x7 * x9**(-1)
             + 20470.2 * x7**2.893 * x11**0.316 * x1**2)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(15)
        g[0]  = 1.524 * x7**(-1) - 1
        g[1]  = 1.524 * x8**(-1) - 1
        g[2]  = 0.07789 * x1 - 2 * x7**(-1) * x9 - 1
        g[3]  = 7.05305 * x9**(-1) * x1**2 * x10 * x8**(-1) * x2**(-1) * x14**(-1) - 1
        g[4]  = 0.0833 / x13 * x14 - 1
        g[5]  = 0.04771 * x10 * x8**1.8812 * x12**0.3424 - 1
        g[6]  = 0.0488 * x9 * x7**1.893 * x11**0.316 - 1
        g[7]  = 0.0099 * x1 / x3 - 1
        g[8]  = 0.0193 * x2 / x4 - 1
        g[9]  = 0.0298 * x1 / x5 - 1
        g[10] = (47.136 * x2**0.333 / x10 * x12 - 1.333 * x8 * x13**2.1195
                 + 62.08 * x13**2.1195 * x8**0.2 / (x12 * x10) - 1)
        g[11] = 0.056 * x2 / x6 - 1
        g[12] = 2 / x9 - 1
        g[13] = 2 / x10 - 1
        g[14] = x12 / x11 - 1

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
