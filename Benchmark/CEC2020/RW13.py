from Benchmark.Functions.__Function import Function
import numpy as np


class RW13(Function):
    def __init__(self, dim):
        x_start = np.zeros(5)
        f_x_start = 0
        domain = [
            (78, 102),   # x1 (continu)
            (33, 45),    # x2 (continu)
            (27, 45),    # x3 (continu)
            (27, 45),    # y1 (entier)
            (27, 45),    # y2 (entier)
        ]
        name = "RW13"
        max_dimension = 5

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
        y1 = round(x[3])   # variable entière
        y2 = round(x[4])   # variable entière

        # ===== COEFFICIENTS =====
        a = [
            85.334407,   # a[0]
            0.0056858,   # a[1]
            0.0006262,   # a[2]
            0.0022053,   # a[3]
            80.51249,    # a[4]
            0.0071317,   # a[5]
            0.0029955,   # a[6]
            0.0021813,   # a[7]
            9.300961,    # a[8]
            0.0047026,   # a[9]
            0.0012547,   # a[10]
            0.0019085    # a[11]
        ]

        # ===== OBJECTIVE FUNCTION =====
        f = (-5.357854 * x1**2
             - 0.835689 * y1 * x3
             - 37.29329 * y1
             + 40792.141)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(3)
        g[0] = a[0] + a[1]*y2*x3 + a[2]*y1*x2 - a[3]*y1*y1*x3 - 92
        g[1] = a[4] + a[5]*y2*x3 + a[6]*y1*x2 + a[7]*x1**2 - 90 - 20
        g[2] = a[8] + a[9]*y1*x2 + a[10]*y1*x1 + a[11]*x1*x2 - 20 - 5

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
