from Benchmark.Functions.__Function import Function
import numpy as np


class RW17(Function):
    def __init__(self, dim):
        x_start = np.zeros(3)
        f_x_start = 0
        domain = [
            (0.05, 2.0),  # x1 : wire diameter
            (0.25, 1.3),  # x2 : mean coil diameter
            (2.0, 15.0),  # x3 : number of active coils
        ]
        name = "RW17"
        max_dimension = 3

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
        x1 = x[0]  # d   : wire diameter
        x2 = x[1]  # D   : mean coil diameter
        x3 = x[2]  # N   : number of active coils

        # ===== OBJECTIVE FUNCTION =====
        f = x1 ** 2 * x2 * (x3 + 2)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(4)

        # g1 : shear stress constraint
        g[0] = 1 - (x2 ** 3 * x3) / (71785 * x1 ** 4)

        # g2 : surge frequency constraint
        g[1] = ((4 * x2 ** 2 - x1 * x2) / (12566 * (x2 * x1 ** 3 - x1 ** 4))
                + 1 / (5108 * x1 ** 2) - 1)

        # g3 : deflection constraint
        g[2] = 1 - (140.45 * x1) / (x2 ** 2 * x3)

        # g4 : diameter constraint
        g[3] = (x1 + x2) / 1.5 - 1

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
