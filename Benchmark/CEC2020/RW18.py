from Benchmark.__Function import Function
import numpy as np


class RW18(Function):
    def __init__(self, dim):
        x_start = np.zeros(4)
        f_x_start = 0
        domain = [
            (0.0625, 6.1875),   # x1 : thickness of shell (discretized by 0.0625)
            (0.0625, 6.1875),   # x2 : thickness of head (discretized by 0.0625)
            (10.0, 200.0),      # x3 : inner radius
            (10.0, 200.0),      # x4 : length of cylindrical section
        ]
        name = "RW18"
        max_dimension = 4

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

        # ===== DISCRETIZATION =====
        # x1 et x2 sont discrétisées : multiples de 0.0625
        x1 = 0.0625 * np.round(x[0] / 0.0625)
        x2 = 0.0625 * np.round(x[1] / 0.0625)
        x3 = x[2]  # inner radius (continu)
        x4 = x[3]  # length of cylindrical section (continu)

        # ===== OBJECTIVE FUNCTION =====
        f = (0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2
             + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(4)
        g[0] = -x1 + 0.0193 * x3
        g[1] = -x2 + 0.00954 * x3
        g[2] = -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000
        g[3] = x4 - 240

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
