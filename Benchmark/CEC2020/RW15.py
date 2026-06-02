from Benchmark.__Function import Function
import numpy as np


class RW15(Function):
    def __init__(self, dim):
        x_start = np.zeros(7)
        f_x_start = 0
        domain = [
            (2.6, 3.6),    # x1 : b (face width)
            (0.7, 0.8),    # x2 : m (module)
            (17, 28),      # x3 : z (number of teeth) - continu ici
            (7.3, 8.3),    # x4 : l1 (length of shaft 1)
            (7.3, 8.3),    # x5 : l2 (length of shaft 2)
            (2.9, 3.9),    # x6 : d1 (diameter shaft 1)
            (5.0, 5.5),    # x7 : d2 (diameter shaft 2)
        ]
        name = "RW15"
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

        # ===== VARIABLES =====
        x1 = x[0]  # b  : face width
        x2 = x[1]  # m  : module
        x3 = x[2]  # z  : number of teeth
        x4 = x[3]  # l1 : length of shaft 1
        x5 = x[4]  # l2 : length of shaft 2
        x6 = x[5]  # d1 : diameter of shaft 1
        x7 = x[6]  # d2 : diameter of shaft 2

        # ===== OBJECTIVE FUNCTION =====
        f = (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
             - 1.508 * x1 * (x6**2 + x7**2)
             + 7.477 * (x6**3 + x7**3)
             + 0.7854 * (x4 * x6**2 + x5 * x7**2))

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(11)
        g[0]  = -x1 * x2**2 * x3 + 27
        g[1]  = -x1 * x2**2 * x3**2 + 397.5
        g[2]  = -x2 * x6**4 * x3 * x4**(-3) + 1.93
        g[3]  = -x2 * x7**4 * x3 / x5**3 + 1.93
        g[4]  = 10 * x6**(-3) * np.sqrt(16.91e6 + (745 * x4 / (x2 * x3))**2) - 1100
        g[5]  = 10 * x7**(-3) * np.sqrt(157.5e6 + (745 * x5 / (x2 * x3))**2) - 850
        g[6]  = x2 * x3 - 40
        g[7]  = -x1 / x2 + 5
        g[8]  = x1 / x2 - 12
        g[9]  = 1.5 * x6 - x4 + 1.9
        g[10] = 1.1 * x7 - x5 + 1.9

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
