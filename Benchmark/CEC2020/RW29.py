from Benchmark.Functions.__Function import Function
import numpy as np


class RW29(Function):
    def __init__(self, dim):
        x_start = np.array([50.0, 40.0, 1.0, 0.5])
        f_x_start = 0
        domain = [
            (10.0, 100.0),     # x1  : inlet gas temperature (°C) or design parameter 1
            (5.0, 100.0),      # x2  : pressure ratio or design parameter 2
            (0.5, 5.0),        # x3  : polytropic efficiency or design parameter 3
            (0.1, 1.0),        # x4  : design parameter 4 (ratio/factor)
        ]
        name = "RW29"
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

        # ===== EXTRACT VARIABLES =====
        x1 = x[0]      # Design parameter 1
        x2 = x[1]      # Design parameter 2 (pressure ratio)
        x3 = x[2]      # Design parameter 3 (efficiency)
        x4 = x[3]      # Design parameter 4 (ratio factor)

        # ===== OBJECTIVE FUNCTION =====
        # Minimize total cost of gas transmission compressor system
        # Cost components:
        # 1. Capital cost (equipment + installation)
        # 2. Operating cost (power + maintenance)
        # 3. Material cost

        term1 = 8.61e5 * (x1 ** 0.5) * x2 * (x3 ** (-2.0/3.0)) * (x4 ** (-0.5))
        term2 = 3.69e4 * x3
        term3 = 7.72e8 * (x1 ** (-1.0)) * (x2 ** 0.219)
        term4 = 765.43e6 * (x1 ** (-1.0))

        f = term1 + term2 + term3 - term4

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(1)

        # Constraint 1: Relationship between design parameters
        # x4 × x2^(-2) + x2^(-2) <= 1
        # Equivalent to: (x4 + 1) / x2^2 <= 1
        # Or: x4 + 1 <= x2^2

        constraint_term = x4 * (x2 ** (-2.0)) + (x2 ** (-2.0))
        g[0] = constraint_term - 1.0

        # ===== EQUALITY CONSTRAINTS =====
        h = np.array([0.0])

        # ===== PENALTY METHOD =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
