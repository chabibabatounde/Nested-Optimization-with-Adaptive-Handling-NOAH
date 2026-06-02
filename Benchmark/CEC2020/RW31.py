from Benchmark.Functions.__Function import Function
import numpy as np


class RW31(Function):
    def __init__(self, dim):
        x_start = np.array([12.0, 13.0, 51.0, 49.0])
        f_x_start = 0
        domain = [
            (12, 60),  # x1  : number of teeth on gear 1
            (12, 60),  # x2  : number of teeth on gear 2
            (12, 60),  # x3  : number of teeth on gear 3
            (12, 60),  # x4  : number of teeth on gear 4
        ]
        name = "RW31"
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
        # Gear train: 4 gears with teeth counts
        x1 = x[0]  # Number of teeth on gear 1 (pinion 1)
        x2 = x[1]  # Number of teeth on gear 2 (gear 1)
        x3 = x[2]  # Number of teeth on gear 3 (pinion 2)
        x4 = x[3]  # Number of teeth on gear 4 (gear 2)

        # ===== OBJECTIVE FUNCTION =====
        # Minimize gear ratio error: minimize deviation from target ratio
        # Target transmission ratio: 1/6.931 ≈ 0.1443
        # Actual ratio: (x1 × x2) / (x3 × x4)
        # Error = |target - actual|²

        target_ratio = 1.0 / 6.931
        actual_ratio = (x1 * x2) / (x3 * x4)

        f = (target_ratio - actual_ratio) ** 2

        # ===== INEQUALITY CONSTRAINTS =====
        # No inequality constraints
        g = np.array([0.0])

        # ===== EQUALITY CONSTRAINTS =====
        # No equality constraints
        h = np.array([0.0])

        # ===== PENALTY METHOD =====
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        penalty_h = np.sum(h ** 2)

        # Check for NaN or Inf
        if np.isnan(f) or np.isinf(f):
            return 1e10

        fitness = f + 1e5 * penalty_g + 1e5 * penalty_h

        return fitness
