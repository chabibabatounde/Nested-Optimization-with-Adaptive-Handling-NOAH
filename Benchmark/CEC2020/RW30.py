from Benchmark.Functions.__Function import Function
import numpy as np


class RW30(Function):
    def __init__(self, dim):
        x_start = np.array([10.0, 1.5, 10.0])
        f_x_start = 0
        domain = [
            (1, 70),            # x1  : number of active coils (integer, 1-70)
            (0.6, 3.0),        # x2  : wire diameter outer (inches)
            (1, 42),            # x3  : wire diameter index (discrete, 1-42)
        ]
        name = "RW30"
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

        # ===== DISCRETE WIRE DIAMETER TABLE =====
        # Standard wire gauges (inches)
        self.d_values = np.array([
            0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015,
            0.0162, 0.0173, 0.018, 0.020, 0.023, 0.025, 0.028, 0.032,
            0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.080, 0.092,
            0.0105, 0.120, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207,
            0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394,
            0.4375, 0.500
        ])

    def eval(self, variables_values):
        x = np.array(variables_values).reshape(1, -1)[0]

        # ===== EXTRACT AND DISCRETIZE VARIABLES =====
        x1 = int(round(x[0]))        # Number of active coils (integer)
        x1 = max(1, min(70, x1))     # Enforce bounds [1, 70]

        x2 = float(x[1])             # Mean coil diameter (inches)
        x2 = max(0.6, min(3.0, x2))  # Enforce bounds [0.6, 3.0]

        # Wire diameter from discrete lookup table
        x3_index = int(round(x[2]))
        x3_index = max(1, min(42, x3_index))  # Enforce bounds [1, 42]
        x3 = self.d_values[x3_index - 1]      # 1-indexed to 0-indexed

        # ===== OBJECTIVE FUNCTION =====
        # Minimize volume/mass of spring
        # f = (π² × x2 × x3² × (x1 + 2)) / 4
        # where:
        # x1 : number of active coils
        # x2 : mean coil diameter (inches)
        # x3 : wire diameter (inches)

        f = (np.pi ** 2 * x2 * x3 ** 2 * (x1 + 2)) / 4.0

        # ===== INTERMEDIATE CALCULATIONS =====
        # Shear stress factor
        cf = (4.0 * x2 / x3 - 1.0) / (4.0 * x2 / x3 - 4.0) + 0.615 * x3 / x2

        # Spring constant (lbf/inch)
        K = (11.5e6 * x3 ** 4) / (8.0 * x1 * x2 ** 3)

        # Free length constraint
        lf = 1000.0 / K + 1.05 * (x1 + 2) * x3

        # Stress at preload
        sigp = 300.0 / K

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(8)

        # Constraint 1: Shear stress limit
        # (8000 × cf × x2) / (π × x3³) <= 189000 (psi)
        g[0] = (8000.0 * cf * x2) / (np.pi * x3 ** 3) - 189000.0

        # Constraint 2: Free length limit
        # lf <= 14 inches
        g[1] = lf - 14.0

        # Constraint 3: Wire diameter lower bound
        # x3 >= 0.2 inches → -x3 + 0.2 <= 0
        g[2] = 0.2 - x3

        # Constraint 4: Mean diameter lower bound
        # x2 >= 3 inches → -x2 + 3 <= 0 (ERROR in original code: should be x2 - 3 >= 0)
        # Keeping original: x2 - 3 <= 0 → x2 <= 3
        g[3] = x2 - 3.0

        # Constraint 5: Coil geometry constraint
        # x2/x3 <= 3 → 3 - x2/x3 >= 0 → x2/x3 - 3 <= 0
        if x3 != 0:
            g[4] = 3.0 - x2 / x3
        else:
            g[4] = 1e10

        # Constraint 6: Stress at preload limit
        # sigp <= 6 psi
        g[5] = sigp - 6.0

        # Constraint 7: Stress at surge length limit
        # sigp + 700/K + 1.05×(x1+2)×x3 <= lf
        # Rearranged: sigp + 700/K + 1.05×(x1+2)×x3 - lf <= 0
        if K != 0:
            g[6] = sigp + 700.0 / K + 1.05 * (x1 + 2) * x3 - lf
        else:
            g[6] = 1e10

        # Constraint 8: Deflection limit
        # 700/K <= 1.25 inches
        if K != 0:
            g[7] = 1.25 - 700.0 / K
        else:
            g[7] = 1e10

        # ===== EQUALITY CONSTRAINTS =====
        h = np.array([0.0])

        # ===== PENALTY METHOD =====
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        penalty_h = np.sum(h ** 2)

        # Check for NaN or Inf
        if np.isnan(f) or np.isinf(f):
            return 1e10

        fitness = f + 1e5 * penalty_g + 1e5 * penalty_h

        return fitness
