from Benchmark.__Function import Function
import numpy as np


class RW32(Function):
    def __init__(self, dim):
        x_start = np.array([78.0, 33.0, 27.0, 27.0, 27.0])
        f_x_start = 0
        domain = [
            (78, 102),  # x1  : design parameter 1
            (33, 45),  # x2  : design parameter 2
            (27, 45),  # x3  : design parameter 3
            (27, 45),  # x4  : design parameter 4
            (27, 45),  # x5  : design parameter 5
        ]
        name = "RW32"
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

        # ===== EXTRACT VARIABLES =====
        x1 = x[0]  # Design parameter 1
        x2 = x[1]  # Design parameter 2
        x3 = x[2]  # Design parameter 3
        x4 = x[3]  # Design parameter 4
        x5 = x[4]  # Design parameter 5

        # ===== OBJECTIVE FUNCTION =====
        # Minimize cost of chemical process design
        # f = 5.3578547×x3² + 0.8356891×x1×x5 + 37.293239×x1 - 40792.141

        f = (5.3578547 * x3 ** 2 +
             0.8356891 * x1 * x5 +
             37.293239 * x1 -
             40792.141)

        # ===== INTERMEDIATE PARAMETERS =====
        # Process constraints based on design variables

        # G1: Process constraint 1
        # G1 = 85.334407 + 0.0056858×x2×x5 + 0.0006262×x1×x4 - 0.0022053×x3×x5
        G1 = (85.334407 +
              0.0056858 * x2 * x5 +
              0.0006262 * x1 * x4 -
              0.0022053 * x3 * x5)

        # G2: Process constraint 2
        # G2 = 80.51249 + 0.0071317×x2×x5 + 0.0029955×x1×x2 + 0.0021813×x3²
        G2 = (80.51249 +
              0.0071317 * x2 * x5 +
              0.0029955 * x1 * x2 +
              0.0021813 * x3 ** 2)

        # G3: Process constraint 3
        # G3 = 9.300961 + 0.0047026×x3×x5 + 0.0012547×x1×x3 + 0.0019085×x3×x4
        G3 = (9.300961 +
              0.0047026 * x3 * x5 +
              0.0012547 * x1 * x3 +
              0.0019085 * x3 * x4)

        # ===== INEQUALITY CONSTRAINTS (6) =====
        # Bounds on process variables G1, G2, G3

        g = np.zeros(6)
        g[0] = G1 - 92.0  # G1 ≤ 92
        g[1] = -G1  # G1 ≥ 0 (or G1 ≥ small_value)
        g[2] = G2 - 110.0  # G2 ≤ 110
        g[3] = -G2 + 90.0  # G2 ≥ 90
        g[4] = G3 - 25.0  # G3 ≤ 25
        g[5] = -G3 + 20.0  # G3 ≥ 20

        # ===== EQUALITY CONSTRAINTS =====
        h = np.array([0.0])

        # ===== PENALTY METHOD =====
        penalty_g = np.sum(np.maximum(0, g) ** 2)

        if np.isnan(f) or np.isinf(f) or f > 1e10:
            return 1e10

        fitness = f + 1e5 * penalty_g

        return fitness
