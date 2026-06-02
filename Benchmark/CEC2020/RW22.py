from Benchmark.Functions.__Function import Function
import numpy as np


class RW22(Function):
    def __init__(self, dim):
        x_start = np.zeros(9)
        f_x_start = 0
        domain = [
            (12.0, 60.0),     # x1 : N1 (number of teeth, gear 1)
            (12.0, 60.0),     # x2 : N2 (number of teeth, gear 2)
            (12.0, 60.0),     # x3 : N3 (number of teeth, gear 3)
            (12.0, 60.0),     # x4 : N4 (number of teeth, gear 4)
            (12.0, 60.0),     # x5 : N5 (number of teeth, gear 5)
            (12.0, 60.0),     # x6 : N6 (number of teeth, gear 6)
            (1.0, 3.0),       # x7 : index for pressure angle p in {3, 4, 5}
            (1.0, 6.0),       # x8 : index for module m1 in {1.75, 2.0, 2.25, 2.5, 2.75, 3.0}
            (1.0, 6.0),       # x9 : index for module m2 in {1.75, 2.0, 2.25, 2.5, 2.75, 3.0}
        ]
        name = "RW22"
        max_dimension = 9

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

        # ===== PARAMETER INITIALIZATION =====
        # Round all variables to integers
        x = np.round(np.abs(x)).astype(int)

        # Gear teeth counts
        N1 = x[0]
        N2 = x[1]
        N3 = x[2]
        N4 = x[3]
        N5 = x[4]
        N6 = x[5]

        # Pressure angle index (1-based, map to actual values)
        Pind = np.array([3, 4, 5])
        idx_p = int(np.clip(x[6], 1, 3)) - 1
        p = Pind[idx_p]

        # Module index (1-based, map to actual values)
        mind = np.array([1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
        idx_m1 = int(np.clip(x[7], 1, 6)) - 1
        idx_m2 = int(np.clip(x[8], 1, 6)) - 1
        m1 = mind[idx_m1]
        m2 = mind[idx_m2]

        # ===== OBJECTIVE FUNCTION =====
        # Compute gear ratios
        i1 = N6 / N4
        i01 = 3.11

        # Avoid division by zero
        if N1 * N3 * (N6 - N4) == 0:
            i2 = 1e10
        else:
            i2 = N6 * (N1 * N3 + N2 * N4) / (N1 * N3 * (N6 - N4))
        i02 = 1.84

        # Reverse ratio
        if N1 * N3 == 0:
            iR = 1e10
        else:
            iR = -(N2 * N6) / (N1 * N3)
        i0R = -3.11

        # Objective: minimize maximum deviation from target ratios
        f = np.max([i1 - i01, i2 - i02, iR - i0R])

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(10)

        Dmax = 220.0
        dlt22 = 0.5
        dlt33 = 0.5
        dlt55 = 0.5
        dlt35 = 0.5
        dlt34 = 0.5
        dlt56 = 0.5

        # Compute angle beta (center distance angle)
        numerator = (N6 - N3)**2 + (N4 + N5)**2 - (N3 + N5)**2
        denominator = 2.0 * (N6 - N3) * (N4 + N5)

        if denominator == 0:
            beta = 0.0
        else:
            cos_beta = numerator / denominator
            cos_beta = np.clip(cos_beta, -1.0, 1.0)  # Ensure valid range for arccos
            beta = np.arccos(cos_beta)

        # Constraints
        g[0] = m2 * (N6 + 2.5) - Dmax
        g[1] = m1 * (N1 + N2) + m1 * (N2 + 2) - Dmax
        g[2] = m2 * (N4 + N5) + m2 * (N5 + 2) - Dmax
        g[3] = np.abs(m1 * (N1 + N2) - m2 * (N6 - N3)) - m1 - m2
        g[4] = -((N1 + N2) * np.sin(np.pi / p) - N2 - 2 - dlt22)
        g[5] = -((N6 - N3) * np.sin(np.pi / p) - N3 - 2 - dlt33)
        g[6] = -((N4 + N5) * np.sin(np.pi / p) - N5 - 2 - dlt55)

        # Constraint 8 : check if beta is real
        if np.isreal(beta) and beta == beta:  # beta is real and not NaN
            g[8] = ((N3 + N5 + 2 + dlt35)**2
                    - ((N6 - N3)**2 + (N4 + N5)**2
                       - 2 * (N6 - N3) * (N4 + N5) * np.cos(2 * np.pi / p - beta)))
        else:
            g[8] = 1e6  # Penalize infeasible angle

        g[7] = g[8]  # Reorder (constraint 8 -> g[7])

        g[9] = -(N6 - 2 * N3 - N4 - 4 - 2 * dlt34)
        g[8] = -(N6 - N4 - 2 * N5 - 4 - 2 * dlt56)

        # ===== EQUALITY CONSTRAINT =====
        # Modulo constraint: (N6 - N4) mod p = 0
        h = np.array([float(np.abs((N6 - N4) % p))])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_h = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * (penalty_h + penalty_g)

        return fitness
