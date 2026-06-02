from Benchmark.Functions.__Function import Function
import numpy as np


class RW14(Function):
    def __init__(self, dim):
        x_start = np.zeros(10)
        f_x_start = 0
        domain = [
            (1, 3),  # N1 (entier)
            (1, 3),  # N2 (entier)
            (1, 3),  # N3 (entier)
            (250, 2500),  # V1 (continu)
            (250, 2500),  # V2 (continu)
            (250, 2500),  # V3 (continu)
            (6, 20),  # TL1 (continu)
            (6, 20),  # TL2 (continu)
            (40, 700),  # B1 (continu)
            (10, 450),  # B2 (continu)
        ]
        name = "RW14"
        max_dimension = 10

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

        # ===== CONSTANTES =====
        S = np.array([[2, 3, 4],
                      [4, 6, 3]])
        t = np.array([[8, 20, 8],
                      [16, 4, 4]])
        H = 6000
        alp = 250
        beta = 0.6
        Q1 = 40000
        Q2 = 20000

        # ===== VARIABLES DE DÉCISION =====
        N1 = round(x[0])  # entier
        N2 = round(x[1])  # entier
        N3 = round(x[2])  # entier
        V1 = x[3]
        V2 = x[4]
        V3 = x[5]
        TL1 = x[6]
        TL2 = x[7]
        B1 = x[8]
        B2 = x[9]

        # ===== OBJECTIVE FUNCTION =====
        f = alp * (N1 * V1 ** beta + N2 * V2 ** beta + N3 * V3 ** beta)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(10)
        g[0] = Q1 * TL1 / B1 + Q2 * TL2 / B2 - H
        g[1] = S[0, 0] * B1 + S[1, 0] * B2 - V1
        g[2] = S[0, 1] * B1 + S[1, 1] * B2 - V2
        g[3] = S[0, 2] * B1 + S[1, 2] * B2 - V3
        g[4] = t[0, 0] - N1 * TL1
        g[5] = t[0, 1] - N2 * TL1
        g[6] = t[0, 2] - N3 * TL1
        g[7] = t[1, 0] - N1 * TL2
        g[8] = t[1, 1] - N2 * TL2
        g[9] = t[1, 2] - N3 * TL2

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
