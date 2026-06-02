from Benchmark.__Function import Function
import numpy as np


class RW6(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        # Bornes pour les 38 variables du problème RC06
        # x1-x4 : flux d'entrée (max 300 selon h1)
        # x5-x20 : flux intermédiaires
        # x21-x38 : fractions (entre 0 et 1)
        domain = [
            (0, 90),    # x1
            (0, 150),   # x2
            (0, 90),    # x3
            (0, 150),   # x4
            (0, 90),    # x5
            (0, 90),    # x6
            (0, 90),    # x7
            (0, 90),    # x8
            (0, 90),    # x9
            (0, 90),    # x10
            (0, 90),    # x11
            (0, 90),    # x12
            (0, 150),   # x13
            (0, 150),   # x14
            (0, 150),   # x15
            (0, 150),   # x16
            (0, 150),   # x17
            (0, 150),   # x18
            (0, 150),   # x19
            (0, 150),   # x20
            (0, 1),     # x21
            (0, 1),     # x22
            (0, 1),     # x23
            (0, 1),     # x24
            (0, 1),     # x25
            (0, 1),     # x26
            (0, 1),     # x27
            (0, 1),     # x28
            (0, 1),     # x29
            (0, 1),     # x30
            (0, 1),     # x31
            (0, 1),     # x32
            (0, 1),     # x33
            (0, 1),     # x34
            (0, 1),     # x35
            (0, 1),     # x36
            (0, 1),     # x37
            (0, 1),     # x38
        ]
        name = "RW6"
        max_dimension = 38  # IMPORTANT: ce problème utilise 38 variables

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
        x = np.array(variables_values).reshape(1, -1)

        # Variables (indexation 1-based comme MATLAB, on adapte avec -1)
        x1  = x[0, 0]
        x2  = x[0, 1]
        x3  = x[0, 2]
        x4  = x[0, 3]
        x5  = x[0, 4]
        x6  = x[0, 5]
        x7  = x[0, 6]
        x8  = x[0, 7]
        x9  = x[0, 8]
        x10 = x[0, 9]
        x11 = x[0, 10]
        x12 = x[0, 11]
        x13 = x[0, 12]
        x14 = x[0, 13]
        x15 = x[0, 14]
        x16 = x[0, 15]
        x17 = x[0, 16]
        x18 = x[0, 17]
        x19 = x[0, 18]
        x20 = x[0, 19]
        x21 = x[0, 20]
        x22 = x[0, 21]
        x23 = x[0, 22]
        x24 = x[0, 23]
        x25 = x[0, 24]
        x26 = x[0, 25]
        x27 = x[0, 26]
        x28 = x[0, 27]
        x29 = x[0, 28]
        x30 = x[0, 29]
        x31 = x[0, 30]
        x32 = x[0, 31]
        x33 = x[0, 32]
        x34 = x[0, 33]
        x35 = x[0, 34]
        x36 = x[0, 35]
        x37 = x[0, 36]
        x38 = x[0, 37]

        # ===== OBJECTIVE FUNCTION =====
        f = 0.9979 + 0.00432 * x5 + 0.01517 * x13

        # ===== INEQUALITY CONSTRAINTS =====
        g = np.array([0.0])  # aucune contrainte d'inégalité

        # ===== EQUALITY CONSTRAINTS (h = 0) =====
        h = np.zeros(32)

        h[0]  = x1 + x2 + x3 + x4 - 300
        h[1]  = x6 - x7 - x8
        h[2]  = x9 - x10 - x11 - x12
        h[3]  = x14 - x15 - x16 - x17
        h[4]  = x18 - x19 - x20
        h[5]  = x5 * x21 - x6 * x22 - x9 * x23
        h[6]  = x5 * x24 - x6 * x25 - x9 * x26
        h[7]  = x5 * x27 - x6 * x28 - x9 * x29
        h[8]  = x13 * x30 - x14 * x31 - x18 * x32
        h[9]  = x13 * x33 - x14 * x34 - x18 * x35
        h[10] = x13 * x36 - x14 * x37 - x18 * x38
        h[11] = (1/3) * x1 + x15 * x31 - x5 * x21
        h[12] = (1/3) * x1 + x15 * x34 - x5 * x24
        h[13] = (1/3) * x1 + x15 * x37 - x5 * x27
        h[14] = (1/3) * x2 + x10 * x23 - x13 * x30
        h[15] = (1/3) * x2 + x10 * x26 - x13 * x33
        h[16] = (1/3) * x2 + x10 * x29 - x13 * x36
        h[17] = (1/3) * x3 + x7 * x22 + x11 * x23 + x16 * x31 + x19 * x32 - 30
        h[18] = (1/3) * x3 + x7 * x25 + x11 * x26 + x16 * x34 + x19 * x35 - 50
        h[19] = (1/3) * x3 + x7 * x28 + x11 * x29 + x16 * x37 + x19 * x38 - 30
        h[20] = x21 + x24 + x27 - 1
        h[21] = x22 + x25 + x28 - 1
        h[22] = x23 + x26 + x29 - 1
        h[23] = x30 + x33 + x36 - 1
        h[24] = x31 + x34 + x37 - 1
        h[25] = x32 + x35 + x38 - 1
        h[26] = x25
        h[27] = x28
        h[28] = x23
        h[29] = x37
        h[30] = x32
        h[31] = x35

        tol = 1e-4
        penalty = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        fitness = f + 1e5 * penalty  # plus souple
        # fitness = f + 1e8 * penalty  # plus strict

        return fitness
