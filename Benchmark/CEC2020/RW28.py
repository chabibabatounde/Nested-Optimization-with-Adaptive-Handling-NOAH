from Benchmark.__Function import Function
import numpy as np


class RW28(Function):
    def __init__(self, dim):
        x_start = np.array([71.5, 12.7, 8, 0.4, 0.4, 0.515, 0.515, 0.6, 0.015, 0.6])
        f_x_start = 0
        domain = [
            (20.0, 150.0),     # x1  : Dm (mean diameter, mm)
            (5.0, 30.0),       # x2  : Db (ball diameter, mm)
            (1, 50),           # x3  : Z (number of rolling elements)
            (0.4, 0.7),        # x4  : fi (inner raceway conformity)
            (0.4, 0.7),        # x5  : fo (outer raceway conformity)
            (0.5, 2.0),        # x6  : KDmin (minimum diameter factor)
            (0.5, 2.0),        # x7  : KDmax (maximum diameter factor)
            (0.1, 1.0),        # x8  : eps (clearance factor)
            (0.001, 0.1),      # x9  : e (eccentricity factor)
            (0.4, 0.9),        # x10 : chi (width factor)
        ]
        name = "RW28"
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

        # ===== EXTRACT VARIABLES =====
        Dm = x[0]      # Mean diameter (mm)
        Db = x[1]      # Ball diameter (mm)
        Z = x[2]       # Number of rolling elements
        fi = x[3]      # Inner raceway conformity
        fo = x[4]      # Outer raceway conformity
        KDmin = x[5]   # Minimum diameter factor
        KDmax = x[6]   # Maximum diameter factor
        eps = x[7]     # Clearance factor
        e = x[8]       # Eccentricity factor
        chi = x[9]     # Width factor

        # ===== FIXED PARAMETERS =====
        D = 160.0      # Outer raceway diameter (mm)
        d = 90.0       # Inner raceway diameter (mm)
        Bw = 30.0      # Bearing width (mm)

        # ===== INTERMEDIATE CALCULATIONS =====
        T = D - d - 2.0 * Db

        # Contact angle calculation
        numerator = ((D - d) * 0.5 - 0.75 * T) ** 2 + (0.5 * D - 0.25 * T - Db) ** 2 - (0.5 * d + 0.25 * T) ** 2
        denominator = 2.0 * (0.5 * (D - d) - 0.75 * T) * (0.5 * D - 0.25 * T - Db)

        if abs(denominator) < 1e-10:
            phi_o = 0.0
        else:
            cos_phi_o = numerator / denominator
            # Clamp to valid range [-1, 1]
            cos_phi_o = np.clip(cos_phi_o, -1.0, 1.0)
            phi_o = 2.0 * np.pi - 2.0 * np.arccos(cos_phi_o)

        # Diameter ratio
        gamma = Db / Dm

        # ===== DYNAMIC CAPACITY COEFFICIENT =====
        # Complex formula for bearing capacity
        term1 = 1.04 * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72
        term2 = (fi * (2.0 * fo - 1.0)) / (fo * (2.0 * fi - 1.0))
        term2 = np.clip(term2, 1e-6, None)  # Avoid division by zero
        term3 = term1 * (term2 ** 0.41)
        term4 = (1.0 + term3) ** (10.0 / 3.0)
        term5 = term4 ** (-0.3)

        numerator_fc = gamma ** 0.3 * (1.0 - gamma) ** 1.39 / (1.0 + gamma) ** (1.0 / 3.0)
        denominator_fc = (2.0 * fi) / (2.0 * fi - 1.0)

        fc = 37.91 * term5 * (numerator_fc / denominator_fc) ** 0.41

        # ===== OBJECTIVE FUNCTION =====
        # Minimize bearing dynamic load rating (dynamic capacity)
        if Db > 25.4:
            f = 3.647 * fc * (Z ** (2.0 / 3.0)) * (Db ** 1.4)
        else:
            f = fc * (Z ** (2.0 / 3.0)) * (Db ** 1.8)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(9)

        # Constraint 1: Rolling element spacing
        if phi_o < np.pi:
            sin_term = np.sin(Db / Dm)
            if sin_term > 0:
                g[0] = Z - 1.0 - phi_o / (2.0 * np.arcsin(Db / Dm))
            else:
                g[0] = 1e6  # Constraint violated
        else:
            g[0] = 1e6

        # Constraint 2: Minimum diameter constraint
        g[1] = KDmin * (D - d) - 2.0 * Db

        # Constraint 3: Maximum diameter constraint
        g[2] = 2.0 * Db - KDmax * (D - d)

        # Constraint 4: Width constraint
        g[3] = chi * Bw - Db

        # Constraint 5: Mean diameter lower bound
        g[4] = 0.5 * (D + d) - Dm

        # Constraint 6: Mean diameter upper bound
        g[5] = Dm - (0.5 + e) * (D + d)

        # Constraint 7: Clearance constraint
        g[6] = eps * Db - 0.5 * (D - Dm - Db)

        # Constraint 8: Inner conformity constraint
        g[7] = 0.515 - fi

        # Constraint 9: Outer conformity constraint
        g[8] = 0.515 - fo

        # ===== EQUALITY CONSTRAINTS =====
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
