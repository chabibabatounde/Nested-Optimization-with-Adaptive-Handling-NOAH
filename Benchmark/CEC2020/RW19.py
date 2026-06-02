from Benchmark.Functions.__Function import Function
import numpy as np


class RW19(Function):
    def __init__(self, dim):
        x_start = np.zeros(4)
        f_x_start = 0
        domain = [
            (0.125, 5.0),    # x1 : thickness of weld (h)
            (0.1, 10.0),     # x2 : length of weld (l)
            (0.1, 10.0),     # x3 : height of beam (H)
            (0.125, 5.0),    # x4 : thickness of beam (t)
        ]
        name = "RW19"
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

        # ===== VARIABLES =====
        x1 = x[0]  # h    : thickness of weld
        x2 = x[1]  # l    : length of weld
        x3 = x[2]  # H    : height of beam
        x4 = x[3]  # t    : thickness of beam

        # ===== PHYSICAL CONSTANTS =====
        P = 6000.0          # load (lbs)
        L = 14.0            # length of beam (inches)
        delta_max = 0.25    # maximum deflection (inches)
        E = 30e6            # Young's modulus (psi)
        G = 12e6            # Shear modulus (psi)
        T_max = 13600.0     # maximum shear stress (psi)
        sigma_max = 30000.0 # maximum normal stress (psi)

        # ===== OBJECTIVE FUNCTION =====
        f = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14 + x2)

        # ===== INTERMEDIATE CALCULATIONS =====
        # Buckling load
        Pc = (4.013 * E * np.sqrt(x3**2 * x4**6 / 30.0) / L**2
              * (1 - x3 / (2 * L) * np.sqrt(E / (4 * G))))

        # Bending stress in beam
        sigma = 6 * P * L / (x4 * x3**2)

        # Deflection of beam
        delta = 6 * P * L**3 / (E * x3**2 * x4)

        # Second moment of area of weld
        J = 2 * (np.sqrt(2) * x1 * x2 * (x2**2 / 4 + (x1 + x3)**2 / 4))

        # Distance from centroid to outer fiber
        R = np.sqrt(x2**2 / 4 + (x1 + x3)**2 / 4)

        # Bending moment
        M = P * (L + x2 / 2)

        # Shear stress due to bending
        ttt = M * R / J

        # Primary shear stress
        tt = P / (np.sqrt(2) * x1 * x2)

        # Combined shear stress
        t = np.sqrt(tt**2 + 2 * tt * ttt * x2 / (2 * R) + ttt**2)

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(5)
        g[0] = t - T_max                    # shear stress constraint
        g[1] = sigma - sigma_max            # bending stress constraint
        g[2] = x1 - x4                      # geometric constraint
        g[3] = delta - delta_max            # deflection constraint
        g[4] = P - Pc                       # buckling constraint

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
