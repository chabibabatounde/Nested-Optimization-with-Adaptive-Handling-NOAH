from Benchmark.Functions.__Function import Function
import numpy as np


class RW21(Function):
    def __init__(self, dim):
        x_start = np.zeros(5)
        f_x_start = 0
        domain = [
            (60.0, 90.0),     # x1 : inner radius of friction surface
            (90.0, 110.0),    # x2 : outer radius of friction surface
            (1.0, 3.0),       # x3 : thickness of one friction lining
            (1000.0, 2500.0), # x4 : normal force
            (0.0, 15.0),      # x5 : number of friction surfaces (minus 1)
        ]
        name = "RW21"
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

        # ===== VARIABLES =====
        x1 = x[0]  # inner radius of friction surface
        x2 = x[1]  # outer radius of friction surface
        x3 = x[2]  # thickness of one friction lining
        x4 = x[3]  # normal force
        x5 = x[4]  # number of friction surfaces (minus 1)

        # ===== PHYSICAL PARAMETERS =====
        Mf = 3.0            # friction torque constant
        Ms = 40.0           # static moment constant
        Iz = 55.0           # moment of inertia
        n = 250.0           # speed (rpm)
        Tmax = 15.0         # maximum stopping time (s)
        s = 1.5             # safety factor
        delta = 0.5         # gap thickness
        Vsrmax = 10.0       # maximum sliding velocity (m/s)
        rho = 0.0000078     # density of friction material (kg/mm³)
        pmax = 1.0          # maximum pressure (MPa)
        mu = 0.6            # coefficient of friction
        Lmax = 30.0         # maximum stack length (mm)
        delR = 20.0         # minimum radial clearance (mm)

        # ===== INTERMEDIATE CALCULATIONS =====
        # Mean radius of friction surface
        Rsr = (2.0 / 3.0) * (x2**3 - x1**3) / (x2**2 * x1**2)

        # Sliding velocity at mean radius
        Vsr = np.pi * Rsr * n / 30.0

        # Friction surface area
        A = np.pi * (x2**2 - x1**2)

        # Pressure on friction surfaces
        Prz = x4 / A

        # Angular velocity
        w = np.pi * n / 30.0

        # Holding torque (braking torque)
        Mh = (2.0 / 3.0) * mu * x4 * (x5 + 1) * (x2**3 - x1**3) / (x2**2 - x1**2)

        # Stopping time
        T = Iz * w / (Mh + Mf)

        # ===== OBJECTIVE FUNCTION =====
        # Minimize mass of friction material
        f = np.pi * (x2**2 - x1**2) * x3 * (x5 + 1) * rho

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(8)
        g[0] = -x2 + x1 + delR                    # radial clearance constraint
        g[1] = (x5 + 1) * (x3 + delta) - Lmax     # stack length constraint
        g[2] = Prz - pmax                         # maximum pressure constraint
        g[3] = Prz * Vsr - pmax * Vsrmax          # pressure-velocity constraint
        g[4] = Vsr - Vsrmax                       # sliding velocity constraint
        g[5] = T - Tmax                           # stopping time constraint
        g[6] = s * Ms - Mh                        # holding torque constraint
        g[7] = -T                                 # stopping time must be positive

        # ===== EQUALITY CONSTRAINTS =====
        # Aucune contrainte d'égalité
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * penalty_g

        return fitness
